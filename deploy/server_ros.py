import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading, time, socket, pickle, struct, queue, bisect
from scipy.spatial.transform import Rotation


class ROSCameraServer(Node):
    """ROS2 node that subscribes to compressed RGB, depth, and pose topics,
    synchronizes them using ApproximateTimeSynchronizer, and exposes a state_dict()
    and get_inputs() API identical to the original Server class.
    """

    def __init__(
        self,
        color_topic="/camera/color/image_raw/compressed",
        depth_topic="/camera/depth/image_raw/compressed",
        pose_topic="/camera_pose",
        queue_size=10,
        sync_slop=0.05,
    ):
        super().__init__("ros_camera_server")

        self.record_enable = threading.Event()
        self.sample_time = -1
        self.rgbd_queue = queue.Queue()
        self.pose_queue = queue.Queue()
        self.pose_timestamps = []
        self.extrinsic_stream = []

        # Subscribers
        self.color_sub = Subscriber(self, CompressedImage, color_topic)
        self.depth_sub = Subscriber(self, CompressedImage, depth_topic)
        self.pose_sub = Subscriber(self, PoseStamped, pose_topic)

        self.cv_bridge = CvBridge()

        # Approximate time synchronizer for all 3 streams
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.pose_sub],
            queue_size=queue_size,
            slop=sync_slop,
        )
        self.ts.registerCallback(self.sync_callback)

        self.record_enable.set()
        self.get_logger().info("[*] ROSCameraServer initialized and listening for messages.")

    # ------------------------------------------------------------------
    # Synchronizer callback
    # ------------------------------------------------------------------
    def sync_callback(self, color_msg, depth_msg, pose_msg):
        """Triggered whenever all 3 topics have approximately matching timestamps."""
        try:
            color_img = self.cv_bridge.compressed_imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
            depth_img = self.cv_bridge.compressed_imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # Pose extraction
            p = pose_msg.pose.position
            q = pose_msg.pose.orientation
            if pose_msg.header.frame_id == color_msg.header.frame_id:
                pose_tuple = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            else:
                # pose is the transformation from camera to world/map
                T_cw = np.eye(4)
                T_cw[:3, 3] = [p.x, p.y, p.z]
                T_cw[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T_wc = np.linalg.inv(T_cw)
                q = Rotation.from_matrix(T_wc[:3, :3]).as_quat()
                pose_tuple = (T_wc[:3, 3].tolist() + q.tolist())

            ts = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9

            # Store data in queues
            self.rgbd_queue.put_nowait((ts, color_img, depth_img))
            self.pose_queue.put_nowait((ts, pose_tuple))
        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")


class Server(object):
    """Camera server using ROS2 compressed image and pose topics."""
    def __init__(self, with_pose=True, ts_thresh=0.01):
        self.ts_thresh = ts_thresh
        self.with_pose = with_pose
        self.pose_timestamps = []
        self.extrinsic_stream = []
        self.rgbd_queue = queue.Queue()
        self.pose_queue = queue.Queue()
        self.record_enable = threading.Event()
        self.sample_time = -1

        rclpy.init()

        self.executor = MultiThreadedExecutor()

        # Start camera node in its own spin thread
        self.camera_node = ROSCameraServer()
        self.executor.add_node(self.camera_node)

        self.exec_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self.exec_thread.start()

        # Link queues
        self.rgbd_queue = self.camera_node.rgbd_queue
        self.pose_queue = self.camera_node.pose_queue

        self.record_enable.set()

        # Communication thread (TCP server)
        self.com_process = threading.Thread(
            target=self.get_server(port=8888),
            args=(self.record_enable,),
            name="com",
            daemon=True,
        )
        self.com_process.start()

    def close(self):
        self.record_enable.clear()

    def get_server(self, port=9000):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_name = "localhost"
        host_ip = socket.gethostbyname(host_name)
        print(f"[*] HOST IP: {host_ip}")
        socket_address = (host_ip, port)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(socket_address)
        server_socket.listen(5)

        def server(record_enable):
            while record_enable.is_set():
                print(f"[*] WAITING FOR CONNECTION AT: {socket_address}")
                client_socket, addr = server_socket.accept()
                print(f"[*] GOT CONNECTION FROM: {addr}")
                while client_socket:
                    res = self.get_inputs()
                    if res is None:
                        continue
                    data = pickle.dumps(res)
                    print(f"[*] Sending data with timestamp: {res['camera_time']}")
                    try:
                        client_socket.send(struct.pack("Q", len(data)) + data)
                    except (ConnectionResetError, BrokenPipeError) as e:
                        print("[*] CONNECTION CLOSED")
                        break
                client_socket.close()

        return server

    def state_dict(self):
        data = list(self.rgbd_queue.queue)
        self.rgbd_queue.queue.clear()
        res = {
            "camera_timestamps": [d[0] for d in data],
            "rgb_stream": [d[1] for d in data],
            "depth_stream": [d[2] for d in data],
        }
        if self.with_pose:
            data = list(self.pose_queue.queue)
            self.pose_queue.queue.clear()
            if len(data) == 0:
                return None
            res["pose_timestamps"] = self.pose_timestamps + [d[0] for d in data]
            res["extrinsic_stream"] = self.extrinsic_stream + [d[1] for d in data]
            self.res = res["extrinsic_stream"][-1]
        return res

    def get_inputs(self):
        state = self.state_dict()
        if state is None or len(state["camera_timestamps"]) == 0:
            return None

        cam_ts = state["camera_timestamps"][-1]
        if cam_ts == self.sample_time:
            return None

        res = {
            "camera_time": cam_ts,
            "rgb": state["rgb_stream"][-1],
            "depth": state["depth_stream"][-1],
        }

        if self.with_pose:
            if len(state["pose_timestamps"]) == 0:
                return None
            pose_idx = min(len(state["pose_timestamps"]) - 1, bisect.bisect_right(state["pose_timestamps"], cam_ts))
            if abs(state["pose_timestamps"][pose_idx] - cam_ts) < abs(state["pose_timestamps"][pose_idx - 1] - cam_ts):
                if abs(state["pose_timestamps"][pose_idx] - cam_ts) > self.ts_thresh:
                    print(f"pose ts too far! pose:{state['pose_timestamps'][pose_idx]} and cam: {cam_ts}")
                    self.sample_time = cam_ts
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx]
                res["extrinsic"] = state["extrinsic_stream"][pose_idx]
            else:
                if abs(state["pose_timestamps"][pose_idx - 1] - cam_ts) > self.ts_thresh:
                    print(f"pose ts too far! pose:{state['pose_timestamps'][pose_idx - 1]} and cam: {cam_ts}")
                    self.sample_time = cam_ts
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx - 1]
                res["extrinsic"] = state["extrinsic_stream"][pose_idx - 1]

            self.extrinsic_stream = state["extrinsic_stream"][pose_idx + 1:]
            self.pose_timestamps = state["pose_timestamps"][pose_idx + 1:]

        self.sample_time = cam_ts
        return res


def main():
    server = Server(with_pose=True)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
