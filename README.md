<p align="center">

  <h1 align="center">OpenFusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation </h1>
  <p align="center">
    <a href="https://kashu7100.github.io/"><strong>Kashu Yamazaki</strong></a>
    ·
    <a href=""><strong>Taisei Hanyu</strong></a>
    ·
    <a href="https://vhvkhoa.github.io/"><strong>Khoa Vo</strong></a>
    ·
    <a href="https://phamtrongthang123.github.io/"><strong>Thang Pham</strong></a>
    ·
    <a href=""><strong>Minh Tran</strong></a>
    <br>
    <a href=""><strong>Gianfranco Doretto</strong></a>
    ·
    <a href=""><strong>Anh Nguyen</strong></a>
    ·
    <a href=""><strong>Ngan Le</strong></a>
  </p>

  <h4 align="center"><a href="https://arxiv.org/pdf/2310.03923.pdf">Paper</a> | <a href="https://arxiv.org/abs/2310.03923">arXiv</a> | <a href="https://uark-aicv.github.io/OpenFusion/">Project Page</a></h4>
  <div align="center"></div>
</p>

<p align="center">
<img src="assets/pipeline.png" width="80%">
</p>

**TL;DR**: *Open-Fusion* builds an open-vocabulary 3D queryable scene from a sequence of posed RGB-D images in real-time.

## Getting Started 🏁
### System Requirements
- Ubuntu 20.04
- 10GB+ VRAM (~ 5 GB for SEEM and 2.5 GB ~ for TSDF) - for a large scene, it may require more memory
- *Azure Kinect, Intel T265* (for real-world data)

### Environment Setup

Please build a Docker image from the Dockerfile. Do not forget to export the following environment variables (`REGISTRY_NAME` and `IMAGE_NAME`) as we use them in the `tools/*.sh` scripts:

```bash
export REGISTRY_NAME=<your-registry-name>
export IMAGE_NAME=<your-image-name>
docker build -t $REGISTRY_NAME/$IMAGE_NAME -f docker/Dockerfile .
```

### Data Preparation

#### ICL and Replica

You can run the following script to download the ICL and Replica datasets:

```bash
bash tools/download.sh --data icl replica
```

This script will create a folder `./sample` and download the datasets into the folder.

#### ScanNet
For ScanNet, please follow the instructions in [ScanNet](https://github.com/ScanNet/ScanNet/tree/master). Once you have the dataset downloaded, you can run the following script to prepare the data (example for scene `scene0001_00`):

```bash
python tools/prepare_scene.py --filename scene0001_00.sens --output_path sample/scannet/scene0001_00
```

### Model Preparation

Please download the pretrained weight for SEEM from [huggingface.co](https://huggingface.co/xdecoder/SEEM/blob/main/seem_focall_v1.pt) or [hf-mirror.com](https://hf-mirror.com/xdecoder/SEEM/blob/main/seem_focall_v1.pt) and put it in as `openfusion/zoo/xdecoder_seem/checkpoints/seem_focall_v1.pt`.

### Run OpenFusion

You can run OpenFusion using `tools/run.sh` as follows:

```bash
bash tools/run.sh --data $DATASET --scene $SCENE
```

Options:
- `--data`: dataset to use (e.g., `icl`)
- `--scene`: scene to use (e.g., `kt0`)
- `--frames`: number of frames to use (default: -1)
- `--live`: run with live monitor (default: False)
- `--stream`: run with data stream from camera server (default: False)



If you want to run OpenFusion with camera stream, please run the following command first on the machine with *Azure Kinect and Intel T265* connected:

```bash
python deploy/server.py
```
Please refer to [this](deploy/README.md) for more details.


<p align="center">
<img src="assets/output.gif">
</p>


## Acknowledgement 🙇

- [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once): VLFM we used to extract region based features
- [Open3D](https://github.com/isl-org/Open3D): GPU accelerated 3D library for the base TSDF implementation

## Citation 🙏

If you find this work helpful, please consider citing our work as:

```bibtex
@inproceedings{yamazaki2024open,
  title={Open-fusion: Real-time open-vocabulary 3d mapping and queryable scene representation},
  author={Yamazaki, Kashu and Hanyu, Taisei and Vo, Khoa and Pham, Thang and Tran, Minh and Doretto, Gianfranco and Nguyen, Anh and Le, Ngan},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={9411--9417},
  year={2024},
  organization={IEEE}
}
```

## Contact 📧

Please create an issue on this repository for questions, comments and reporting bugs. Send an email to [Kashu Yamazaki](https://kashu7100.github.io/) for other inquiries.
