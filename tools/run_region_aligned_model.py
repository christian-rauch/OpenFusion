#!/usr/bin/env python3
"""Run RegionAlignedModel on RGB images and compare text query embeddings."""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from openfusion.zoo.base_model import RegionAlignedModel


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenFusion's RegionAlignedModel on colour images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--model",
        default="seem",
        choices=RegionAlignedModel.model_names(),
        help="RegionAlignedModel backend to load.",
    )
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "semseg", "emb"],
        help="Image encoding mode passed to RegionAlignedModel.encode_image.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Resize model input. Defaults to the model metadata value.",
    )
    parser.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for the text embedding matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("region_aligned_outputs.pt"),
        help="Path for the torch-saved result dictionary.",
    )
    return parser.parse_args()


def expand_images(inputs):
    paths = []
    for item in inputs:
        matches = [Path(p) for p in glob.glob(item)]
        candidates = matches if matches else [Path(item)]
        for path in candidates:
            if path.is_dir():
                paths.extend(
                    sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
                )
            elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)

    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def pairwise_distances(embeddings, metric):
    if embeddings.numel() == 0:
        return torch.empty(0, 0)
    if metric == "cosine":
        normalized = F.normalize(embeddings, dim=-1)
        return 1.0 - normalized @ normalized.T
    return torch.cdist(embeddings, embeddings, p=2)


def tensor_summary(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {k: tensor_summary(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_summary(v) for v in value]
    return value


def detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(v) for v in value]
    return value


queries_nyu40 = [
    # "unlabeled",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floormat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "showercurtain",
    "box",
    "whiteboard",
    "person",
    "nightstand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    # "otherstructure",
    # "otherfurniture",
    # "otherprop",
]

@torch.no_grad()
def main():
    args = parse_args()
    image_paths = expand_images(args.images)
    if not image_paths:
        raise FileNotFoundError("No colour image files matched the provided inputs.")

    model_kwargs = {}
    if args.input_size is not None:
        model_kwargs["input_size"] = args.input_size

    model = RegionAlignedModel(args.model, **model_kwargs)

    text_embeddings = model.encode_text(queries_nyu40)
    print("text_embeddings", text_embeddings.shape)

    for path in image_paths:
        rgb_images = [load_rgb(path)]
        image_outputs = model.encode_image(rgb_images, mode=args.mode)[0]
        # print("image_outputs", image_outputs)
        for k in image_outputs.keys():
            print(">>> img_embed shape", k, image_outputs[k].shape, flush=True)

        distance_matrix = None

        # TODO:
        distance_matrix = pairwise_distances(text_embeddings.float(), args.distance)

        result = {
            "image_paths": [str(path) for path in image_paths],
            "mode": args.mode,
            "image_outputs": detach_to_cpu(image_outputs),
            "queries": queries_nyu40,
            "text_embeddings": detach_to_cpu(text_embeddings),
            "distance_metric": args.distance,
            "distance_matrix": detach_to_cpu(distance_matrix),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.output)

        # print(json.dumps(
        #     {
        #         "output": str(args.output),
        #         "images": [str(path) for path in image_paths],
        #         "mode": args.mode,
        #         "image_output_summary": tensor_summary(image_outputs),
        #         "queries": queries,
        #         "distance_metric": args.distance if queries else None,
        #         "distance_matrix": distance_matrix.detach().cpu().tolist()
        #         if distance_matrix is not None
        #         else None,
        #     },
        #     indent=2,
        # ))


if __name__ == "__main__":
    main()
