"""Export a YOLOv8 detector to INT8 ONNX for cheap CPU-only inference.

The signature detector ships as FP32 PyTorch weights; quantising to INT8
roughly quarters the model size and keeps per-document cost in the
fraction-of-a-cent range on CPU.

Usage:
    python scripts/export_int8_yolo.py                       # yolov8n.pt
    python scripts/export_int8_yolo.py --weights yolov8s.pt
    python scripts/export_int8_yolo.py --weights path/to/signature-yolov8.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLOv8 weights to export (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size used for the export (default: 640)",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = model.export(format="onnx", int8=True, imgsz=args.imgsz)
    size_mb = Path(exported).stat().st_size / 1e6
    print(f"INT8 ONNX written to {exported} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
