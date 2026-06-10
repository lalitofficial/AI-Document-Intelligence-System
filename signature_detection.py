import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from ultralytics import YOLO


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_models(device: str, token: Optional[str]):
    sig_model_path = hf_hub_download(
        repo_id="tech4humans/yolov8s-signature-detector",
        filename="yolov8s.pt",
        token=token,
    )
    model_sig = YOLO(sig_model_path)
    model_sig.to(device)

    processor = AutoImageProcessor.from_pretrained(
        "Ooredoo-Group/ooredoo-stamp-detection",
        token=token,
    )
    model_stamp = AutoModelForObjectDetection.from_pretrained(
        "Ooredoo-Group/ooredoo-stamp-detection",
        token=token,
    ).eval()

    if device == "cuda":
        model_stamp.to(device)

    return model_sig, processor, model_stamp


def _load_font(font_size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    pad = 4
    text_w, text_h = _text_size(draw, text, font)
    rect = (x, y, x + text_w + pad * 2, y + text_h + pad * 2)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def _annotate_with_supervision(
    image_bgr: np.ndarray,
    signature_detections: List[Dict[str, Any]],
    stamp_detections: List[Dict[str, Any]],
) -> Optional[np.ndarray]:
    try:
        import supervision as sv
    except Exception:
        return None

    detections: List[List[float]] = []
    for d in signature_detections:
        detections.append([*d["bbox_xyxy"], d["confidence"], 0])
    for d in stamp_detections:
        detections.append([*d["bbox_xyxy"], d["confidence"], 1])

    if not detections:
        return image_bgr

    det = sv.Detections(
        xyxy=np.array([d[:4] for d in detections]),
        confidence=np.array([d[4] for d in detections]),
        class_id=np.array([d[5] for d in detections]),
    )

    labels = [
        f"{'signature' if c == 0 else 'stamp'} {conf:.2f}"
        for c, conf in zip(det.class_id, det.confidence)
    ]

    try:
        annotated = sv.BoxAnnotator(thickness=2).annotate(image_bgr, det)
        annotated = sv.LabelAnnotator().annotate(annotated, det, labels)
        return annotated
    except TypeError:
        try:
            annotated = sv.BoxAnnotator(thickness=2).annotate(
                scene=image_bgr,
                detections=det,
            )
            annotated = sv.LabelAnnotator().annotate(
                scene=annotated,
                detections=det,
                labels=labels,
            )
            return annotated
        except Exception:
            return None


def _annotate_with_pil(
    image_bgr: np.ndarray,
    signature_detections: List[Dict[str, Any]],
    stamp_detections: List[Dict[str, Any]],
    draw_labels: bool,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    font = _load_font(18)

    for entry, color, name in (
        (signature_detections, (0, 153, 255), "signature"),
        (stamp_detections, (255, 64, 64), "stamp"),
    ):
        for det in entry:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            if draw_labels:
                label = f"{name} {det['confidence']:.2f}"
                _draw_label(draw, max(0, x1), max(0, y1 - 24), label, font)

    annotated_rgb = np.array(image)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def run_stamp_signature_detection(
    image_path: Path,
    model_sig: YOLO,
    processor: AutoImageProcessor,
    model_stamp: AutoModelForObjectDetection,
    device: str,
    sig_conf: float,
    stamp_conf: float,
    use_supervision: bool,
    draw_labels: bool,
) -> Tuple[Dict[str, Any], np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    sig_results = model_sig.predict(
        source=str(image_path),
        conf=sig_conf,
        device=device,
        verbose=False,
    )[0]

    signature_detections: List[Dict[str, Any]] = []
    if sig_results.boxes is not None:
        for box in sig_results.boxes:
            signature_detections.append(
                {
                    "class": "signature",
                    "bbox_xyxy": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                }
            )

    inputs = processor(images=image_pil, return_tensors="pt")
    stamp_device = next(model_stamp.parameters()).device
    inputs = {k: v.to(stamp_device) for k, v in inputs.items()}

    with torch.no_grad():
        stamp_outputs = model_stamp(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]])
    processed = processor.post_process_object_detection(
        stamp_outputs,
        target_sizes=target_sizes,
        threshold=stamp_conf,
    )[0]

    stamp_detections: List[Dict[str, Any]] = []
    for score, box in zip(processed["scores"], processed["boxes"]):
        stamp_detections.append(
            {
                "class": "stamp",
                "bbox_xyxy": box.tolist(),
                "confidence": float(score),
            }
        )

    formatted_output = {
        "image": str(image_path),
        "detections": {
            "signature": signature_detections,
            "stamp": stamp_detections,
        },
    }

    annotated = None
    if use_supervision:
        annotated = _annotate_with_supervision(
            image_bgr,
            signature_detections,
            stamp_detections,
        )
    if annotated is None:
        annotated = _annotate_with_pil(
            image_bgr,
            signature_detections,
            stamp_detections,
            draw_labels,
        )

    return formatted_output, annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run signature (YOLO) + stamp (transformer) detection on a single image.",
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--out-dir", default="signature_detection_out")
    parser.add_argument("--sig-conf", type=float, default=0.25)
    parser.add_argument("--stamp-conf", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--no-supervision", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--image-out", default="")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json_out) if args.json_out else out_dir / "result.json"
    image_out = Path(args.image_out) if args.image_out else out_dir / "result.jpg"

    device = _resolve_device(args.device)
    token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")

    model_sig, processor, model_stamp = _load_models(device, token)

    formatted_output, annotated_image = run_stamp_signature_detection(
        image_path=image_path,
        model_sig=model_sig,
        processor=processor,
        model_stamp=model_stamp,
        device=device,
        sig_conf=args.sig_conf,
        stamp_conf=args.stamp_conf,
        use_supervision=not args.no_supervision,
        draw_labels=args.labels,
    )

    json_path.write_text(json.dumps(formatted_output, indent=2))
    cv2.imwrite(str(image_out), annotated_image)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {image_out}")


if __name__ == "__main__":
    main()
