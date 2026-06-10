import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label


def _save_image(img: Image.Image, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)


def _binary_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255))


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    # Keep the normalization identical to the stage-4 implementation.
    return (gray - gray.min()) / (gray.max() - gray.min() + 1e-5) * 255


def _record_step(steps: list[tuple[str, Image.Image]], name: str, img: Image.Image) -> None:
    steps.append((name, img))


def _resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size for collage.")
    scale = target_width / float(width)
    new_height = max(1, int(height * scale))
    return img.resize((target_width, new_height), Image.LANCZOS)


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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _truncate_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if max_width <= 0:
        return text
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    trimmed = text
    while trimmed and _text_size(draw, trimmed + "...", font)[0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + "...") if trimmed else text


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> None:
    pad = 6
    label = _truncate_text(draw, text, font, max_width - pad * 2)
    text_w, text_h = _text_size(draw, label, font)
    rect = (x, y, x + text_w + pad * 2, y + text_h + pad * 2)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x + pad, y + pad), label, fill=(255, 255, 255), font=font)


def _make_collage(
    steps: list[tuple[str, Image.Image]],
    out_dir: Path,
    cols: int,
    tile_width: int,
    gutter: int,
    label_font_size: int,
) -> None:
    cols = max(1, cols)
    gutter = max(0, gutter)

    if tile_width <= 0:
        tile_width = max(img.width for _, img in steps)
    tile_width = max(32, tile_width)

    resized_steps = []
    for name, img in steps:
        rgb = img.convert("RGB")
        resized = rgb if rgb.width == tile_width else _resize_to_width(rgb, tile_width)
        resized_steps.append((name, resized))

    tile_height = max(img.height for _, img in resized_steps)
    rows = int(math.ceil(len(resized_steps) / float(cols)))
    collage_width = cols * tile_width + (cols - 1) * gutter
    collage_height = rows * tile_height + (rows - 1) * gutter

    collage = Image.new("RGB", (collage_width, collage_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(collage)
    font_size = label_font_size
    if font_size <= 0:
        font_size = max(25, min(55, int(tile_width * 0.08)))
    font = _load_font(font_size)

    for idx, (name, img) in enumerate(resized_steps):
        row = idx // cols
        col = idx % cols
        x = col * (tile_width + gutter)
        y = row * (tile_height + gutter)
        y_offset = y + (tile_height - img.height) // 2
        collage.paste(img, (x, y_offset))

        label_x = x + 8
        label_y = y + 8
        label_text = name.replace("_", " ")
        _draw_label(draw, label_x, label_y, label_text, font, tile_width - 16)

    _save_image(collage, out_dir, "13_collage.png")


def visualize_stage4_steps(
    image_path: Path,
    out_dir: Path,
    sig_min_area: int,
    stamp_min_area: int,
    red_threshold: float,
    bottom_fraction: float,
    collage_cols: int,
    collage_tile_width: int,
    collage_gutter: int,
    collage_label_font_size: int,
) -> None:
    steps: list[tuple[str, Image.Image]] = []
    img = Image.open(image_path).convert("RGB")
    _record_step(steps, "00_original", img)

    arr = np.array(img)
    h, w = arr.shape[:2]
    crop_h = int(h * bottom_fraction)
    bottom_img = arr[h - crop_h:, :, :]

    if bottom_img.size == 0:
        raise ValueError("Bottom crop is empty. Adjust bottom_fraction.")

    _record_step(steps, "01_cropped_bottom", Image.fromarray(bottom_img))

    gray = np.dot(bottom_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray = gray.astype(np.float32)
    _record_step(steps, "02_gray", Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)))

    gray_norm = _normalize_gray(gray).astype(np.uint8)
    _record_step(steps, "03_gray_normalized", Image.fromarray(gray_norm))

    blurred = gaussian_filter(gray_norm, sigma=1)
    _record_step(steps, "04_blurred", Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8)))

    thresh = np.mean(blurred) - 0.5 * np.std(blurred)
    binary = blurred < thresh
    _record_step(steps, "05_binary_thresh", _binary_to_pil(binary))

    binary = binary_erosion(binary, structure=np.ones((3, 3)))
    _record_step(steps, "06_binary_eroded", _binary_to_pil(binary))

    binary = binary_dilation(binary, structure=np.ones((5, 5)))
    _record_step(steps, "07_binary_dilated", _binary_to_pil(binary))

    labeled, num_features = label(binary)

    sig_boxes = []
    for i in range(1, num_features + 1):
        component = labeled == i
        area = np.sum(component)
        if sig_min_area < area < 8000:
            rows, cols = np.where(component)
            if len(rows) > 0 and len(cols) > 0:
                y_offset = h - crop_h
                x1, y1 = int(min(cols)), int(min(rows)) + y_offset
                x2, y2 = int(max(cols)), int(max(rows)) + y_offset
                sig_boxes.append((x1, y1, x2, y2))

    red_channel = bottom_img[..., 0].astype(float)
    green_blue_avg = (bottom_img[..., 1] + bottom_img[..., 2]) / 2
    red_ratio = red_channel / (green_blue_avg + 1e-5)
    red_ratio_vis = np.clip(red_ratio / (red_threshold * 2.0), 0, 1) * 255
    _record_step(steps, "08_red_ratio", Image.fromarray(red_ratio_vis.astype(np.uint8)))

    stamp_binary = (red_ratio > red_threshold) & (red_channel > 50)
    _record_step(steps, "09_stamp_binary", _binary_to_pil(stamp_binary))

    stamp_binary = binary_erosion(stamp_binary, structure=np.ones((3, 3)))
    _record_step(steps, "10_stamp_eroded", _binary_to_pil(stamp_binary))

    stamp_binary = binary_dilation(stamp_binary, structure=np.ones((5, 5)))
    _record_step(steps, "11_stamp_dilated", _binary_to_pil(stamp_binary))

    stamp_labeled, stamp_num = label(stamp_binary)
    stamp_boxes = []
    for i in range(1, stamp_num + 1):
        component = stamp_labeled == i
        area = np.sum(component)
        if area > stamp_min_area:
            rows, cols = np.where(component)
            if len(rows) > 0 and len(cols) > 0:
                y_offset = h - crop_h
                x1, y1 = int(min(cols)), int(min(rows)) + y_offset
                x2, y2 = int(max(cols)), int(max(rows)) + y_offset
                stamp_boxes.append((x1, y1, x2, y2))

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    for box in stamp_boxes:
        draw.rectangle(box, outline="red", width=3)
    for box in sig_boxes:
        draw.rectangle(box, outline="blue", width=3)
    _record_step(steps, "12_boxes_overlay", overlay)

    _make_collage(
        steps=steps,
        out_dir=out_dir,
        cols=collage_cols,
        tile_width=collage_tile_width,
        gutter=collage_gutter,
        label_font_size=collage_label_font_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize stage-4 image-processing steps for stamp/signature detection."
    )
    parser.add_argument("image_path", help="Path to the image to visualize.")
    parser.add_argument("--out-dir", default="stage4_viz_out", help="Output folder for images.")
    parser.add_argument("--sig-min-area", type=int, default=800)
    parser.add_argument("--stamp-min-area", type=int, default=1200)
    parser.add_argument("--red-threshold", type=float, default=1.5)
    parser.add_argument("--bottom-fraction", type=float, default=0.25)
    parser.add_argument("--collage-cols", type=int, default=4)
    parser.add_argument(
        "--collage-tile-width",
        type=int,
        default=0,
        help="Tile width in pixels (0 uses the max step width for higher quality).",
    )
    parser.add_argument("--collage-gutter", type=int, default=8)
    parser.add_argument(
        "--collage-label-font-size",
        type=int,
        default=0,
        help="Label font size (0 uses an auto size based on tile width).",
    )
    args = parser.parse_args()

    visualize_stage4_steps(
        image_path=Path(args.image_path),
        out_dir=Path(args.out_dir),
        sig_min_area=args.sig_min_area,
        stamp_min_area=args.stamp_min_area,
        red_threshold=args.red_threshold,
        bottom_fraction=args.bottom_fraction,
        collage_cols=args.collage_cols,
        collage_tile_width=args.collage_tile_width,
        collage_gutter=args.collage_gutter,
        collage_label_font_size=args.collage_label_font_size,
    )


if __name__ == "__main__":
    main()
