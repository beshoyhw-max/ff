"""
Bounding box and visualization drawing utilities.

Shared across all check modules for consistent visual output.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Color constants (RGB tuples)
RED = (255, 50, 50)
ORANGE = (255, 140, 50)
YELLOW = (255, 210, 50)
GREEN = (50, 200, 100)
BLUE = (50, 120, 255)
CYAN = (50, 210, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Severity-to-color mapping
SEVERITY_COLORS = {
    "clean": GREEN,
    "low": (136, 204, 68),
    "medium": YELLOW,
    "high": ORANGE,
    "critical": RED,
}


def draw_bounding_boxes(
    image: Image.Image,
    regions: List[dict],
    color: Tuple[int, int, int] = RED,
    thickness: int = 3,
    label_key: Optional[str] = "label",
    alpha: float = 0.3,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input PIL Image.
        regions: List of dicts with keys: x, y, width, height, and optionally a label.
        color: RGB color tuple.
        thickness: Box line thickness.
        label_key: Key in region dict for the label text. None to skip labels.
        alpha: Fill opacity for semi-transparent background.

    Returns:
        New PIL Image with boxes drawn.
    """
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_result = ImageDraw.Draw(result)

    for region in regions:
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("width", 0))
        h = int(region.get("height", 0))

        if w <= 0 or h <= 0:
            continue

        # Semi-transparent fill
        fill_color = (*color, int(255 * alpha))
        draw_overlay.rectangle([x, y, x + w, y + h], fill=fill_color)

        # Solid border
        draw_result.rectangle(
            [x, y, x + w, y + h],
            outline=(*color, 255),
            width=thickness,
        )

        # Label
        if label_key and label_key in region:
            label = str(region[label_key])
            _draw_label(draw_result, x, y - 20, label, color)

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def draw_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Overlay a heatmap on the original image.

    Args:
        image: Input PIL Image.
        heatmap: 2D numpy array (any dtype, will be normalized to 0-255).
        alpha: Blend factor (0=original only, 1=heatmap only).
        colormap: OpenCV colormap constant.

    Returns:
        New PIL Image with heatmap overlaid.
    """
    img_array = np.array(image.convert("RGB"))

    # Normalize heatmap to 0-255
    h_min, h_max = float(np.min(heatmap)), float(np.max(heatmap))
    if h_max - h_min > 0:
        normalized = ((heatmap - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(heatmap, dtype=np.uint8)

    # Resize heatmap to match image
    normalized = cv2.resize(normalized, (img_array.shape[1], img_array.shape[0]))

    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(img_array, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(blended)


def draw_arrows(
    image: Image.Image,
    pairs: List[Tuple[dict, dict]],
    color: Tuple[int, int, int] = CYAN,
    thickness: int = 2,
) -> Image.Image:
    """
    Draw arrows between source-clone region pairs.

    Args:
        image: Input PIL Image.
        pairs: List of (source_region, clone_region) dicts with x, y, width, height.
        color: RGB color.
        thickness: Arrow line thickness.

    Returns:
        New PIL Image with arrows drawn.
    """
    img_array = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv_color = (color[2], color[1], color[0])

    for source, clone in pairs:
        sx = int(source["x"] + source["width"] / 2)
        sy = int(source["y"] + source["height"] / 2)
        cx = int(clone["x"] + clone["width"] / 2)
        cy = int(clone["y"] + clone["height"] / 2)

        cv2.arrowedLine(bgr, (sx, sy), (cx, cy), cv_color, thickness, tipLength=0.03)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def create_grid_image(
    images: List[Image.Image],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    cell_size: int = 300,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (30, 30, 40),
    title_color: Tuple[int, int, int] = WHITE,
) -> Image.Image:
    """
    Arrange multiple images in a labeled grid.

    Args:
        images: List of PIL Images to arrange.
        titles: Optional list of titles for each image.
        cols: Number of columns.
        cell_size: Width/height of each cell.
        padding: Padding between cells.
        bg_color: Background color.
        title_color: Title text color.

    Returns:
        Single PIL Image containing the grid.
    """
    if not images:
        return Image.new("RGB", (cell_size, cell_size), bg_color)

    n = len(images)
    rows = (n + cols - 1) // cols
    title_height = 25 if titles else 0

    total_w = cols * (cell_size + padding) + padding
    total_h = rows * (cell_size + title_height + padding) + padding

    grid = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(grid)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + title_height + padding)

        # Title
        if titles and i < len(titles):
            _draw_label(draw, x, y, titles[i], title_color, bg=False)
            y += title_height

        # Resize image to fit cell
        thumb = img.copy()
        thumb.thumbnail((cell_size, cell_size), Image.LANCZOS)

        # Center in cell
        offset_x = x + (cell_size - thumb.width) // 2
        offset_y = y + (cell_size - thumb.height) // 2
        grid.paste(thumb, (offset_x, offset_y))

    return grid


def create_side_by_side(
    left: Image.Image,
    right: Image.Image,
    left_title: str = "Left",
    right_title: str = "Right",
    max_height: int = 600,
) -> Image.Image:
    """Create a side-by-side comparison image."""
    title_h = 30
    padding = 20

    # Resize both to same height
    scale_l = max_height / left.height
    scale_r = max_height / right.height
    scale = min(scale_l, scale_r, 1.0)

    left_r = left.resize((int(left.width * scale), int(left.height * scale)), Image.LANCZOS)
    right_r = right.resize((int(right.width * scale), int(right.height * scale)), Image.LANCZOS)

    total_w = left_r.width + right_r.width + padding * 3
    total_h = max(left_r.height, right_r.height) + title_h + padding * 2

    result = Image.new("RGB", (total_w, total_h), (30, 30, 40))
    draw = ImageDraw.Draw(result)

    # Titles
    _draw_label(draw, padding, padding, left_title, WHITE, bg=False)
    _draw_label(draw, padding * 2 + left_r.width, padding, right_title, WHITE, bg=False)

    # Images
    result.paste(left_r, (padding, title_h + padding))
    result.paste(right_r, (padding * 2 + left_r.width, title_h + padding))

    return result


def create_info_card(
    data: dict,
    title: str = "Information",
    width: int = 800,
    bg_color: Tuple[int, int, int] = (30, 30, 40),
) -> Image.Image:
    """Render a metadata dict as a visual info card."""
    line_height = 22
    padding = 20
    title_height = 40

    lines = []
    for key, value in data.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        lines.append(f"{key}: {val_str}")

    height = title_height + padding * 2 + len(lines) * line_height + padding
    height = max(height, 100)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, width, title_height], fill=(50, 50, 70))
    _draw_label(draw, padding, 10, title, CYAN, bg=False)

    # Data lines
    y = title_height + padding
    for line in lines:
        draw.text((padding, y), line, fill=WHITE)
        y += line_height

    return img


def numpy_to_pil_heatmap(
    array: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Convert a 2D numpy array to a colorized PIL heatmap image."""
    h_min, h_max = float(np.min(array)), float(np.max(array))
    if h_max - h_min > 0:
        normalized = ((array - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(array, dtype=np.uint8)

    colored = cv2.applyColorMap(normalized, colormap)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def _draw_label(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    text: str,
    color: Tuple[int, int, int],
    bg: bool = True,
) -> None:
    """Draw a small label with optional background."""
    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((x, y), text, font=font)
    if bg:
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(*color, 180),
        )
        draw.text((x, y), text, fill=WHITE, font=font)
    else:
        draw.text((x, y), text, fill=color, font=font)
