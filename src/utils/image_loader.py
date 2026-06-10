"""
Image and PDF loading utilities.

Handles all file types: JPEG, PNG, BMP, TIFF, PDF.
PDFs are rendered to images via PyMuPDF.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.models import FileType

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def detect_file_type(file_path: str) -> FileType:
    """Detect whether a file is an image or PDF."""
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return FileType.IMAGE
    if ext in SUPPORTED_PDF_EXTENSIONS:
        return FileType.PDF
    return FileType.UNKNOWN


def load_image(
    file_path: str,
    max_dimension: int = 4096,
) -> Tuple[Image.Image, FileType, Dict]:
    """
    Load an image or PDF and return as PIL Image.

    For PDFs, renders the first page at 200 DPI.

    Args:
        file_path: Path to the file.
        max_dimension: Maximum dimension (width or height) to resize to.

    Returns:
        Tuple of (PIL Image, FileType, metadata_dict).
    """
    file_type = detect_file_type(file_path)

    if file_type == FileType.PDF:
        image, meta = _load_pdf(file_path, max_dimension)
        return image, file_type, meta
    elif file_type == FileType.IMAGE:
        image, meta = _load_image_file(file_path, max_dimension)
        return image, file_type, meta
    else:
        raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")


def _load_image_file(
    file_path: str,
    max_dimension: int,
) -> Tuple[Image.Image, Dict]:
    """Load an image file."""
    image = Image.open(file_path)

    meta = {
        "format": image.format,
        "mode": image.mode,
        "original_size": image.size,
    }

    # Convert to RGB
    if image.mode != "RGB":
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

    # Resize if needed
    image = _constrain_size(image, max_dimension)
    meta["loaded_size"] = image.size

    return image, meta


def _load_pdf(
    file_path: str,
    max_dimension: int,
) -> Tuple[Image.Image, Dict]:
    """Load first page of a PDF as an image."""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    meta = {
        "page_count": doc.page_count,
        "pdf_metadata": doc.metadata or {},
        "format": "PDF",
    }

    page = doc[0]
    # Render at 200 DPI
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat)

    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    image = _constrain_size(image, max_dimension)
    meta["original_size"] = (pix.width, pix.height)
    meta["loaded_size"] = image.size

    return image, meta


def load_all_pdf_pages(
    file_path: str,
    max_dimension: int = 4096,
) -> List[Image.Image]:
    """Load all pages of a PDF as images."""
    import fitz

    doc = fitz.open(file_path)
    pages = []
    mat = fitz.Matrix(200 / 72, 200 / 72)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = _constrain_size(img, max_dimension)
        pages.append(img)

    doc.close()
    return pages


def _constrain_size(image: Image.Image, max_dim: int) -> Image.Image:
    """Resize image if any dimension exceeds max_dim."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def extract_exif(file_path: str) -> Dict:
    """Extract EXIF data from an image file."""
    try:
        import exifread

        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        return {
            str(k): str(v)
            for k, v in tags.items()
            if not k.startswith("Thumbnail")
        }
    except Exception as e:
        logger.debug(f"EXIF extraction failed: {e}")
        return {}


def extract_pdf_metadata(file_path: str) -> Dict:
    """Extract metadata from a PDF file."""
    try:
        import fitz
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        doc.close()
        return meta
    except Exception as e:
        logger.debug(f"PDF metadata extraction failed: {e}")
        return {}


def extract_exif_thumbnail(file_path: str) -> Optional[Image.Image]:
    """Extract embedded EXIF thumbnail from a JPEG file."""
    try:
        import exifread

        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=True)

        if "JPEGThumbnail" in tags:
            import io
            thumb_data = tags["JPEGThumbnail"]
            return Image.open(io.BytesIO(thumb_data)).convert("RGB")
    except Exception as e:
        logger.debug(f"Thumbnail extraction failed: {e}")
    return None
