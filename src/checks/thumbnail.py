"""
Thumbnail Consistency Check.

Compares the embedded EXIF thumbnail with the main image.
If the main image was edited but the thumbnail wasn't updated,
there will be a detectable mismatch.

Completely new — not in fraud V2.
"""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import create_side_by_side, numpy_to_pil_heatmap, draw_heatmap_overlay
from src.utils.image_loader import extract_exif_thumbnail

logger = logging.getLogger(__name__)


class ThumbnailCheck(BaseCheck):
    name = "Thumbnail Consistency"
    check_id = "thumbnail"
    description = "Compare embedded EXIF thumbnail with main image"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().thumbnail_config
        ssim_thresh = config.get("ssim_mismatch_threshold", 0.9)
        diff_thresh = config.get("pixel_diff_threshold", 15)

        visual_outputs = []

        if not file_path:
            fallback = image.copy()
            path = self._save_evidence(fallback, "thumbnail", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Thumbnail",
                image_path=path,
                description="No file path — cannot extract EXIF thumbnail.",
            ))
            return self._make_pass_result(
                self.name, self.check_id,
                "No file path available for thumbnail extraction.",
                visual_outputs, {},
            )

        # Extract embedded thumbnail
        thumbnail = extract_exif_thumbnail(file_path)

        if thumbnail is None:
            fallback = image.copy()
            path = self._save_evidence(fallback, "thumbnail", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Thumbnail",
                image_path=path,
                description="No embedded EXIF thumbnail found in this file.",
            ))
            return self._make_pass_result(
                self.name, self.check_id,
                "No EXIF thumbnail found (file may not be JPEG or thumbnail was stripped).",
                visual_outputs, {"thumbnail_found": False},
            )

        # Resize main image to thumbnail size for comparison
        main_resized = image.convert("RGB").resize(thumbnail.size, Image.LANCZOS)

        # Compute SSIM
        from skimage.metrics import structural_similarity as ssim_fn

        main_arr = np.array(main_resized)
        thumb_arr = np.array(thumbnail)

        if main_arr.shape != thumb_arr.shape:
            h = min(main_arr.shape[0], thumb_arr.shape[0])
            w = min(main_arr.shape[1], thumb_arr.shape[1])
            main_arr = main_arr[:h, :w]
            thumb_arr = thumb_arr[:h, :w]

        ssim_val, ssim_map = ssim_fn(
            main_arr, thumb_arr, channel_axis=2, full=True, data_range=255,
        )

        # Pixel difference
        diff = np.abs(main_arr.astype(np.float32) - thumb_arr.astype(np.float32))
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        # Side-by-side comparison
        comparison = create_side_by_side(
            thumbnail, main_resized,
            left_title="EXIF Thumbnail",
            right_title="Main Image (resized)",
        )
        path = self._save_evidence(comparison, "comparison", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Comparison",
            image_path=path,
            description=f"Side-by-side: SSIM={ssim_val:.4f}, Mean diff={mean_diff:.1f}",
        ))

        # Difference heatmap
        diff_gray = np.mean(diff, axis=2)
        diff_heatmap = numpy_to_pil_heatmap(diff_gray)
        path = self._save_evidence(diff_heatmap, "difference_map", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Difference Map",
            image_path=path,
            description="Pixel difference between thumbnail and main image",
        ))

        evidence = {
            "thumbnail_found": True,
            "thumbnail_size": thumbnail.size,
            "ssim": round(ssim_val, 4),
            "mean_pixel_diff": round(mean_diff, 2),
            "max_pixel_diff": round(max_diff, 2),
        }

        # Score
        if ssim_val < ssim_thresh * 0.8 or mean_diff > diff_thresh * 2:
            score = min(100, int(60 + (1.0 - ssim_val) * 100))
            severity = Severity.HIGH
            summary = f"Thumbnail mismatch detected (SSIM: {ssim_val:.3f})"
            detail = (
                f"EXIF thumbnail differs significantly from main image. "
                f"SSIM={ssim_val:.4f}, mean diff={mean_diff:.1f}. "
                f"The image may have been edited after thumbnail creation."
            )
        elif ssim_val < ssim_thresh or mean_diff > diff_thresh:
            score = min(60, int(30 + (1.0 - ssim_val) * 50))
            severity = Severity.MEDIUM
            summary = f"Minor thumbnail discrepancy (SSIM: {ssim_val:.3f})"
            detail = (
                f"Slight differences between thumbnail and main image. "
                f"SSIM={ssim_val:.4f}, mean diff={mean_diff:.1f}."
            )
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"Thumbnail matches main image. SSIM={ssim_val:.4f}, diff={mean_diff:.1f}.",
                visual_outputs, evidence,
            )

        return self._make_fail_result(
            self.name, self.check_id, score, severity,
            summary, detail, visual_outputs, evidence,
        )
