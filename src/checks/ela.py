"""
Multi-Level Error Level Analysis (ELA).

Runs ELA at multiple JPEG compression quality levels and builds
a consensus map. Only regions that trigger across multiple levels
are considered strongly suspicious.

Enhanced from fraud V2 ela_check.py: single Q=95 → 7 quality levels.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import (
    draw_bounding_boxes,
    draw_heatmap_overlay,
    create_grid_image,
    numpy_to_pil_heatmap,
    RED,
    YELLOW,
    GREEN,
)

logger = logging.getLogger(__name__)


class ELACheck(BaseCheck):
    name = "ELA Multi-Level"
    check_id = "ela"
    description = "Error Level Analysis at multiple JPEG compression levels"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().ela
        quality_levels = config.get("quality_levels", [60, 70, 75, 80, 85, 90, 95])
        scale_factor = config.get("scale_factor", 15)
        region_threshold = config.get("region_threshold", 60)
        consensus_min = config.get("consensus_min_levels", 3)
        min_contour_area = config.get("min_contour_area", 200)

        image_rgb = image.convert("RGB")
        orig_array = np.array(image_rgb, dtype=np.float32)

        ela_images = []
        binary_masks = []
        per_level_stats = {}
        visual_outputs = []

        # Run ELA at each quality level
        for quality in quality_levels:
            ela_array, gray_diff, mask = self._compute_ela(
                image_rgb, orig_array, quality, scale_factor, region_threshold
            )

            ela_pil = Image.fromarray(ela_array)
            ela_images.append(ela_pil)
            binary_masks.append(mask)

            mean_error = float(np.mean(gray_diff))
            max_error = float(np.max(gray_diff))
            high_pct = float(np.sum(mask > 0) / mask.size * 100)

            per_level_stats[quality] = {
                "mean_error": round(mean_error, 2),
                "max_error": round(max_error, 2),
                "high_error_pct": round(high_pct, 2),
            }

            # Find contours and draw bounding boxes on this level's image
            contours_info = self._find_contour_regions(mask, min_contour_area)
            if contours_info:
                boxed = draw_bounding_boxes(
                    ela_pil,
                    contours_info,
                    color=YELLOW,
                    thickness=2,
                    label_key="label",
                    alpha=0.15,
                )
            else:
                boxed = ela_pil

            path = self._save_evidence(boxed, f"q{quality}", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title=f"Q={quality}",
                image_path=path,
                description=(
                    f"ELA at quality {quality}: mean error={mean_error:.1f}, "
                    f"{high_pct:.1f}% hot regions"
                ),
            ))

        # Build consensus mask — regions that trigger across multiple levels
        consensus = np.zeros_like(binary_masks[0], dtype=np.int32)
        for mask in binary_masks:
            consensus += (mask > 0).astype(np.int32)

        consensus_binary = (consensus >= consensus_min).astype(np.uint8) * 255
        consensus_contours = self._find_contour_regions(
            consensus_binary, min_contour_area
        )

        # Consensus overlay on original image
        consensus_heatmap = (consensus.astype(np.float32) / len(quality_levels) * 255)
        consensus_overlay = draw_heatmap_overlay(
            image, consensus_heatmap, alpha=0.5
        )

        if consensus_contours:
            consensus_overlay = draw_bounding_boxes(
                consensus_overlay,
                consensus_contours,
                color=RED,
                thickness=3,
                label_key="label",
                alpha=0.2,
            )

        path = self._save_evidence(consensus_overlay, "consensus", evidence_store, page_num)
        visual_outputs.insert(0, VisualOutput(
            title="Consensus",
            image_path=path,
            description=(
                f"Regions triggering across ≥{consensus_min} quality levels. "
                f"Red boxes = high-confidence tampering indicators."
            ),
        ))

        # Also add a grid view of all levels
        grid = create_grid_image(
            ela_images,
            titles=[f"Q={q}" for q in quality_levels],
            cols=4,
            cell_size=250,
        )
        path = self._save_evidence(grid, "all_levels", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="All Levels",
            image_path=path,
            description="Side-by-side comparison of all ELA quality levels",
        ))

        # Scoring
        n_consensus = len(consensus_contours)
        consensus_area_pct = float(np.sum(consensus_binary > 0) / consensus_binary.size * 100)

        # Average mean error across high quality levels
        high_q_errors = [
            per_level_stats[q]["mean_error"]
            for q in quality_levels if q >= 80
        ]
        avg_high_error = np.mean(high_q_errors) if high_q_errors else 0

        if n_consensus >= 5 or consensus_area_pct > 10 or avg_high_error > 40:
            score = min(100, int(60 + avg_high_error + consensus_area_pct))
            severity = Severity.CRITICAL
            summary = f"Strong tampering indicators: {n_consensus} consensus regions"
            detail = (
                f"ELA detected {n_consensus} regions that show elevated error "
                f"across {consensus_min}+ quality levels. "
                f"Consensus area: {consensus_area_pct:.1f}%, "
                f"avg high-Q error: {avg_high_error:.1f}."
            )
        elif n_consensus >= 2 or consensus_area_pct > 3 or avg_high_error > 25:
            score = min(80, int(30 + avg_high_error + consensus_area_pct * 2))
            severity = Severity.HIGH
            summary = f"Suspicious regions found: {n_consensus} consensus areas"
            detail = (
                f"ELA found {n_consensus} suspicious regions across multiple levels. "
                f"Consensus area: {consensus_area_pct:.1f}%, "
                f"avg high-Q error: {avg_high_error:.1f}."
            )
        elif n_consensus >= 1 or avg_high_error > 15:
            score = min(50, int(15 + avg_high_error))
            severity = Severity.MEDIUM
            summary = "Minor ELA anomalies detected"
            detail = (
                f"Minor inconsistencies: {n_consensus} weak consensus region(s), "
                f"avg error: {avg_high_error:.1f}."
            )
        else:
            score = 0
            severity = Severity.CLEAN
            summary = "No ELA anomalies detected"
            detail = (
                f"Compression artifacts are consistent across all {len(quality_levels)} "
                f"quality levels. Avg high-Q error: {avg_high_error:.1f}."
            )

        evidence = {
            "per_level_stats": per_level_stats,
            "consensus_regions": n_consensus,
            "consensus_area_pct": round(consensus_area_pct, 2),
            "avg_high_q_error": round(avg_high_error, 2),
            "quality_levels_tested": quality_levels,
            "consensus_min_levels": consensus_min,
        }

        triggered = score > 0
        if not triggered:
            return self._make_pass_result(
                self.name, self.check_id, detail, visual_outputs, evidence
            )

        return self._make_fail_result(
            self.name, self.check_id, score, severity,
            summary, detail, visual_outputs, evidence
        )

    @staticmethod
    def _compute_ela(
        image_rgb: Image.Image,
        orig_array: np.ndarray,
        quality: int,
        scale_factor: int,
        threshold: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ELA at a specific quality level."""
        buf = io.BytesIO()
        image_rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)

        diff = np.abs(orig_array - resaved)
        gray_diff = np.mean(diff, axis=2)

        # Scale for visibility
        ela_array = np.clip(diff * scale_factor, 0, 255).astype(np.uint8)

        # Binary mask of high-error regions
        mask = (gray_diff > threshold).astype(np.uint8) * 255

        return ela_array, gray_diff, mask

    @staticmethod
    def _find_contour_regions(
        binary_mask: np.ndarray,
        min_area: int,
    ) -> List[Dict[str, Any]]:
        """Find bounding box regions from a binary mask."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "label": f"{int(area)}px²",
            })

        return regions
