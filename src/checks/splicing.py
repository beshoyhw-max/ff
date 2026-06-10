"""
Splicing / Image Compositing Detection.

Detects regions pasted from a different image via:
- Color temperature consistency analysis (LAB color space)
- Illumination direction estimation
- Edge artifact detection at region boundaries
- Local noise level comparison

Completely new — not in fraud V2.
"""

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
    numpy_to_pil_heatmap,
    RED,
    YELLOW,
)

logger = logging.getLogger(__name__)


class SplicingCheck(BaseCheck):
    name = "Splicing Detection"
    check_id = "splicing"
    description = "Detect spliced/composited regions from different images"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().splicing
        block_size = config.get("block_size", 64)
        color_z = config.get("color_temp_z_threshold", 2.5)
        edge_thresh = config.get("edge_gradient_threshold", 100)
        noise_thresh = config.get("noise_mismatch_threshold", 2.0)

        visual_outputs = []

        # 1. Color temperature analysis
        color_score, color_ev, color_heatmap, color_outliers = self._color_temp_analysis(
            image, block_size, color_z
        )

        if color_heatmap is not None:
            path = self._save_evidence(color_heatmap, "color_temp", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Color Temperature",
                image_path=path,
                description=(
                    f"Color temperature deviation map. "
                    f"{len(color_outliers)} outlier block(s) detected."
                ),
            ))

        # 2. Illumination consistency
        illum_score, illum_ev, illum_visual = self._illumination_analysis(image, block_size)

        if illum_visual is not None:
            path = self._save_evidence(illum_visual, "illumination", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Illumination",
                image_path=path,
                description=(
                    f"Gradient direction consistency. "
                    f"Direction variance: {illum_ev.get('direction_variance', 0):.2f}"
                ),
            ))

        # 3. Edge artifact detection
        edge_score, edge_ev, edge_visual = self._edge_artifact_detection(
            image, edge_thresh
        )

        if edge_visual is not None:
            path = self._save_evidence(edge_visual, "edge_artifacts", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Edge Artifacts",
                image_path=path,
                description=(
                    f"Suspicious edge artifacts. "
                    f"Score: {edge_score}/100"
                ),
            ))

        # 4. Local noise mismatch
        noise_score, noise_ev = self._local_noise_comparison(image, block_size, noise_thresh)

        # Combined score
        combined = int(
            color_score * 0.30 + illum_score * 0.25
            + edge_score * 0.25 + noise_score * 0.20
        )

        evidence = {
            "color_temperature": {"score": color_score, **color_ev},
            "illumination": {"score": illum_score, **illum_ev},
            "edge_artifacts": {"score": edge_score, **edge_ev},
            "noise_mismatch": {"score": noise_score, **noise_ev},
            "combined": combined,
        }

        if combined >= 55:
            severity = Severity.HIGH
            summary = f"Possible image splicing detected (score: {combined})"
            detail = (
                f"Color={color_score}, Illumination={illum_score}, "
                f"Edge={edge_score}, Noise={noise_score}."
            )
        elif combined >= 30:
            severity = Severity.MEDIUM
            summary = f"Minor splicing indicators (score: {combined})"
            detail = (
                f"Color={color_score}, Illumination={illum_score}, "
                f"Edge={edge_score}, Noise={noise_score}."
            )
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"No splicing indicators detected. Score: {combined}/100.",
                visual_outputs, evidence,
            )

        return self._make_fail_result(
            self.name, self.check_id, combined, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _color_temp_analysis(
        self, image: Image.Image, block_size: int, z_threshold: float,
    ) -> Tuple[int, Dict, Optional[Image.Image], List[Dict]]:
        """Analyze color temperature consistency in LAB color space."""
        img = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
        h, w, _ = lab.shape

        block_data = []
        a_map = np.zeros((h, w), dtype=np.float64)
        b_map = np.zeros((h, w), dtype=np.float64)

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = lab[y:y + block_size, x:x + block_size]
                mean_a = float(np.mean(block[:, :, 1]))
                mean_b = float(np.mean(block[:, :, 2]))
                a_map[y:y + block_size, x:x + block_size] = mean_a
                b_map[y:y + block_size, x:x + block_size] = mean_b
                block_data.append({
                    "x": x, "y": y,
                    "width": block_size, "height": block_size,
                    "mean_a": mean_a, "mean_b": mean_b,
                })

        if len(block_data) < 4:
            return 0, {"insufficient_blocks": True}, None, []

        a_vals = np.array([b["mean_a"] for b in block_data])
        b_vals = np.array([b["mean_b"] for b in block_data])
        a_mean, a_std = np.mean(a_vals), max(np.std(a_vals), 0.5)
        b_mean, b_std = np.mean(b_vals), max(np.std(b_vals), 0.5)

        outliers = []
        for bd in block_data:
            z_a = abs(bd["mean_a"] - a_mean) / a_std
            z_b = abs(bd["mean_b"] - b_mean) / b_std
            z_max = max(z_a, z_b)
            if z_max > z_threshold:
                outliers.append({
                    **bd, "label": f"z={z_max:.1f}", "z_score": round(z_max, 2),
                })

        deviation = np.sqrt(
            ((a_map - a_mean) / a_std) ** 2 + ((b_map - b_mean) / b_std) ** 2
        )

        heatmap = draw_heatmap_overlay(image, deviation, alpha=0.5)
        if outliers:
            heatmap = draw_bounding_boxes(heatmap, outliers, color=RED, thickness=2, alpha=0.15)

        outlier_pct = len(outliers) / len(block_data) * 100

        if outlier_pct > 15:
            score = min(100, int(50 + outlier_pct * 2))
        elif outlier_pct > 5:
            score = int(20 + outlier_pct * 3)
        else:
            score = 0

        evidence = {
            "block_count": len(block_data),
            "outlier_count": len(outliers),
            "outlier_pct": round(outlier_pct, 1),
        }

        return score, evidence, heatmap, outliers

    def _illumination_analysis(
        self, image: Image.Image, block_size: int,
    ) -> Tuple[int, Dict, Optional[Image.Image]]:
        """Estimate illumination direction per block."""
        gray = np.array(image.convert("L"), dtype=np.float64)
        h, w = gray.shape

        directions = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                if np.std(block) < 3.0:
                    continue
                gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                angle = np.arctan2(np.mean(gy), np.mean(gx))
                magnitude = np.sqrt(np.mean(gx) ** 2 + np.mean(gy) ** 2)
                if magnitude > 1.0:
                    directions.append({"x": x, "y": y, "angle": float(angle), "magnitude": float(magnitude)})

        if len(directions) < 4:
            return 0, {"insufficient_data": True}, None

        angles = np.array([d["angle"] for d in directions])
        sin_mean = float(np.mean(np.sin(angles)))
        cos_mean = float(np.mean(np.cos(angles)))
        direction_variance = 1.0 - np.sqrt(sin_mean ** 2 + cos_mean ** 2)

        if direction_variance > 0.5:
            score = min(100, int(40 + direction_variance * 80))
        elif direction_variance > 0.3:
            score = int(15 + direction_variance * 50)
        else:
            score = 0

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        illum_visual = draw_heatmap_overlay(image, grad_mag, alpha=0.4)

        evidence = {
            "direction_variance": round(direction_variance, 4),
            "block_count": len(directions),
            "dominant_direction": round(float(np.arctan2(sin_mean, cos_mean)), 3),
        }

        return score, evidence, illum_visual

    def _edge_artifact_detection(
        self, image: Image.Image, threshold: int
    ) -> Tuple[int, Dict, Optional[Image.Image]]:
        """Detect unnatural edge artifacts that may indicate splicing boundaries."""
        gray = np.array(image.convert("L"))
        edges_tight = cv2.Canny(gray, threshold, threshold * 2)
        edges_loose = cv2.Canny(gray, threshold // 2, threshold)
        artificial = cv2.bitwise_and(edges_tight, cv2.bitwise_not(edges_loose))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(artificial, cv2.MORPH_CLOSE, kernel)

        artifact_ratio = float(np.sum(closed > 0) / closed.size * 100)

        if artifact_ratio > 2.0:
            score = min(100, int(40 + artifact_ratio * 10))
        elif artifact_ratio > 0.5:
            score = int(10 + artifact_ratio * 15)
        else:
            score = 0

        edge_heatmap = draw_heatmap_overlay(
            image, closed.astype(np.float64), alpha=0.5, colormap=cv2.COLORMAP_HOT
        )

        evidence = {"artifact_ratio_pct": round(artifact_ratio, 3), "edge_threshold": threshold}
        return score, evidence, edge_heatmap

    def _local_noise_comparison(
        self, image: Image.Image, block_size: int, threshold: float
    ) -> Tuple[int, Dict]:
        """Compare local noise levels between adjacent blocks."""
        gray = np.array(image.convert("L"), dtype=np.float64)
        h, w = gray.shape

        noise_grid = []
        for y in range(0, h - block_size, block_size):
            row = []
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                noise = float(np.std(laplacian))
                row.append(noise)
            noise_grid.append(row)

        if len(noise_grid) < 2 or len(noise_grid[0]) < 2:
            return 0, {"insufficient_data": True}

        grid = np.array(noise_grid)
        diffs = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if x + 1 < grid.shape[1]:
                    diffs.append(abs(grid[y, x] - grid[y, x + 1]))
                if y + 1 < grid.shape[0]:
                    diffs.append(abs(grid[y, x] - grid[y + 1, x]))

        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs))
        max_diff = float(np.max(diffs))
        mean_noise = float(np.mean(grid))

        ratio = max_diff / mean_noise if mean_noise > 0 else 0

        if ratio > threshold * 2:
            score = min(100, int(50 + ratio * 10))
        elif ratio > threshold:
            score = int(20 + ratio * 10)
        else:
            score = 0

        evidence = {
            "mean_adjacent_diff": round(mean_diff, 2),
            "max_adjacent_diff": round(max_diff, 2),
            "noise_ratio": round(ratio, 3),
        }

        return score, evidence
