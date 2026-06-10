"""
Noise Pattern Analysis with Visual Bounding Boxes.

Detects image manipulation through noise consistency analysis:
- Multi-level wavelet noise estimation
- Per-block noise variance with bounding boxes on outliers
- Sliding-window PRNU analysis

Enhanced from fraud V2: adds bounding boxes, multi-scale wavelets, noise heatmap.
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
    GREEN,
    BLUE,
)

logger = logging.getLogger(__name__)


class NoiseCheck(BaseCheck):
    name = "Noise Analysis"
    check_id = "noise"
    description = "Multi-scale noise pattern consistency analysis"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().noise
        block_size = config.get("block_size", 64)
        outlier_z = config.get("outlier_z_threshold", 2.0)
        min_blocks = config.get("min_blocks", 8)
        wavelet_levels = config.get("wavelet_levels", 4)
        wavelet_name = config.get("wavelet_name", "db4")

        gray = np.array(image.convert("L"), dtype=np.float64)
        analysis_gray = cv2.resize(gray, (512, 512))
        visual_outputs = []

        # 1. Wavelet noise analysis
        wavelet_score, wavelet_ev = self._wavelet_noise(
            analysis_gray, wavelet_name, wavelet_levels
        )

        # 2. Per-block noise variance with bounding boxes
        region_score, region_ev, noise_map, outlier_regions = self._region_noise(
            gray, image, block_size, outlier_z, min_blocks
        )

        # Build noise variance heatmap
        if noise_map is not None:
            heatmap_img = draw_heatmap_overlay(image, noise_map, alpha=0.5)
            path = self._save_evidence(heatmap_img, "noise_map", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Noise Map",
                image_path=path,
                description="Block-level noise variance heatmap. Hot regions = different noise profile.",
            ))

        # Build bounding box overlay on outlier blocks
        if outlier_regions:
            boxed = draw_bounding_boxes(
                image, outlier_regions, color=RED, thickness=2,
                label_key="label", alpha=0.2
            )
        else:
            boxed = image.copy()

        path = self._save_evidence(boxed, "outlier_blocks", evidence_store, page_num)
        visual_outputs.insert(0, VisualOutput(
            title="Outlier Blocks",
            image_path=path,
            description=(
                f"{len(outlier_regions)} noise outlier blocks detected "
                f"(z-score > {outlier_z})."
            ),
        ))

        # 3. PRNU analysis
        prnu_score, prnu_ev, prnu_heatmap = self._prnu_analysis(
            analysis_gray, image
        )

        if prnu_heatmap is not None:
            path = self._save_evidence(prnu_heatmap, "prnu_residual", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="PRNU Residual",
                image_path=path,
                description="Sensor noise residual. Inconsistencies suggest compositing.",
            ))

        # Weighted combination
        combined = int(wavelet_score * 0.30 + region_score * 0.45 + prnu_score * 0.25)

        evidence = {
            "wavelet": {"score": wavelet_score, **wavelet_ev},
            "region": {"score": region_score, **region_ev},
            "prnu": {"score": prnu_score, **prnu_ev},
            "combined_score": combined,
        }

        if combined >= 60:
            severity = Severity.HIGH
            summary = f"Significant noise inconsistencies (score: {combined})"
            detail = (
                f"Noise analysis found significant inconsistencies: "
                f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score}. "
                f"This may indicate image compositing or editing."
            )
        elif combined >= 35:
            severity = Severity.MEDIUM
            summary = f"Moderate noise irregularities (score: {combined})"
            detail = (
                f"Moderate noise irregularities: "
                f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score}."
            )
        else:
            severity = Severity.CLEAN
            summary = "Noise patterns are consistent"
            detail = (
                f"Noise patterns appear consistent: "
                f"wavelet={wavelet_score}, region={region_score}, PRNU={prnu_score}."
            )

        triggered = combined > 0 and combined >= 35
        if not triggered:
            return self._make_pass_result(
                self.name, self.check_id, detail, visual_outputs, evidence
            )

        return self._make_fail_result(
            self.name, self.check_id, combined, severity,
            summary, detail, visual_outputs, evidence
        )

    def _wavelet_noise(
        self, gray: np.ndarray, wavelet_name: str, levels: int
    ) -> Tuple[int, Dict]:
        """Multi-level wavelet noise estimation."""
        import pywt

        coeffs = pywt.wavedec2(gray, wavelet_name, level=levels)

        noise_levels = []
        for level in range(1, len(coeffs)):
            _, _, hh = coeffs[level]
            mad = float(np.median(np.abs(hh - np.median(hh))))
            sigma = mad / 0.6745
            noise_levels.append({
                "level": level,
                "mad": round(mad, 4),
                "sigma": round(sigma, 4),
            })

        if len(noise_levels) < 2:
            return 0, {"insufficient_levels": True}

        sigmas = [n["sigma"] for n in noise_levels]

        # Compare adjacent level ratios
        ratios = []
        for i in range(len(sigmas) - 1):
            if sigmas[i + 1] > 0:
                ratios.append(sigmas[i] / sigmas[i + 1])
            else:
                ratios.append(0)

        max_deviation = max(abs(r - 2.0) for r in ratios) if ratios else 0

        if max_deviation > 2.0:
            score = min(100, int(50 + max_deviation * 20))
        elif max_deviation > 1.0:
            score = int(20 + max_deviation * 15)
        else:
            score = 0

        evidence = {
            "noise_levels": noise_levels,
            "level_ratios": [round(r, 3) for r in ratios],
            "max_deviation": round(max_deviation, 3),
        }
        return score, evidence

    def _region_noise(
        self,
        gray: np.ndarray,
        original_image: Image.Image,
        block_size: int,
        outlier_z: float,
        min_blocks: int,
    ) -> Tuple[int, Dict, Optional[np.ndarray], List[Dict]]:
        """Per-block noise variance with visual bounding boxes."""
        h, w = gray.shape
        noise_map = np.zeros((h, w), dtype=np.float64)
        block_data = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]

                if np.std(block) < 2.0:
                    continue

                laplacian = cv2.Laplacian(block.astype(np.float64), cv2.CV_64F)
                noise_var = float(np.var(laplacian))
                noise_map[y:y + block_size, x:x + block_size] = noise_var

                block_data.append({
                    "x": int(x),
                    "y": int(y),
                    "width": block_size,
                    "height": block_size,
                    "noise_variance": round(noise_var, 2),
                })

        if len(block_data) < min_blocks:
            return 0, {"block_count": len(block_data)}, None, []

        variances = np.array([b["noise_variance"] for b in block_data])
        mean_var = float(np.mean(variances))
        std_var = float(np.std(variances))

        # Find outlier blocks
        outlier_regions = []
        for bd in block_data:
            if std_var > 0:
                z = abs(bd["noise_variance"] - mean_var) / std_var
            else:
                z = 0
            if z > outlier_z:
                outlier_regions.append({
                    **bd,
                    "label": f"z={z:.1f}",
                    "z_score": round(z, 2),
                })

        outlier_pct = len(outlier_regions) / len(block_data) * 100
        cv_var = std_var / mean_var if mean_var > 0 else 0

        if cv_var > 1.0 or outlier_pct > 15:
            score = min(100, int(50 + cv_var * 20 + outlier_pct))
        elif cv_var > 0.6 or outlier_pct > 8:
            score = int(20 + cv_var * 25 + outlier_pct * 2)
        else:
            score = 0

        evidence = {
            "block_count": len(block_data),
            "mean_noise_variance": round(mean_var, 2),
            "std_noise_variance": round(std_var, 2),
            "cv_variance": round(cv_var, 3),
            "outlier_count": len(outlier_regions),
            "outlier_pct": round(outlier_pct, 1),
        }

        return score, evidence, noise_map, outlier_regions

    def _prnu_analysis(
        self, gray: np.ndarray, original_image: Image.Image
    ) -> Tuple[int, Dict, Optional[Image.Image]]:
        """PRNU analysis with sliding window for finer spatial resolution."""
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.5)
        residual = gray - denoised

        # Sliding window analysis
        window = 64
        step = 32
        h, w = residual.shape
        energy_map = np.zeros_like(residual, dtype=np.float64)

        window_stats = []
        for y in range(0, h - window, step):
            for x in range(0, w - window, step):
                patch = residual[y:y + window, x:x + window]
                energy = float(np.sum(patch ** 2) / patch.size)
                std = float(np.std(patch))
                energy_map[y:y + window, x:x + window] = energy
                window_stats.append({
                    "x": x, "y": y, "energy": energy, "std": std
                })

        if len(window_stats) < 4:
            return 0, {"insufficient_data": True}, None

        energies = np.array([ws["energy"] for ws in window_stats])
        stds = np.array([ws["std"] for ws in window_stats])

        e_cv = float(np.std(energies) / np.mean(energies)) if np.mean(energies) > 0 else 0
        s_cv = float(np.std(stds) / np.mean(stds)) if np.mean(stds) > 0 else 0
        combined_cv = (e_cv + s_cv) / 2

        if combined_cv > 0.4:
            score = min(100, int(50 + combined_cv * 100))
        elif combined_cv > 0.2:
            score = int(20 + combined_cv * 60)
        else:
            score = 0

        # Visual: PRNU energy heatmap
        heatmap_img = draw_heatmap_overlay(original_image, energy_map, alpha=0.5)

        evidence = {
            "energy_cv": round(e_cv, 4),
            "std_cv": round(s_cv, 4),
            "combined_cv": round(combined_cv, 4),
            "window_count": len(window_stats),
        }

        return score, evidence, heatmap_img
