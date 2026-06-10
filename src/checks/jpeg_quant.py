"""
JPEG Quantization Table Analysis.

Detects double compression and quantization artifacts:
- Extract DCT quantization tables from JPEG
- JPEG ghost map at multiple quality levels
- Identify original compression quality via error dip

Completely new — not in fraud V2.
"""

import io
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import (
    create_grid_image,
    create_info_card,
    draw_heatmap_overlay,
    numpy_to_pil_heatmap,
)

logger = logging.getLogger(__name__)


class JPEGQuantCheck(BaseCheck):
    name = "JPEG Quantization"
    check_id = "jpeg_quant"
    description = "Analyze JPEG quantization tables and detect double compression"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().jpeg_quantization
        test_qualities = config.get("test_qualities", [50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
        visual_outputs = []

        # 1. Extract quantization tables
        qtables = self._extract_qtables(file_path) if file_path else None
        if qtables:
            qtable_card = create_info_card(
                {f"Table {i}": str(t[:8]) + "..." for i, t in enumerate(qtables)},
                title="Quantization Tables",
            )
            path = self._save_evidence(qtable_card, "qtables", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Q-Tables",
                image_path=path,
                description=f"Extracted {len(qtables)} quantization table(s) from JPEG",
            ))

        # 2. JPEG ghost analysis at multiple quality levels
        ghost_score, ghost_ev, ghost_images, error_plot = self._ghost_analysis(
            image, test_qualities
        )

        if error_plot:
            path = self._save_evidence(error_plot, "error_curve", evidence_store, page_num)
            visual_outputs.insert(0, VisualOutput(
                title="Error Curve",
                image_path=path,
                description=(
                    f"JPEG ghost error curve. Dip at Q={ghost_ev.get('estimated_quality', '?')} "
                    f"suggests original compression quality."
                ),
            ))

        if ghost_images:
            grid = create_grid_image(
                ghost_images,
                titles=[f"Q={q}" for q in test_qualities[:len(ghost_images)]],
                cols=5,
                cell_size=200,
            )
            path = self._save_evidence(grid, "ghost_maps", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Ghost Maps",
                image_path=path,
                description="JPEG ghost maps at each quality level",
            ))

        # 3. Double compression detection
        double_score, double_ev = self._detect_double_compression(
            image, test_qualities
        )

        combined = int(ghost_score * 0.5 + double_score * 0.5)

        evidence = {
            "ghost_analysis": {"score": ghost_score, **ghost_ev},
            "double_compression": {"score": double_score, **double_ev},
            "qtables_found": qtables is not None,
            "combined": combined,
        }

        if combined >= 50:
            severity = Severity.HIGH
            summary = f"Double compression detected (score: {combined})"
            detail = (
                f"Ghost score={ghost_score}, Double compression={double_score}. "
                f"Estimated original quality: Q={ghost_ev.get('estimated_quality', '?')}."
            )
        elif combined >= 25:
            severity = Severity.MEDIUM
            summary = f"Possible re-compression (score: {combined})"
            detail = f"Ghost score={ghost_score}, Double compression={double_score}."
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"No significant compression artifacts. Score: {combined}/100.",
                visual_outputs, evidence,
            )

        return self._make_fail_result(
            self.name, self.check_id, combined, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _extract_qtables(self, file_path: str) -> Optional[List[np.ndarray]]:
        """Extract JPEG quantization tables from file."""
        try:
            img = Image.open(file_path)
            if hasattr(img, "quantization") and img.quantization:
                tables = []
                for key in sorted(img.quantization.keys()):
                    table = np.array(list(img.quantization[key]), dtype=np.int32)
                    tables.append(table)
                return tables if tables else None
        except Exception as e:
            logger.debug(f"Q-table extraction failed: {e}")
        return None

    def _ghost_analysis(
        self, image: Image.Image, qualities: List[int],
    ) -> Tuple[int, Dict, List[Image.Image], Optional[Image.Image]]:
        """JPEG ghost analysis — find original compression quality."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original = np.array(image.convert("RGB"), dtype=np.float32)
        rgb = image.convert("RGB")
        errors = {}
        ghost_images = []

        for q in qualities:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
            diff = np.abs(original - recomp)
            mean_error = float(np.mean(diff))
            errors[q] = mean_error
            gray_diff = np.mean(diff, axis=2)
            ghost_img = numpy_to_pil_heatmap(gray_diff)
            ghost_images.append(ghost_img)

        error_vals = list(errors.values())
        min_error_idx = int(np.argmin(error_vals))
        estimated_quality = qualities[min_error_idx]
        min_error = error_vals[min_error_idx]

        has_dip = False
        if 0 < min_error_idx < len(error_vals) - 1:
            left = error_vals[min_error_idx - 1]
            right = error_vals[min_error_idx + 1]
            if min_error < left * 0.9 and min_error < right * 0.9:
                has_dip = True

        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#161b22")
        ax.set_facecolor("#0d1117")
        ax.plot(qualities, error_vals, "o-", color="#00d2ff", linewidth=2, markersize=5)
        ax.axvline(x=estimated_quality, color="#ff6644", linestyle="--", alpha=0.7, label=f"Est. Q={estimated_quality}")
        ax.set_xlabel("JPEG Quality", color="#8b949e", fontsize=10)
        ax.set_ylabel("Mean Error", color="#8b949e", fontsize=10)
        ax.set_title("JPEG Ghost Error Curve", color="#e6edf3", fontsize=12)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, alpha=0.2, color="#30363d")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=120)
        plt.close(fig)
        buf.seek(0)
        plot_img = Image.open(buf).convert("RGB")

        if has_dip:
            dip_depth = 1.0 - (min_error / max(error_vals[0], 0.01))
            score = min(100, int(40 + dip_depth * 60))
        else:
            score = 0

        evidence = {
            "estimated_quality": estimated_quality,
            "has_dip": has_dip,
            "min_error": round(min_error, 4),
            "quality_errors": {str(k): round(v, 4) for k, v in errors.items()},
        }

        return score, evidence, ghost_images, plot_img

    def _detect_double_compression(
        self, image: Image.Image, qualities: List[int],
    ) -> Tuple[int, Dict]:
        """Detect double JPEG compression via block artifact analysis."""
        gray = np.array(image.convert("L"), dtype=np.float64)
        h, w = gray.shape

        block_boundary_strength = 0.0
        count = 0

        for y in range(8, h - 8, 8):
            for x in range(0, w):
                diff = abs(gray[y, x] - gray[y - 1, x])
                block_boundary_strength += diff
                count += 1

        for x in range(8, w - 8, 8):
            for y in range(0, h):
                diff = abs(gray[y, x] - gray[y, x - 1])
                block_boundary_strength += diff
                count += 1

        avg_boundary = block_boundary_strength / count if count > 0 else 0

        non_boundary_strength = 0.0
        nb_count = 0
        for y in range(1, h - 1):
            if y % 8 == 0:
                continue
            for x in range(0, w, 16):
                diff = abs(gray[y, x] - gray[y - 1, x])
                non_boundary_strength += diff
                nb_count += 1

        avg_non_boundary = non_boundary_strength / nb_count if nb_count > 0 else 1
        boundary_ratio = avg_boundary / avg_non_boundary if avg_non_boundary > 0 else 1

        if boundary_ratio > 1.5:
            score = min(100, int(40 + (boundary_ratio - 1.0) * 40))
        elif boundary_ratio > 1.2:
            score = int(15 + (boundary_ratio - 1.0) * 30)
        else:
            score = 0

        evidence = {
            "avg_boundary_diff": round(avg_boundary, 3),
            "avg_non_boundary_diff": round(avg_non_boundary, 3),
            "boundary_ratio": round(boundary_ratio, 3),
        }

        return score, evidence
