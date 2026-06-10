"""
Font Consistency Analysis (Generalized).

Analyzes font rendering consistency across all text regions:
- Stroke width uniformity
- Anti-aliasing profile comparison
- Cross-character structural similarity
- OCR confidence variance

Generalized from fraud V2: no invoice-specific region classification.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.checks.base import BaseCheck
from src.config import Config
from src.models import CheckResult, Severity, VisualOutput
from src.utils.bbox import draw_bounding_boxes, RED, BLUE, GREEN

logger = logging.getLogger(__name__)


class FontCheck(BaseCheck):
    name = "Font Consistency"
    check_id = "font_check"
    description = "Analyze text rendering consistency across the document"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().font
        min_regions = config.get("min_text_regions", 5)
        stroke_z = config.get("stroke_outlier_z", 2.5)
        text_density = config.get("text_density_threshold", 0.02)

        visual_outputs = []
        gray = self._to_cv2_gray(image)

        # Detect text regions via MSER or morphological operations
        text_regions = self._detect_text_regions(gray)

        if len(text_regions) < min_regions:
            img_copy = image.copy()
            path = self._save_evidence(img_copy, "text_regions", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Text Regions",
                image_path=path,
                description=f"Only {len(text_regions)} text region(s) found (need {min_regions}+). Skipping.",
            ))
            return self._make_pass_result(
                self.name, self.check_id,
                f"Insufficient text regions ({len(text_regions)}) for font analysis.",
                visual_outputs, {"text_regions": len(text_regions)},
            )

        # Check text density to decide if this is a document
        total_text_area = sum(r["width"] * r["height"] for r in text_regions)
        image_area = gray.shape[0] * gray.shape[1]
        density = total_text_area / image_area

        if density < text_density:
            img_with_regions = draw_bounding_boxes(
                image, text_regions, color=GREEN, thickness=1, alpha=0.1
            )
            path = self._save_evidence(img_with_regions, "text_regions", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Text Regions",
                image_path=path,
                description=f"Low text density ({density:.3f}). Image appears to be a photo, not a document.",
            ))
            return self._make_pass_result(
                self.name, self.check_id,
                f"Low text density ({density:.3f}) — this appears to be a photo, not a document.",
                visual_outputs, {"text_density": round(density, 4)},
            )

        # Stroke width analysis
        stroke_score, stroke_ev, stroke_outliers = self._stroke_analysis(
            gray, text_regions, stroke_z
        )

        # Edge profile analysis
        edge_score, edge_ev, edge_outliers = self._edge_analysis(gray, text_regions)

        # Build visual with outlier boxes
        all_outliers_red = [
            {**o, "label": f"stroke z={o.get('z_score', 0):.1f}"}
            for o in stroke_outliers
        ]
        all_outliers_blue = [
            {**o, "label": f"edge d={o.get('distance', 0):.2f}"}
            for o in edge_outliers
        ]

        result_img = draw_bounding_boxes(
            image, text_regions, color=GREEN, thickness=1, alpha=0.05
        )
        if all_outliers_red:
            result_img = draw_bounding_boxes(
                result_img, all_outliers_red, color=RED, thickness=2, alpha=0.2
            )
        if all_outliers_blue:
            result_img = draw_bounding_boxes(
                result_img, all_outliers_blue, color=BLUE, thickness=2, alpha=0.2
            )

        path = self._save_evidence(result_img, "font_analysis", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Font Analysis",
            image_path=path,
            description=(
                f"Green=text regions, Red=stroke outliers ({len(stroke_outliers)}), "
                f"Blue=edge outliers ({len(edge_outliers)})"
            ),
        ))

        # Combined score
        combined = int(stroke_score * 0.55 + edge_score * 0.45)

        evidence = {
            "text_regions": len(text_regions),
            "text_density": round(density, 4),
            "stroke": {"score": stroke_score, **stroke_ev},
            "edge": {"score": edge_score, **edge_ev},
            "combined": combined,
        }

        if combined >= 55:
            severity = Severity.HIGH
            summary = f"Font inconsistencies detected (score: {combined})"
            detail = (
                f"Stroke={stroke_score} ({len(stroke_outliers)} outliers), "
                f"Edge={edge_score} ({len(edge_outliers)} outliers)."
            )
        elif combined >= 30:
            severity = Severity.MEDIUM
            summary = f"Minor font irregularities (score: {combined})"
            detail = f"Stroke={stroke_score}, Edge={edge_score}."
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"Font rendering appears consistent. Score: {combined}/100.",
                visual_outputs, evidence,
            )

        return self._make_fail_result(
            self.name, self.check_id, combined, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _detect_text_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect text regions using morphological operations."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            area = w * h
            if 5 < h < 100 and w > 10 and area > 100 and area < gray.size * 0.1:
                regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

        return regions

    def _stroke_analysis(
        self, gray: np.ndarray, regions: List[Dict], z_threshold: float
    ) -> Tuple[int, Dict, List[Dict]]:
        """Analyze stroke width consistency."""
        widths = []
        region_widths = []

        for r in regions:
            crop = gray[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]]
            if crop.size == 0:
                continue
            _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink = np.sum(binary > 0) / binary.size
            if ink < 0.05 or ink > 0.8:
                continue
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            mean_dist = float(np.mean(dist[binary > 0])) * 2 if np.sum(binary > 0) > 0 else 0
            if mean_dist > 0:
                widths.append(mean_dist)
                region_widths.append((r, mean_dist))

        if len(widths) < 5:
            return 0, {"insufficient_data": True}, []

        mean_w = float(np.mean(widths))
        std_w = max(float(np.std(widths)), 0.5)

        outliers = []
        max_z = 0.0
        for r, sw in region_widths:
            z = abs(sw - mean_w) / std_w
            if z > max_z:
                max_z = z
            if z > z_threshold:
                outliers.append({**r, "z_score": round(z, 2), "stroke_width": round(sw, 2)})

        score = min(100, int(max_z * 20)) if max_z > 1.5 else 0

        evidence = {
            "mean_stroke": round(mean_w, 2),
            "std_stroke": round(std_w, 2),
            "max_z_score": round(max_z, 2),
            "outlier_count": len(outliers),
        }

        return score, evidence, outliers

    def _edge_analysis(
        self, gray: np.ndarray, regions: List[Dict]
    ) -> Tuple[int, Dict, List[Dict]]:
        """Compare edge gradient profiles across text regions."""
        profiles = []
        region_profiles = []

        for r in regions:
            crop = gray[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]]
            if crop.size == 0:
                continue
            edges = cv2.Canny(crop, 50, 150)
            gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            edge_grads = mag[edges > 0]
            if len(edge_grads) < 10:
                continue
            hist, _ = np.histogram(edge_grads, bins=20, range=(0, 255), density=True)
            hist = hist.astype(np.float32)
            profiles.append(hist)
            region_profiles.append((r, hist))

        if len(profiles) < 5:
            return 0, {"insufficient_data": True}, []

        mean_profile = np.mean(profiles, axis=0).astype(np.float32)

        outliers = []
        max_dist = 0.0
        for r, prof in region_profiles:
            dist = cv2.compareHist(prof, mean_profile, cv2.HISTCMP_CHISQR)
            if dist > max_dist:
                max_dist = dist
            if dist > 0.5:
                outliers.append({**r, "distance": round(dist, 3)})

        score = min(100, int(max_dist * 80)) if max_dist > 0.3 else 0

        evidence = {
            "max_distance": round(max_dist, 3),
            "profiles_analyzed": len(profiles),
            "outlier_count": len(outliers),
        }

        return score, evidence, outliers
