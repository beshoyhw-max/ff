"""
Text Overlay Detection.

Detects text that was overlaid (added) on a photograph:
- Detect text regions via morphological operations
- Compare noise profile of text regions vs surrounding background
- Sharp text edges on blurred/noisy background = added text
- Uniform color blocks under text = rectangular fill before text

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
from src.utils.bbox import draw_bounding_boxes, draw_heatmap_overlay, RED, YELLOW, GREEN

logger = logging.getLogger(__name__)


class TextOverlayCheck(BaseCheck):
    name = "Text Overlay Detection"
    check_id = "text_overlay"
    description = "Detect text overlaid on photographs"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().text_overlay
        noise_ratio_thresh = config.get("noise_ratio_threshold", 2.0)
        min_text_height = config.get("min_text_height", 10)
        edge_thresh = config.get("edge_sharpness_threshold", 50)

        visual_outputs = []
        gray = self._to_cv2_gray(image)

        # Detect text-like regions
        text_regions = self._detect_text_candidates(gray, min_text_height)

        if len(text_regions) < 2:
            fallback = image.copy()
            path = self._save_evidence(fallback, "overlay_analysis", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Text Overlay",
                image_path=path,
                description=f"Only {len(text_regions)} text region(s) found. Insufficient for analysis.",
            ))
            return self._make_pass_result(
                self.name, self.check_id,
                "Insufficient text regions for overlay detection.",
                visual_outputs, {"text_regions": len(text_regions)},
            )

        # Analyze noise mismatch between text regions and their surroundings
        overlay_candidates = []
        noise_scores = []

        for region in text_regions:
            text_noise = self._region_noise(gray, region)
            bg_noise = self._background_noise(gray, region)

            if bg_noise > 0 and text_noise > 0:
                ratio = text_noise / bg_noise
                region["noise_ratio"] = round(ratio, 2)
                region["text_noise"] = round(text_noise, 2)
                region["bg_noise"] = round(bg_noise, 2)
                noise_scores.append(ratio)

                if ratio < 1.0 / noise_ratio_thresh:
                    region["label"] = f"overlay (r={ratio:.2f})"
                    overlay_candidates.append(region)
                elif ratio > noise_ratio_thresh:
                    region["label"] = f"suspicious (r={ratio:.2f})"
                    overlay_candidates.append(region)

        # Edge sharpness analysis
        sharp_regions = self._analyze_edge_sharpness(gray, text_regions, edge_thresh)
        overlay_candidates.extend(sharp_regions)
        overlay_candidates = self._deduplicate_regions(overlay_candidates)

        # Build visual
        result_img = draw_bounding_boxes(
            image, text_regions, color=GREEN, thickness=1, alpha=0.05
        )
        if overlay_candidates:
            result_img = draw_bounding_boxes(
                result_img, overlay_candidates, color=RED, thickness=2, alpha=0.2
            )

        path = self._save_evidence(result_img, "overlay_analysis", evidence_store, page_num)
        visual_outputs.append(VisualOutput(
            title="Overlay Analysis",
            image_path=path,
            description=(
                f"Green=all text regions ({len(text_regions)}), "
                f"Red=suspected overlays ({len(overlay_candidates)})."
            ),
        ))

        # Build noise comparison heatmap
        noise_map = self._build_noise_contrast_map(gray, text_regions)
        if noise_map is not None:
            heatmap = draw_heatmap_overlay(image, noise_map, alpha=0.4)
            path = self._save_evidence(heatmap, "noise_contrast", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Noise Contrast",
                image_path=path,
                description="Noise difference between text and background regions",
            ))

        # Scoring
        n_overlays = len(overlay_candidates)
        overlay_ratio = n_overlays / len(text_regions) * 100 if text_regions else 0

        if n_overlays >= 3 or overlay_ratio > 30:
            score = min(100, int(50 + n_overlays * 10))
            severity = Severity.HIGH
            summary = f"Text overlay detected: {n_overlays} suspicious region(s)"
            detail = (
                f"Found {n_overlays} text region(s) with noise profiles inconsistent "
                f"with their background ({overlay_ratio:.0f}% of text regions)."
            )
        elif n_overlays >= 1:
            score = min(50, int(20 + n_overlays * 10))
            severity = Severity.MEDIUM
            summary = f"Possible text overlay: {n_overlays} region(s)"
            detail = f"Found {n_overlays} potentially overlaid text region(s)."
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"No text overlays detected. Analyzed {len(text_regions)} text regions.",
                visual_outputs,
                {"text_regions": len(text_regions), "overlays": 0},
            )

        evidence = {
            "text_regions": len(text_regions),
            "overlay_candidates": n_overlays,
            "overlay_ratio_pct": round(overlay_ratio, 1),
            "noise_scores": noise_scores[:20],
        }

        return self._make_fail_result(
            self.name, self.check_id, score, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _detect_text_candidates(self, gray: np.ndarray, min_height: int) -> List[Dict]:
        """Detect text-like regions using MSER + morphological fallback."""
        regions = []
        try:
            mser = cv2.MSER_create()
            mser.setMinArea(50)
            mser.setMaxArea(int(gray.size * 0.05))
            mser_regions, _ = mser.detectRegions(gray)
            for pts in mser_regions:
                x, y, w, h = cv2.boundingRect(pts)
                if h >= min_height and w > h * 0.3:
                    regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        except Exception:
            pass

        if len(regions) < 5:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 2))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= min_height and w > 10:
                    regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

        return self._deduplicate_regions(regions)

    @staticmethod
    def _region_noise(gray: np.ndarray, region: Dict) -> float:
        """Compute noise level of a region using Laplacian variance."""
        y, x = region["y"], region["x"]
        h, w = region["height"], region["width"]
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0
        laplacian = cv2.Laplacian(crop.astype(np.float64), cv2.CV_64F)
        return float(np.std(laplacian))

    @staticmethod
    def _background_noise(gray: np.ndarray, region: Dict, margin: int = 20) -> float:
        """Compute noise level of the area surrounding a text region."""
        img_h, img_w = gray.shape
        y = max(0, region["y"] - margin)
        x = max(0, region["x"] - margin)
        y2 = min(img_h, region["y"] + region["height"] + margin)
        x2 = min(img_w, region["x"] + region["width"] + margin)
        crop = gray[y:y2, x:x2].copy()
        if crop.size == 0:
            return 0.0
        ry = region["y"] - y
        rx = region["x"] - x
        rh = region["height"]
        rw = region["width"]
        crop[ry:ry + rh, rx:rx + rw] = np.mean(crop)
        laplacian = cv2.Laplacian(crop.astype(np.float64), cv2.CV_64F)
        return float(np.std(laplacian))

    def _analyze_edge_sharpness(
        self, gray: np.ndarray, regions: List[Dict], threshold: int
    ) -> List[Dict]:
        """Find text regions with unnaturally sharp edges relative to background."""
        sharp_regions = []
        for r in regions:
            crop = gray[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]]
            if crop.size == 0:
                continue
            edges = cv2.Canny(crop, threshold, threshold * 2)
            edge_density = float(np.sum(edges > 0) / edges.size)
            bg_crop = self._get_bg_crop(gray, r)
            if bg_crop is not None and bg_crop.size > 0:
                bg_edges = cv2.Canny(bg_crop, threshold, threshold * 2)
                bg_edge_density = float(np.sum(bg_edges > 0) / bg_edges.size)
                if bg_edge_density > 0 and edge_density / bg_edge_density > 3.0:
                    sharp_regions.append({
                        **r,
                        "label": f"sharp edges ({edge_density:.3f} vs bg {bg_edge_density:.3f})",
                    })
        return sharp_regions

    @staticmethod
    def _get_bg_crop(gray: np.ndarray, region: Dict, margin: int = 30) -> Optional[np.ndarray]:
        """Get background crop around a region."""
        img_h, img_w = gray.shape
        y = max(0, region["y"] - margin)
        x = max(0, region["x"] - margin)
        y2 = min(img_h, region["y"] + region["height"] + margin)
        x2 = min(img_w, region["x"] + region["width"] + margin)
        if y2 - y < 10 or x2 - x < 10:
            return None
        return gray[y:y2, x:x2]

    @staticmethod
    def _build_noise_contrast_map(gray: np.ndarray, regions: List[Dict]) -> Optional[np.ndarray]:
        """Build a map showing noise contrast between text and background."""
        noise_map = np.zeros_like(gray, dtype=np.float64)
        for r in regions:
            y, x = r["y"], r["x"]
            h, w = r["height"], r["width"]
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            laplacian = cv2.Laplacian(crop.astype(np.float64), cv2.CV_64F)
            noise_map[y:y + h, x:x + w] = np.abs(laplacian)
        if np.max(noise_map) == 0:
            return None
        return noise_map

    @staticmethod
    def _deduplicate_regions(regions: List[Dict], overlap_thresh: float = 0.5) -> List[Dict]:
        """Remove overlapping regions."""
        if not regions:
            return []
        regions.sort(key=lambda r: r["width"] * r["height"], reverse=True)
        kept = []
        for r in regions:
            overlaps = False
            for k in kept:
                x1 = max(r["x"], k["x"])
                y1 = max(r["y"], k["y"])
                x2 = min(r["x"] + r["width"], k["x"] + k["width"])
                y2 = min(r["y"] + r["height"], k["y"] + k["height"])
                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    area_r = r["width"] * r["height"]
                    if area_r > 0 and inter / area_r > overlap_thresh:
                        overlaps = True
                        break
            if not overlaps:
                kept.append(r)
        return kept
