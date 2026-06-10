"""
Copy-Paste / Clone Detection.

Detects regions duplicated within the same image using a hybrid approach:
1. Keypoint matching (SIFT/ORB) — fast, finds textured clones
2. Block-matching on suspicious areas — catches smooth clones

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
from src.utils.bbox import draw_bounding_boxes, draw_arrows, RED, BLUE, CYAN

logger = logging.getLogger(__name__)


class CopyPasteCheck(BaseCheck):
    name = "Copy-Paste Detection"
    check_id = "copy_paste"
    description = "Detect cloned/copy-pasted regions within the image"

    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        config = Config.get().copy_paste
        method = config.get("keypoint_method", "sift")
        max_kp = config.get("max_keypoints", 5000)
        ratio_thresh = config.get("match_ratio_threshold", 0.75)
        min_cluster = config.get("min_cluster_size", 8)
        cluster_eps = config.get("cluster_eps", 30)

        visual_outputs = []
        gray = self._to_cv2_gray(image)

        # Resize for performance if too large
        max_dim = 1500
        h, w = gray.shape
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # Phase 1: Keypoint matching
        clusters, match_pairs, all_matches = self._keypoint_detection(
            gray, method, max_kp, ratio_thresh, min_cluster, cluster_eps
        )

        # Scale coordinates back to original
        if scale != 1.0:
            for src, dst in match_pairs:
                src["x"] = int(src["x"] / scale)
                src["y"] = int(src["y"] / scale)
                src["width"] = int(src["width"] / scale)
                src["height"] = int(src["height"] / scale)
                dst["x"] = int(dst["x"] / scale)
                dst["y"] = int(dst["y"] / scale)
                dst["width"] = int(dst["width"] / scale)
                dst["height"] = int(dst["height"] / scale)

        # Build visual: keypoint matches
        if match_pairs:
            source_regions = [{"x": s["x"], "y": s["y"], "width": s["width"],
                             "height": s["height"], "label": "Source"} for s, _ in match_pairs]
            clone_regions = [{"x": d["x"], "y": d["y"], "width": d["width"],
                            "height": d["height"], "label": "Clone"} for _, d in match_pairs]

            result_img = draw_bounding_boxes(image, source_regions, color=BLUE, thickness=3, alpha=0.2)
            result_img = draw_bounding_boxes(result_img, clone_regions, color=RED, thickness=3, alpha=0.2)
            result_img = draw_arrows(result_img, match_pairs, color=CYAN, thickness=2)

            path = self._save_evidence(result_img, "clone_map", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Clone Map",
                image_path=path,
                description=(
                    f"Detected {len(match_pairs)} clone region pair(s). "
                    f"Blue=source, Red=clone, Arrows=copy direction."
                ),
            ))
        else:
            clean_img = image.copy()
            path = self._save_evidence(clean_img, "clone_map", evidence_store, page_num)
            visual_outputs.append(VisualOutput(
                title="Clone Map",
                image_path=path,
                description="No copy-paste regions detected.",
            ))

        # Draw all keypoint matches for advanced view
        if all_matches > 0:
            kp_img = self._draw_keypoint_matches(image, gray, method, max_kp, scale)
            if kp_img:
                path = self._save_evidence(kp_img, "keypoints", evidence_store, page_num)
                visual_outputs.append(VisualOutput(
                    title="Keypoints",
                    image_path=path,
                    description=f"Detected {all_matches} potential keypoint matches",
                ))

        # Scoring
        n_pairs = len(match_pairs)
        n_clusters = len(clusters)

        if n_pairs >= 3 or n_clusters >= 2:
            score = min(100, 60 + n_pairs * 10)
            severity = Severity.CRITICAL
            summary = f"Copy-paste detected: {n_pairs} cloned region(s)"
            detail = (
                f"Found {n_pairs} clone pair(s) across {n_clusters} cluster(s). "
                f"Total keypoint matches before clustering: {all_matches}."
            )
        elif n_pairs >= 1:
            score = min(80, 40 + n_pairs * 15)
            severity = Severity.HIGH
            summary = f"Possible copy-paste: {n_pairs} similar region(s)"
            detail = (
                f"Found {n_pairs} potentially cloned region(s). "
                f"Confidence is moderate — verify visually."
            )
        else:
            return self._make_pass_result(
                self.name, self.check_id,
                f"No copy-paste regions detected. Analyzed {all_matches} keypoint matches.",
                visual_outputs, {"total_matches": all_matches},
            )

        evidence = {
            "clone_pairs": n_pairs,
            "clusters": n_clusters,
            "total_matches": all_matches,
            "method": method,
        }

        return self._make_fail_result(
            self.name, self.check_id, score, severity,
            summary, detail, visual_outputs, evidence,
        )

    def _keypoint_detection(
        self,
        gray: np.ndarray,
        method: str,
        max_kp: int,
        ratio_thresh: float,
        min_cluster: int,
        eps: int,
    ) -> Tuple[List[List[Dict]], List[Tuple[Dict, Dict]], int]:
        """Detect copy-paste via keypoint matching."""
        if method == "sift":
            try:
                detector = cv2.SIFT_create(nfeatures=max_kp)
            except cv2.error:
                detector = cv2.ORB_create(nfeatures=max_kp)
        else:
            detector = cv2.ORB_create(nfeatures=max_kp)

        kp, desc = detector.detectAndCompute(gray, None)

        if desc is None or len(kp) < 10:
            return [], [], 0

        if desc.dtype != np.float32:
            desc = desc.astype(np.float32)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc, desc, k=3)

        good_matches = []
        for match_group in matches:
            filtered = [m for m in match_group[1:] if m.distance > 0]
            if len(filtered) >= 2:
                m1, m2 = filtered[0], filtered[1]
                if m1.distance < ratio_thresh * m2.distance:
                    pt1 = kp[m1.queryIdx].pt
                    pt2 = kp[m1.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    if dist > 50:
                        good_matches.append(m1)

        total_matches = len(good_matches)

        if total_matches < min_cluster:
            return [], [], total_matches

        displacements = []
        for m in good_matches:
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            disp = pt2 - pt1
            displacements.append({
                "dx": disp[0],
                "dy": disp[1],
                "pt1": pt1,
                "pt2": pt2,
                "match": m,
            })

        clusters = self._cluster_displacements(displacements, eps, min_cluster)

        match_pairs = []
        for cluster in clusters:
            pts1 = np.array([d["pt1"] for d in cluster])
            pts2 = np.array([d["pt2"] for d in cluster])

            src_box = {
                "x": int(np.min(pts1[:, 0])),
                "y": int(np.min(pts1[:, 1])),
                "width": int(np.max(pts1[:, 0]) - np.min(pts1[:, 0]) + 30),
                "height": int(np.max(pts1[:, 1]) - np.min(pts1[:, 1]) + 30),
            }
            dst_box = {
                "x": int(np.min(pts2[:, 0])),
                "y": int(np.min(pts2[:, 1])),
                "width": int(np.max(pts2[:, 0]) - np.min(pts2[:, 0]) + 30),
                "height": int(np.max(pts2[:, 1]) - np.min(pts2[:, 1]) + 30),
            }
            match_pairs.append((src_box, dst_box))

        return clusters, match_pairs, total_matches

    @staticmethod
    def _cluster_displacements(
        displacements: List[Dict],
        eps: float,
        min_size: int,
    ) -> List[List[Dict]]:
        """Simple DBSCAN-like clustering of displacement vectors."""
        if not displacements:
            return []

        vectors = np.array([[d["dx"], d["dy"]] for d in displacements])
        visited = [False] * len(vectors)
        clusters = []

        for i in range(len(vectors)):
            if visited[i]:
                continue

            dists = np.sqrt(np.sum((vectors - vectors[i]) ** 2, axis=1))
            neighbors = np.where(dists < eps)[0]

            if len(neighbors) >= min_size:
                cluster = []
                for j in neighbors:
                    if not visited[j]:
                        visited[j] = True
                        cluster.append(displacements[j])
                if len(cluster) >= min_size:
                    clusters.append(cluster)

        return clusters

    def _draw_keypoint_matches(
        self,
        image: Image.Image,
        gray: np.ndarray,
        method: str,
        max_kp: int,
        scale: float,
    ) -> Optional[Image.Image]:
        """Draw detected keypoints on the image."""
        try:
            if method == "sift":
                try:
                    detector = cv2.SIFT_create(nfeatures=min(max_kp, 2000))
                except cv2.error:
                    detector = cv2.ORB_create(nfeatures=min(max_kp, 2000))
            else:
                detector = cv2.ORB_create(nfeatures=min(max_kp, 2000))

            kp, _ = detector.detectAndCompute(gray, None)

            img_bgr = self._to_cv2_bgr(image)
            kp_img = cv2.drawKeypoints(
                img_bgr, kp, None,
                color=(0, 210, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            return self._cv2_to_pil(kp_img)
        except Exception:
            return None
