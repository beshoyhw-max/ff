"""
Base class for all forensic checks.

Every check must return a CheckResult with at least one VisualOutput.
Evidence images are saved to disk via EvidenceStore.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from src.models import CheckResult, Severity, VisualOutput

logger = logging.getLogger(__name__)


class BaseCheck(ABC):
    """Abstract base class for forensic checks."""

    name: str = "Unnamed Check"
    check_id: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        """
        Execute the forensic check.

        Args:
            image: Input image as PIL Image (RGB).
            file_path: Original file path (for EXIF/PDF metadata extraction).
            metadata: Pre-extracted metadata dict if available.
            evidence_store: EvidenceStore instance for saving visual outputs to disk.
            page_num: Page number (0-indexed) for multi-page documents.

        Returns:
            CheckResult with visual_outputs populated (image_path references).
        """

    def safe_run(
        self,
        image: Image.Image,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_store=None,
        page_num: int = 0,
    ) -> CheckResult:
        """Run with error handling and timing."""
        start = time.perf_counter()
        try:
            result = self.run(image, file_path, metadata, evidence_store, page_num)
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            logger.error(f"Check '{self.check_id}' failed: {e}", exc_info=True)
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult(
                name=self.name,
                check_id=self.check_id,
                score=0,
                severity=Severity.CLEAN,
                triggered=False,
                summary=f"Check failed: {e}",
                detail=f"An error occurred during {self.name}: {e}",
                error=str(e),
                duration_ms=elapsed,
            )

    def _save_evidence(self, image: Image.Image, name: str,
                       evidence_store=None, page_num: int = 0) -> str:
        """Save an evidence image to disk and return the path.

        If no evidence_store is provided, returns empty string.
        """
        if evidence_store is None:
            return ""
        return evidence_store.save_evidence(image, self.check_id, name, page_num)

    @staticmethod
    def _to_cv2_gray(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV grayscale numpy array."""
        return np.array(image.convert("L"))

    @staticmethod
    def _to_cv2_bgr(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR numpy array."""
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv2_to_pil(array: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR or grayscale array to PIL Image."""
        if len(array.shape) == 2:
            return Image.fromarray(array)
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _resize_for_analysis(
        image: Image.Image, max_dim: int = 2048
    ) -> Image.Image:
        """Resize image if larger than max_dim, preserving aspect ratio."""
        w, h = image.size
        if max(w, h) <= max_dim:
            return image
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _make_pass_result(
        name: str,
        check_id: str,
        detail: str,
        visual_outputs: List[VisualOutput],
        evidence: Dict[str, Any] = None,
    ) -> CheckResult:
        """Convenience: build a passing CheckResult."""
        return CheckResult(
            name=name,
            check_id=check_id,
            score=0,
            severity=Severity.CLEAN,
            triggered=False,
            summary="No issues detected",
            detail=detail,
            visual_outputs=visual_outputs,
            evidence=evidence or {},
        )

    @staticmethod
    def _make_fail_result(
        name: str,
        check_id: str,
        score: int,
        severity: Severity,
        summary: str,
        detail: str,
        visual_outputs: List[VisualOutput],
        evidence: Dict[str, Any] = None,
    ) -> CheckResult:
        """Convenience: build a failing CheckResult."""
        return CheckResult(
            name=name,
            check_id=check_id,
            score=min(100, max(0, score)),
            severity=severity,
            triggered=True,
            summary=summary,
            detail=detail,
            visual_outputs=visual_outputs,
            evidence=evidence or {},
        )
