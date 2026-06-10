"""
Data models for the Universal Forgery Detection system.

All check modules return CheckResult instances with disk-based visual evidence.
Multi-page PDFs produce one PageResult per page, each with its own checks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


class Severity(Enum):
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def color_hex(self) -> str:
        return {
            Severity.CLEAN: "#00ff88",
            Severity.LOW: "#88cc44",
            Severity.MEDIUM: "#ffaa00",
            Severity.HIGH: "#ff6644",
            Severity.CRITICAL: "#ff2244",
        }[self]

    @property
    def rank(self) -> int:
        return {
            Severity.CLEAN: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }[self]


class FileType(Enum):
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


class Verdict(Enum):
    CLEAN = "Clean"
    SUSPICIOUS = "Suspicious"
    TAMPERED = "Likely Tampered"


@dataclass
class VisualOutput:
    """A single evidence image produced by a check — stored on disk."""

    title: str
    image_path: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "image_path": self.image_path,
        }


@dataclass
class BoundingBox:
    """A region of interest on the image."""

    x: int
    y: int
    width: int
    height: int
    label: str = ""
    confidence: float = 0.0


@dataclass
class CheckResult:
    """Result from a single forensic check."""

    name: str
    check_id: str
    score: int  # 0-100
    severity: Severity
    triggered: bool
    summary: str
    detail: str
    visual_outputs: List[VisualOutput] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.triggered

    @property
    def status_icon(self) -> str:
        if self.error:
            return "⚠️"
        if not self.triggered:
            return "✅"
        if self.severity in (Severity.CRITICAL, Severity.HIGH):
            return "🔴"
        if self.severity == Severity.MEDIUM:
            return "🟡"
        return "🟠"


@dataclass
class PageResult:
    """Analysis results for a single page of a document."""

    page_number: int
    page_image_path: str
    page_size: Tuple[int, int]
    checks: List[CheckResult] = field(default_factory=list)
    overall_score: int = 0
    overall_verdict: Verdict = Verdict.CLEAN

    def compute_overall(self, weights: Dict[str, int] = None) -> None:
        """Compute page-level score and verdict from check results."""
        if not self.checks:
            return

        weights = weights or {}
        total_weight = 0
        weighted_sum = 0

        for check in self.checks:
            w = weights.get(check.check_id, 10)
            weighted_sum += check.score * w
            total_weight += w

        if total_weight > 0:
            self.overall_score = min(100, int(weighted_sum / total_weight))

        if self.overall_score >= 60:
            self.overall_verdict = Verdict.TAMPERED
        elif self.overall_score >= 30:
            self.overall_verdict = Verdict.SUSPICIOUS
        else:
            self.overall_verdict = Verdict.CLEAN

    @property
    def triggered_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if c.triggered]

    @property
    def highest_severity(self) -> Severity:
        if not self.checks:
            return Severity.CLEAN
        return max(self.checks, key=lambda c: c.severity.rank).severity

    @property
    def verdict_icon(self) -> str:
        icons = {
            Verdict.CLEAN: "✅",
            Verdict.SUSPICIOUS: "⚠️",
            Verdict.TAMPERED: "🔴",
        }
        return icons.get(self.overall_verdict, "❓")


@dataclass
class AnalysisResult:
    """Complete analysis result for a file (single or multi-page)."""

    file_path: str
    file_name: str
    file_type: FileType
    pages: List[PageResult] = field(default_factory=list)
    overall_score: int = 0
    overall_verdict: Verdict = Verdict.CLEAN
    analysis_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    session_id: str = ""

    @property
    def checks(self) -> List[CheckResult]:
        """Flat list of all checks across all pages."""
        return [c for p in self.pages for c in p.checks]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def compute_overall(self, weights: Dict[str, int] = None) -> None:
        """Compute document-level score from page scores (worst page wins)."""
        if not self.pages:
            return

        # Compute each page first
        for page in self.pages:
            page.compute_overall(weights)

        # Document verdict = worst page
        self.overall_score = max(p.overall_score for p in self.pages)

        if self.overall_score >= 60:
            self.overall_verdict = Verdict.TAMPERED
        elif self.overall_score >= 30:
            self.overall_verdict = Verdict.SUSPICIOUS
        else:
            self.overall_verdict = Verdict.CLEAN

    @property
    def triggered_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if c.triggered]

    @property
    def highest_severity(self) -> Severity:
        if not self.checks:
            return Severity.CLEAN
        return max(self.checks, key=lambda c: c.severity.rank).severity
