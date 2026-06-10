"""
Collapsible check result card widget with lazy image loading.

Shows: status icon, check name, score, severity badge.
Expands to show: evidence images in tabs (loaded from disk on demand),
detail text, raw evidence.
Collapsing clears loaded images from memory.
"""

import json

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.image_viewer import ImageViewer
from src.models import CheckResult, Severity


class CheckCard(QFrame):
    """Collapsible card showing a single check result with lazy image loading."""

    def __init__(self, result: CheckResult, advanced: bool = False, parent=None):
        super().__init__(parent)
        self._result = result
        self._advanced = advanced
        self._expanded = False
        self._images_loaded = False

        # Set frame style based on result
        if result.triggered:
            self.setObjectName("card_triggered")
        else:
            self.setObjectName("card_clean")

        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(0)

        # ── Header row (always visible) ──
        header = QHBoxLayout()
        header.setSpacing(12)

        # Status icon
        icon = QLabel(result.status_icon)
        icon.setStyleSheet("font-size: 20px; background: transparent;")
        icon.setFixedWidth(30)
        header.addWidget(icon)

        # Check name
        name_label = QLabel(result.name)
        name_label.setStyleSheet(
            "font-size: 15px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        header.addWidget(name_label)

        header.addStretch()

        # Score badge
        score_color = result.severity.color_hex
        score_label = QLabel(f"{result.score}/100")
        score_label.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {score_color}; "
            f"background: rgba({_hex_to_rgb_str(score_color)}, 0.15); "
            f"padding: 4px 10px; border-radius: 10px;"
        )
        header.addWidget(score_label)

        # Severity badge
        sev_text = result.severity.value.upper()
        sev_label = QLabel(sev_text)
        sev_label.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {score_color}; "
            f"background: rgba({_hex_to_rgb_str(score_color)}, 0.1); "
            f"padding: 3px 8px; border-radius: 8px;"
        )
        header.addWidget(sev_label)

        # Expand arrow
        self._arrow = QLabel("▶")
        self._arrow.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        self._arrow.setFixedWidth(20)
        header.addWidget(self._arrow)

        main_layout.addLayout(header)

        # Summary line
        summary = QLabel(result.summary)
        summary.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent; "
            "padding-left: 42px; padding-top: 4px;"
        )
        summary.setWordWrap(True)
        main_layout.addWidget(summary)

        # ── Expandable content (built lazily) ──
        self._content = QWidget()
        self._content.setVisible(False)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(42, 12, 0, 0)
        self._content_layout.setSpacing(8)

        main_layout.addWidget(self._content)

        # Tab widget & viewers created on first expand
        self._tab_widget = None
        self._viewers = []

    def _build_content(self) -> None:
        """Build content widgets on first expansion (lazy)."""
        if self._images_loaded:
            return
        self._images_loaded = True

        result = self._result

        # Evidence images in tabs — load from disk paths
        if result.visual_outputs:
            self._tab_widget = QTabWidget()
            self._tab_widget.setMinimumHeight(350)
            self._tab_widget.setMaximumHeight(500)

            for vo in result.visual_outputs:
                viewer = ImageViewer()
                # Load image from disk path
                if vo.image_path:
                    viewer.set_image_from_path(vo.image_path)

                self._viewers.append(viewer)

                tab_widget = QWidget()
                tab_layout = QVBoxLayout(tab_widget)
                tab_layout.setContentsMargins(4, 4, 4, 4)
                tab_layout.addWidget(viewer)

                if vo.description:
                    desc = QLabel(vo.description)
                    desc.setStyleSheet(
                        "font-size: 11px; color: #8b949e; background: transparent;"
                    )
                    desc.setWordWrap(True)
                    tab_layout.addWidget(desc)

                self._tab_widget.addTab(tab_widget, vo.title)

            self._content_layout.addWidget(self._tab_widget)

        # Detail text
        if result.detail:
            detail_label = QLabel(result.detail)
            detail_label.setStyleSheet(
                "font-size: 12px; color: #c9d1d9; background: transparent; "
                "padding: 8px; line-height: 1.5;"
            )
            detail_label.setWordWrap(True)
            self._content_layout.addWidget(detail_label)

        # Duration
        if result.duration_ms > 0:
            dur = QLabel(f"⏱ {result.duration_ms:.0f}ms")
            dur.setStyleSheet(
                "font-size: 11px; color: #484f58; background: transparent;"
            )
            self._content_layout.addWidget(dur)

        # Raw evidence (advanced mode only)
        if self._advanced and result.evidence:
            evidence_text = QTextEdit()
            evidence_text.setPlainText(
                json.dumps(result.evidence, indent=2, default=str)
            )
            evidence_text.setReadOnly(True)
            evidence_text.setMaximumHeight(200)
            self._content_layout.addWidget(evidence_text)

    def _release_images(self) -> None:
        """Release loaded images from memory when collapsing."""
        for viewer in self._viewers:
            viewer.clear()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._toggle()

    def _toggle(self) -> None:
        self._expanded = not self._expanded

        if self._expanded:
            # Build content on first expand
            self._build_content()
            # Reload images if they were released
            if self._tab_widget and self._result.visual_outputs:
                for viewer, vo in zip(self._viewers, self._result.visual_outputs):
                    if vo.image_path and viewer._pixmap is None:
                        viewer.set_image_from_path(vo.image_path)
        else:
            # Release images from memory
            self._release_images()

        self._content.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._toggle()

    @property
    def result(self) -> CheckResult:
        return self._result


def _hex_to_rgb_str(hex_color: str) -> str:
    """Convert '#ff2244' to '255, 34, 68'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"
