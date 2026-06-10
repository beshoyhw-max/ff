"""
Results panel — scrollable list of check result cards with page selector
and overall verdict. Supports multi-page documents.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gui.check_card import CheckCard
from src.models import AnalysisResult, PageResult, Verdict


class ResultsPanel(QWidget):
    """Displays analysis results as a scrollable list of check cards."""

    page_changed = Signal(int)  # Emitted when user selects a different page

    def __init__(self, parent=None):
        super().__init__(parent)
        self._advanced = False
        self._cards: list[CheckCard] = []
        self._analysis_result: AnalysisResult = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Verdict header ──
        self._header = QFrame()
        self._header.setObjectName("card")
        self._header.setVisible(False)
        header_layout = QVBoxLayout(self._header)
        header_layout.setSpacing(4)

        self._verdict_label = QLabel()
        self._verdict_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self._verdict_label)

        self._score_label = QLabel()
        self._score_label.setObjectName("score_label")
        self._score_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self._score_label)

        # Stats row
        self._stats_row = QHBoxLayout()
        self._stats_row.setSpacing(16)

        self._time_label = QLabel()
        self._time_label.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        self._checks_label = QLabel()
        self._checks_label.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        self._triggered_label = QLabel()
        self._triggered_label.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )

        self._stats_row.addStretch()
        self._stats_row.addWidget(self._time_label)
        self._stats_row.addWidget(self._checks_label)
        self._stats_row.addWidget(self._triggered_label)
        self._stats_row.addStretch()
        header_layout.addLayout(self._stats_row)

        layout.addWidget(self._header)

        # ── Page selector (visible for multi-page) ──
        self._page_bar = QFrame()
        self._page_bar.setObjectName("card")
        self._page_bar.setVisible(False)
        page_bar_layout = QHBoxLayout(self._page_bar)
        page_bar_layout.setContentsMargins(12, 6, 12, 6)
        page_bar_layout.setSpacing(8)

        page_icon = QLabel("📄")
        page_icon.setStyleSheet("font-size: 16px; background: transparent;")
        page_bar_layout.addWidget(page_icon)

        page_label = QLabel("Page:")
        page_label.setStyleSheet(
            "font-size: 13px; color: #e6edf3; font-weight: 600; background: transparent;"
        )
        page_bar_layout.addWidget(page_label)

        self._page_combo = QComboBox()
        self._page_combo.setMinimumWidth(200)
        self._page_combo.setStyleSheet(
            "QComboBox { font-size: 13px; padding: 4px 8px; "
            "background: #21262d; color: #e6edf3; border: 1px solid #30363d; "
            "border-radius: 6px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #161b22; color: #e6edf3; "
            "border: 1px solid #30363d; }"
        )
        self._page_combo.currentIndexChanged.connect(self._on_page_selected)
        page_bar_layout.addWidget(self._page_combo)

        self._page_verdict = QLabel()
        self._page_verdict.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        page_bar_layout.addWidget(self._page_verdict)

        page_bar_layout.addStretch()
        layout.addWidget(self._page_bar)

        # ── Scrollable card list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 8, 0, 8)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()

        scroll.setWidget(self._cards_container)
        layout.addWidget(scroll)

        # ── Placeholder ──
        self._placeholder = QLabel(
            "Drop a file to start analysis\nor click Browse Files"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "font-size: 16px; color: #484f58; background: transparent; padding: 60px;"
        )
        layout.addWidget(self._placeholder)

    def set_advanced(self, advanced: bool) -> None:
        self._advanced = advanced

    def clear(self) -> None:
        """Remove all cards and reset header."""
        self._clear_cards()
        self._header.setVisible(False)
        self._page_bar.setVisible(False)
        self._page_combo.clear()
        self._placeholder.setVisible(True)
        self._analysis_result = None

    def _clear_cards(self) -> None:
        """Remove all cards from the layout."""
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

    def show_result(self, result: AnalysisResult) -> None:
        """Display a full analysis result."""
        self.clear()
        self._analysis_result = result
        self._placeholder.setVisible(False)
        self._header.setVisible(True)

        # Update verdict header
        self._update_verdict_header(result)

        # Page selector
        if result.page_count > 1:
            self._page_bar.setVisible(True)
            self._page_combo.blockSignals(True)
            for i, page in enumerate(result.pages):
                self._page_combo.addItem(
                    f"Page {i + 1} — {page.verdict_icon} {page.overall_verdict.value} "
                    f"(score: {page.overall_score})"
                )
            self._page_combo.blockSignals(False)
            # Show first page
            self._show_page_checks(result.pages[0])
        else:
            # Single page — show all checks directly
            self._page_bar.setVisible(False)
            if result.pages:
                self._show_page_checks(result.pages[0])

    def _show_page_checks(self, page: PageResult) -> None:
        """Display check cards for a specific page."""
        self._clear_cards()

        self._page_verdict.setText(
            f"{page.verdict_icon} Score: {page.overall_score}/100"
        )

        # Sort: triggered first, by severity descending
        sorted_checks = sorted(
            page.checks,
            key=lambda c: (-c.severity.rank, -c.score),
        )

        for check in sorted_checks:
            card = CheckCard(check, advanced=self._advanced)
            if check.triggered:
                card.set_expanded(True)
            self._cards.append(card)
            self._cards_layout.insertWidget(
                self._cards_layout.count() - 1, card
            )

    def _on_page_selected(self, index: int) -> None:
        """Handle page selection from combo box."""
        if self._analysis_result and 0 <= index < len(self._analysis_result.pages):
            self._show_page_checks(self._analysis_result.pages[index])
            self.page_changed.emit(index)

    def add_check_result(self, result, page_num: int = 0) -> None:
        """Add a single check card (for real-time updates during analysis)."""
        self._placeholder.setVisible(False)
        card = CheckCard(result, advanced=self._advanced)
        if result.triggered:
            card.set_expanded(True)
        self._cards.append(card)
        self._cards_layout.insertWidget(
            self._cards_layout.count() - 1, card
        )

    def update_header(self, result: AnalysisResult) -> None:
        """Update just the header verdict/score."""
        self._analysis_result = result
        self._header.setVisible(True)
        self._update_verdict_header(result)

        # Update page selector if multi-page
        if result.page_count > 1 and self._page_combo.count() == 0:
            self._page_bar.setVisible(True)
            self._page_combo.blockSignals(True)
            for i, page in enumerate(result.pages):
                self._page_combo.addItem(
                    f"Page {i + 1} — {page.verdict_icon} {page.overall_verdict.value} "
                    f"(score: {page.overall_score})"
                )
            self._page_combo.blockSignals(False)

    def _update_verdict_header(self, result: AnalysisResult) -> None:
        """Update the verdict header with overall result."""
        verdict_map = {
            Verdict.CLEAN: ("verdict_clean", "✅ CLEAN"),
            Verdict.SUSPICIOUS: ("verdict_suspicious", "⚠️ SUSPICIOUS"),
            Verdict.TAMPERED: ("verdict_tampered", "🔴 LIKELY TAMPERED"),
        }
        obj_name, text = verdict_map.get(
            result.overall_verdict, ("verdict_clean", "UNKNOWN")
        )
        self._verdict_label.setObjectName(obj_name)
        self._verdict_label.setText(text)
        self._verdict_label.style().unpolish(self._verdict_label)
        self._verdict_label.style().polish(self._verdict_label)

        self._score_label.setText(f"Overall Score: {result.overall_score}/100")
        self._time_label.setText(f"⏱ {result.analysis_time_seconds:.1f}s")

        total_checks = len(result.checks)
        triggered = len(result.triggered_checks)
        pages_text = f" across {result.page_count} page(s)" if result.page_count > 1 else ""
        self._checks_label.setText(f"📋 {total_checks} checks{pages_text}")
        self._triggered_label.setText(f"⚠️ {triggered} triggered")
