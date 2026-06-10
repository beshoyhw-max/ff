"""
Main application window.

Layout: Toolbar on top, left panel with drop zone + file info + page preview,
right panel with scrollable check results (with page selector for multi-page).
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import Config
from src.gui.drop_zone import DropZone
from src.gui.image_viewer import ImageViewer
from src.gui.results_panel import ResultsPanel
from src.gui.toolbar import Toolbar
from src.gui.workers import AnalysisWorker
from src.models import AnalysisResult, FileType
from src.utils.image_loader import detect_file_type


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._config = Config.get()
        self._worker = None
        self._current_result: AnalysisResult = None

        gui = self._config.gui
        self.setWindowTitle("Forgery Detector — Universal Image & PDF Analysis")
        self.setMinimumSize(1000, 700)
        self.resize(gui.get("window_width", 1400), gui.get("window_height", 900))

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Toolbar ──
        self._toolbar = Toolbar()
        main_layout.addWidget(self._toolbar)

        # ── Body: splitter with left panel + right panel ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left panel: drop zone + preview + progress
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(12)

        self._drop_zone = DropZone()
        left_layout.addWidget(self._drop_zone)

        # Image/page preview
        self._preview = ImageViewer()
        self._preview.setMinimumHeight(200)
        self._preview.setVisible(False)
        left_layout.addWidget(self._preview)

        # File info
        self._file_info_label = QLabel()
        self._file_info_label.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent; padding: 4px;"
        )
        self._file_info_label.setWordWrap(True)
        self._file_info_label.setVisible(False)
        left_layout.addWidget(self._file_info_label)

        # Progress section
        self._progress_label = QLabel()
        self._progress_label.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        self._progress_label.setVisible(False)
        left_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        left_layout.addWidget(self._progress_bar)

        left_layout.addStretch()

        # Right panel: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 16, 16, 16)
        right_layout.setSpacing(0)

        self._results_panel = ResultsPanel()
        right_layout.addWidget(self._results_panel)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._drop_zone.file_selected.connect(self._on_file_selected)
        self._toolbar.mode_changed.connect(self._on_mode_changed)
        self._toolbar.export_requested.connect(self._on_export)
        self._toolbar.new_file_requested.connect(self._on_new_file)
        # Page selector → update preview
        self._results_panel.page_changed.connect(self._on_page_changed)

    def _on_file_selected(self, file_path: str) -> None:
        """Handle file selection — detect type and start analysis."""
        # Cancel any running analysis
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()

        file_type = detect_file_type(file_path)

        if file_type == FileType.UNKNOWN:
            QMessageBox.critical(self, "Error", "Unsupported file type.")
            return

        # Update UI
        self._drop_zone.setVisible(False)
        self._preview.setVisible(True)
        self._file_info_label.setVisible(True)

        # File info
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        size_str = _format_size(file_size)

        self._file_info_label.setText(
            f"📎 {file_name}\n"
            f"💾 {size_str}\n"
            f"📄 {file_type.value.upper()}"
        )
        self._toolbar.set_file_info(file_name, size_str, "")

        # Start analysis
        self._results_panel.clear()
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setVisible(True)
        self._progress_label.setText("Starting analysis...")

        metadata = {}

        self._worker = AnalysisWorker(
            file_path=file_path,
            file_type=file_type,
            metadata=metadata,
        )
        self._worker.page_started.connect(self._on_page_started)
        self._worker.check_started.connect(self._on_check_started)
        self._worker.check_completed.connect(self._on_check_completed)
        self._worker.page_completed.connect(self._on_page_completed)
        self._worker.progress.connect(self._on_progress)
        self._worker.all_completed.connect(self._on_all_completed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_page_started(self, page_num: int, total_pages: int) -> None:
        pages_str = f"Page {page_num + 1}/{total_pages}" if total_pages > 1 else ""
        self._progress_label.setText(f"Rendering {pages_str}...")

    def _on_check_started(self, name: str) -> None:
        self._progress_label.setText(f"Running: {name}...")

    def _on_check_completed(self, result, page_num: int) -> None:
        self._results_panel.add_check_result(result, page_num)

    def _on_page_completed(self, page_result) -> None:
        """When a page finishes — update the preview to show its rendered image."""
        if page_result and page_result.page_image_path:
            self._preview.set_image_from_path(page_result.page_image_path)

            # Update file info with page dimensions
            w, h = page_result.page_size
            info_text = self._file_info_label.text()
            if "📐" not in info_text:
                self._file_info_label.setText(
                    info_text + f"\n📐 {w}×{h}"
                )

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            pct = int((current / total) * 100)
            self._progress_bar.setValue(pct)
            self._progress_label.setText(
                f"Step {current + 1}/{total}..."
            )

    def _on_all_completed(self, result: AnalysisResult) -> None:
        self._current_result = result
        self._progress_bar.setValue(100)
        pages_str = f" ({result.page_count} pages)" if result.page_count > 1 else ""
        self._progress_label.setText(
            f"✅ Analysis complete — {result.analysis_time_seconds:.1f}s{pages_str}"
        )

        # Clear real-time cards and show organized result
        self._results_panel.show_result(result)
        self._toolbar.set_export_enabled(True)

        # Show first page preview
        if result.pages:
            self._preview.set_image_from_path(result.pages[0].page_image_path)

    def _on_page_changed(self, page_index: int) -> None:
        """When user selects a different page in the results panel."""
        if self._current_result and 0 <= page_index < len(self._current_result.pages):
            page = self._current_result.pages[page_index]
            self._preview.set_image_from_path(page.page_image_path)

    def _on_error(self, error_msg: str) -> None:
        self._progress_label.setText(f"❌ Error: {error_msg}")
        QMessageBox.warning(self, "Analysis Error", error_msg)

    def _on_mode_changed(self, advanced: bool) -> None:
        self._results_panel.set_advanced(advanced)
        # Re-render if we have results
        if self._current_result:
            self._results_panel.show_result(self._current_result)

    def _on_new_file(self) -> None:
        """Reset UI for a new file."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()

        self._current_result = None
        self._drop_zone.setVisible(True)
        self._preview.setVisible(False)
        self._preview.clear()
        self._file_info_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._results_panel.clear()
        self._toolbar.clear_file_info()

    def _on_export(self) -> None:
        """Export results as PDF."""
        if not self._current_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            f"forgery_report_{self._current_result.file_name}.pdf",
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return

        try:
            from src.export.pdf_report import generate_report

            generate_report(self._current_result, file_path)
            QMessageBox.information(
                self, "Export Complete", f"Report saved to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export report:\n{e}"
            )


def _format_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
