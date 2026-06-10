"""
Drag-and-drop file input zone widget.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QSizePolicy,
)

from src.utils.image_loader import SUPPORTED_EXTENSIONS


class DropZone(QFrame):
    """Drag-and-drop file input area with browse button."""

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        # Icon
        icon_label = QLabel("🔍")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px; background: transparent;")
        layout.addWidget(icon_label)

        # Title
        title = QLabel("Drop image or PDF here")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Supports: JPG, PNG, BMP, TIFF, WebP, PDF"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #8b949e; background: transparent;")
        layout.addWidget(subtitle)

        # Browse button
        browse_btn = QPushButton("📁  Browse Files")
        browse_btn.setObjectName("primary_btn")
        browse_btn.setFixedWidth(180)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

    def _browse(self) -> None:
        """Open file dialog."""
        extensions = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image or PDF",
            "",
            f"Supported Files ({extensions});;All Files (*.*)",
        )
        if file_path:
            self.file_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self.setObjectName("drop_zone_hover")
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setObjectName("drop_zone")
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setObjectName("drop_zone")
        self.style().unpolish(self)
        self.style().polish(self)

        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                ext = Path(file_path).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self.file_selected.emit(file_path)
                    return
        event.ignore()
