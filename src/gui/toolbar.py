"""
Top toolbar with mode toggle, export button, and file info.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class Toolbar(QFrame):
    """Top toolbar widget."""

    mode_changed = Signal(bool)  # True = advanced mode
    export_requested = Signal()
    new_file_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # App title
        title = QLabel("🔍 Forgery Detector")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #00d2ff; background: transparent;"
        )
        layout.addWidget(title)

        layout.addStretch()

        # File info label
        self._file_info = QLabel()
        self._file_info.setStyleSheet(
            "font-size: 12px; color: #8b949e; background: transparent;"
        )
        layout.addWidget(self._file_info)

        layout.addStretch()

        # Mode toggle
        self._mode_toggle = QCheckBox("Advanced Mode")
        self._mode_toggle.setObjectName("mode_toggle")
        self._mode_toggle.setChecked(False)
        self._mode_toggle.toggled.connect(self.mode_changed.emit)
        layout.addWidget(self._mode_toggle)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #30363d;")
        layout.addWidget(sep)

        # New file button
        new_btn = QPushButton("📂 New File")
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.clicked.connect(self.new_file_requested.emit)
        layout.addWidget(new_btn)

        # Export button
        self._export_btn = QPushButton("📄 Export PDF")
        self._export_btn.setObjectName("export_btn")
        self._export_btn.setCursor(Qt.PointingHandCursor)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self._export_btn)

    def set_file_info(self, name: str, size_str: str, dimensions: str) -> None:
        self._file_info.setText(f"📎 {name}  •  {size_str}  •  {dimensions}")

    def set_export_enabled(self, enabled: bool) -> None:
        self._export_btn.setEnabled(enabled)

    def clear_file_info(self) -> None:
        self._file_info.setText("")
        self._export_btn.setEnabled(False)

    @property
    def is_advanced(self) -> bool:
        return self._mode_toggle.isChecked()
