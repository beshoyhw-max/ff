"""
Zoomable, pannable image viewer widget for evidence display.

Supports both PIL Image and disk-path loading.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PIL import Image
import numpy as np


class ImageViewer(QWidget):
    """Zoomable/pannable image viewer for forensic evidence images."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._drag_start = None
        self._min_zoom = 0.1
        self._max_zoom = 10.0

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def set_image(self, image: Image.Image) -> None:
        """Set the image from a PIL Image."""
        if image is None:
            self._pixmap = None
            self.update()
            return

        # Convert PIL Image to QPixmap
        image = image.convert("RGBA")
        data = np.array(image)
        h, w, ch = data.shape
        bytes_per_line = ch * w
        qimg = QImage(data.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(qimg.copy())

        self._fit_to_view()
        self.update()

    def set_image_from_path(self, image_path: str) -> None:
        """Load and display an image from a file path on disk."""
        if not image_path or not Path(image_path).exists():
            self._pixmap = None
            self.update()
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._pixmap = None
            self.update()
            return

        self._pixmap = pixmap
        self._fit_to_view()
        self.update()

    def clear(self) -> None:
        """Clear the displayed image and release pixmap."""
        self._pixmap = None
        self.update()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set image from QPixmap directly."""
        self._pixmap = pixmap
        self._fit_to_view()
        self.update()

    def _fit_to_view(self) -> None:
        """Fit image to widget size."""
        if self._pixmap is None:
            return
        vw = self.width() - 20
        vh = self.height() - 20
        if vw <= 0 or vh <= 0:
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return

        self._zoom = min(vw / pw, vh / ph, 1.0)
        self._pan_offset = QPointF(
            (vw - pw * self._zoom) / 2 + 10,
            (vh - ph * self._zoom) / 2 + 10,
        )

    def fit_to_view(self) -> None:
        """Public method to refit."""
        self._fit_to_view()
        self.update()

    def paintEvent(self, event) -> None:
        if self._pixmap is None:
            painter = QPainter(self)
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.NoBrush)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.translate(self._pan_offset)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom with mouse wheel."""
        if self._pixmap is None:
            return

        old_zoom = self._zoom
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        new_zoom = self._zoom * factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        # Zoom toward mouse position
        mouse_pos = event.position()
        img_pos = (mouse_pos - self._pan_offset) / old_zoom
        self._zoom = new_zoom
        self._pan_offset = mouse_pos - img_pos * new_zoom

        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            delta = event.position() - self._drag_start
            self._pan_offset += delta
            self._drag_start = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to fit to view."""
        self._fit_to_view()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._fit_to_view()
