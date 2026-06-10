"""
Universal Forgery Detector — Entry Point.

Launch the PySide6 GUI application.
"""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from src.gui.main_window import MainWindow
from src.gui.styles import DARK_THEME


def main():
    app = QApplication(sys.argv)

    # Apply dark theme
    app.setStyleSheet(DARK_THEME)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
