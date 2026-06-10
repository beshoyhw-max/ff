"""
Dark theme QSS stylesheet for the forgery detector GUI.
"""

DARK_THEME = """
/* ─── Global ─────────────────────────────────────── */
QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: "Segoe UI", "SF Pro Display", "Inter", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #0d1117;
}

/* ─── Scroll Areas ───────────────────────────────── */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #161b22;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #161b22;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 5px;
    min-width: 30px;
}

/* ─── Buttons ────────────────────────────────────── */
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #484f58;
}

QPushButton:pressed {
    background-color: #1a1f25;
}

QPushButton:disabled {
    color: #484f58;
    background-color: #161b22;
    border-color: #21262d;
}

QPushButton#primary_btn {
    background-color: #1f6feb;
    border-color: #1f6feb;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primary_btn:hover {
    background-color: #388bfd;
}

QPushButton#export_btn {
    background-color: #238636;
    border-color: #238636;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#export_btn:hover {
    background-color: #2ea043;
}

/* ─── Labels ─────────────────────────────────────── */
QLabel {
    color: #e6edf3;
    background-color: transparent;
}

QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}

QLabel#subtitle {
    font-size: 14px;
    color: #8b949e;
}

QLabel#verdict_clean {
    font-size: 28px;
    font-weight: 800;
    color: #00ff88;
}

QLabel#verdict_suspicious {
    font-size: 28px;
    font-weight: 800;
    color: #ffaa00;
}

QLabel#verdict_tampered {
    font-size: 28px;
    font-weight: 800;
    color: #ff2244;
}

QLabel#score_label {
    font-size: 18px;
    font-weight: 600;
    color: #00d2ff;
}

/* ─── Frames / Cards ─────────────────────────────── */
QFrame#card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 12px;
}

QFrame#card:hover {
    border-color: #30363d;
}

QFrame#card_triggered {
    background-color: #161b22;
    border: 1px solid #da3633;
    border-radius: 12px;
    padding: 12px;
}

QFrame#card_clean {
    background-color: #161b22;
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 12px;
}

QFrame#drop_zone {
    background-color: #161b22;
    border: 2px dashed #30363d;
    border-radius: 16px;
}

QFrame#drop_zone_hover {
    background-color: #1a2233;
    border: 2px dashed #1f6feb;
    border-radius: 16px;
}

QFrame#toolbar {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 8px;
}

QFrame#sidebar {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

/* ─── Progress Bar ───────────────────────────────── */
QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 6px;
    height: 12px;
    text-align: center;
    color: #ffffff;
    font-size: 10px;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #1f6feb,
        stop: 1 #00d2ff
    );
    border-radius: 6px;
}

/* ─── Tab Widget ─────────────────────────────────── */
QTabWidget::pane {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 4px;
}

QTabBar::tab {
    background-color: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 16px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #161b22;
    color: #e6edf3;
    border-color: #21262d;
}

QTabBar::tab:hover:!selected {
    background-color: #30363d;
    color: #e6edf3;
}

/* ─── Splitter ───────────────────────────────────── */
QSplitter::handle {
    background-color: #21262d;
    width: 2px;
}

QSplitter::handle:hover {
    background-color: #1f6feb;
}

/* ─── Toggle Switch (custom via checkbox) ────────── */
QCheckBox#mode_toggle {
    spacing: 8px;
    color: #8b949e;
}

QCheckBox#mode_toggle::indicator {
    width: 40px;
    height: 20px;
    border-radius: 10px;
    background-color: #30363d;
    border: 1px solid #484f58;
}

QCheckBox#mode_toggle::indicator:checked {
    background-color: #1f6feb;
    border-color: #1f6feb;
}

/* ─── Tooltip ────────────────────────────────────── */
QToolTip {
    background-color: #1c2128;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px;
    font-size: 12px;
}

/* ─── Text Edit (for detail/evidence display) ────── */
QTextEdit, QPlainTextEdit {
    background-color: #0d1117;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}

/* ─── Combo Box ──────────────────────────────────── */
QComboBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    color: #e6edf3;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #1f6feb;
}

/* ─── File Dialog ────────────────────────────────── */
QFileDialog {
    background-color: #0d1117;
}

/* ─── Group Box ──────────────────────────────────── */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #8b949e;
}
"""
