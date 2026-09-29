"""Colours, fonts, the application style sheet and SVG icons."""
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).parent / "assets"

NAVY = "#0E1A3A"
NAVY_2 = "#16254D"
BLUE = "#2F5BD8"
BLUE_SOFT = "#E6EDFF"
AMBER = "#FFB300"
GREEN = "#17A673"
GREEN_SOFT = "#E3F6EE"
RED = "#E0474C"
RED_SOFT = "#FDECEC"
BG = "#F3F5FA"
CARD = "#FFFFFF"
TEXT = "#141B2D"
MUTED = "#6B7489"
LINE = "#E3E7F0"
FONT = "Vazirmatn"


def load_fonts():
    for f in sorted((ASSETS / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))


def font(size=10, weight=QFont.Weight.Normal):
    f = QFont(FONT)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


STYLE = f"""
* {{ font-family: "{FONT}"; color: {TEXT}; }}
QMainWindow, #content {{ background: {BG}; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 6px 10px; border-radius: 6px; }}

#sidebar {{ background: {NAVY}; }}
#brand {{ color: white; font-size: 20pt; font-weight: 800; }}
#brandSub {{ color: #9FB0D9; font-size: 9pt; }}
#navButton {{ color: #C9D3EE; background: transparent; border: none; border-radius: 12px; padding: 12px 16px;
              text-align: right; font-size: 11pt; font-weight: 500; }}
#navButton:hover {{ background: {NAVY_2}; color: white; }}
#navButton:checked {{ background: {BLUE}; color: white; font-weight: 700; }}
#clock {{ color: white; font-size: 18pt; font-weight: 700; }}
#clockDate {{ color: #9FB0D9; font-size: 9.5pt; }}

#pageTitle {{ font-size: 20pt; font-weight: 800; }}
#pageSub {{ color: {MUTED}; font-size: 10pt; }}
#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; }}
#cardTitle {{ font-size: 12pt; font-weight: 700; }}
#muted {{ color: {MUTED}; }}
#kpiValue {{ font-size: 22pt; font-weight: 800; }}
#kpiLabel {{ color: {MUTED}; font-size: 10pt; }}
#bigMoney {{ font-size: 28pt; font-weight: 900; color: {BLUE}; }}

QPushButton {{ background: {BLUE_SOFT}; color: {BLUE}; border: none; border-radius: 10px; padding: 9px 16px;
               font-size: 10pt; font-weight: 600; }}
QPushButton:hover {{ background: #D6E1FF; }}
QPushButton:disabled {{ color: #9AA6C2; background: #EEF1F7; }}
QPushButton#primary {{ background: {BLUE}; color: white; }}
QPushButton#primary:hover {{ background: #264DC0; }}
QPushButton#success {{ background: {GREEN}; color: white; font-size: 12pt; padding: 12px 22px; }}
QPushButton#success:hover {{ background: #128A60; }}
QPushButton#danger {{ background: {RED_SOFT}; color: {RED}; }}
QPushButton#ghost {{ background: transparent; color: {MUTED}; }}
QPushButton#ghost:hover {{ background: #EBEEF5; }}

QLineEdit, QSpinBox, QComboBox, QDateEdit {{ background: white; border: 1px solid {LINE}; border-radius: 10px;
        padding: 8px 10px; font-size: 10.5pt; selection-background-color: {BLUE}; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {BLUE}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox QAbstractItemView {{ background: white; border: 1px solid {LINE}; selection-background-color: {BLUE_SOFT};
        selection-color: {TEXT}; outline: none; padding: 4px; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
QCheckBox {{ spacing: 10px; font-size: 10.5pt; }}
QCheckBox::indicator {{ width: 20px; height: 20px; border-radius: 6px; border: 2px solid #B8C2D9; background: white; }}
QCheckBox::indicator:checked {{ background: {BLUE}; border-color: {BLUE}; }}

QTableWidget {{ background: white; border: none; gridline-color: transparent; font-size: 10.5pt;
               selection-background-color: {BLUE_SOFT}; selection-color: {TEXT}; alternate-background-color: #FAFBFE; }}
QTableWidget::item {{ padding: 6px; border-bottom: 1px solid {LINE}; }}
QHeaderView::section {{ background: white; color: {MUTED}; border: none; border-bottom: 1px solid {LINE};
                        padding: 10px 6px; font-size: 9.5pt; font-weight: 600; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #CAD2E3; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QListWidget {{ background: transparent; border: none; font-size: 10pt; }}
QListWidget::item {{ padding: 8px 4px; border-bottom: 1px solid {LINE}; }}
QProgressBar {{ background: #E9EDF6; border: none; border-radius: 5px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 6px; background: #E3E8F3; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {BLUE}; border-radius: 3px; }}
QSlider::add-page:horizontal {{ background: #E3E8F3; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 3px solid {BLUE}; width: 14px; height: 14px; margin: -7px 0; border-radius: 10px; }}
QDialog {{ background: {BG}; }}
QMessageBox {{ background: white; }}
"""

# Material icons (24 x 24 paths)
ICONS = {
    "dashboard": "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
    "gate": "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z",
    "car": "M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z",
    "report": "M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z",
    "people": "M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z",
    "settings": "M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z",
    "login": "M11 7L9.6 8.4l2.6 2.6H2v2h10.2l-2.6 2.6L11 17l5-5-5-5zm9 12h-8v2h8c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2h-8v2h8v14z",
    "logout": "M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z",
    "print": "M19 8H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3zm-3 11H8v-5h8v5zm3-7c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm-1-9H6v4h12V3z",
    "image": "M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z",
    "edit": "M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a.996.996 0 0 0 0-1.41l-2.34-2.34a.996.996 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z",
    "play": "M8 5v14l11-7z",
    "stop": "M6 6h12v12H6z",
    "search": "M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z",
    "download": "M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z",
    "add": "M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z",
    "delete": "M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z",
    "parking": "M13 3H6v18h4v-6h3c3.31 0 6-2.69 6-6s-2.69-6-6-6zm.2 8H10V7h3.2c1.1 0 2 .9 2 2s-.9 2-2 2z",
    "wallet": "M21 18v1c0 1.1-.9 2-2 2H5c-1.11 0-2-.9-2-2V5c0-1.1.89-2 2-2h14c1.1 0 2 .9 2 2v1h-9a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h9zm-9-2h10V8H12v8zm4-2.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5z",
    "clock": "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z",
    "check": "M9 16.17L4.83 12l-1.42 1.42L9 19 21 7l-1.41-1.41z",
    "video": "M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z",
}


def pixmap(name, color="#FFFFFF", size=24):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="{color}" d="{ICONS[name]}"/></svg>'
    r = QSvgRenderer(QByteArray(svg.encode()))
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    r.render(p, QRectF(0, 0, size * 2, size * 2))
    p.end()
    pm.setDevicePixelRatio(2)
    return pm


def icon(name, color=BLUE, size=24):
    return QIcon(pixmap(name, color, size))


def app_icon(size=256):
    """Blue rounded square with a white P and amber scan corners."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(BLUE))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    s = size / 24
    r = QSvgRenderer(QByteArray(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="white" d="{ICONS["parking"]}"/></svg>'.encode()))
    r.render(p, QRectF(size * 0.2, size * 0.2, size * 0.6, size * 0.6))
    pen = p.pen()
    pen.setStyle(Qt.PenStyle.SolidLine)
    pen.setColor(QColor(AMBER))
    pen.setWidthF(s * 1.3)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    m, L = size * 0.12, size * 0.16
    for x, y, dx, dy in ((m, m, 1, 1), (size - m, m, -1, 1), (m, size - m, 1, -1), (size - m, size - m, -1, -1)):
        p.drawLine(int(x), int(y), int(x + dx * L), int(y))
        p.drawLine(int(x), int(y), int(x), int(y + dy * L))
    p.end()
    return pm
