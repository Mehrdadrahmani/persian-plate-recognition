"""ParkYar main window and controller: entry / exit logic, fees, dialogs, navigation."""
import argparse
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QLocale, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import __version__, billing, jalali, plates, theme
from .camera import GateWorker
from .db import Database
from .dialogs import ActivationDialog, CheckoutDialog, NotFoundDialog, PlateDialog
from .engine import PlateEngine
from .licensing import AUTHOR, CONTACT, License
from .pages import DashboardPage, GatesPage, ReportsPage, SettingsPage, SubscribersPage


def data_dir():
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ParkYar"


class Controller(QObject):
    refreshed = Signal()
    license_changed = Signal()

    def __init__(self, db, engine, lic=None):
        super().__init__()
        self.db, self.engine = db, engine
        self.license = lic or License(db.path.parent)
        self.letters = engine.letters
        self.settings = db.settings()
        self.window = None
        self._busy = False
        self._queue = []

    # ------------------------------------------------------------------ helpers
    def conf(self):
        """Detection threshold from the sensitivity setting (50 = the validated default)."""
        return min(0.9, max(0.2, self.engine.conf + (50 - int(self.settings["sensitivity"])) / 100 * 0.5))

    def make_worker(self, source):
        return GateWorker(source, self.engine, self.conf, float(self.settings["cooldown_s"]))

    def save_settings(self, values):
        self.db.save_settings(values)
        self.settings = self.db.settings()
        self.changed()

    def changed(self):
        self.refreshed.emit()

    def toast(self, text):
        if self.window:
            self.window.toast(text)

    def log_event(self, kind, text, plate=None):
        if self.window:
            self.window.gates.add_event(kind, text, plate)

    @property
    def tariff(self):
        return billing.Tariff.from_settings(self.settings)

    # ------------------------------------------------------------------ free trial / activation
    def show_activation(self, trial_over=False):
        if self.window:
            self.window.gates.stop_all()
        d = ActivationDialog(self.license, trial_over, self.window)
        d.exec()
        self.license_changed.emit()
        return self.license.can_recognize()

    def require_license(self):
        """True if plates may be recognised; otherwise offers activation."""
        return self.license.can_recognize() or self.show_activation(trial_over=True)

    # ------------------------------------------------------------------ gate logic
    def plate_seen(self, kind, text, crop=None, review=False):
        """A plate read at a gate (camera, photo). Low-confidence photo reads are confirmed by the operator first."""
        if not self.license.can_recognize():
            self.show_activation(trial_over=True)
            return
        if review or not plates.is_valid(text):
            d = PlateDialog("بررسی پلاک", self.letters, text if plates.is_valid(text) else "", crop,
                            "تصویر پلاک واضح نیست؛ شماره را بررسی و در صورت نیاز اصلاح کنید.", self.window)
            if not d.exec():
                return
            text = d.plate
        self.license.consume()
        self.license_changed.emit()
        if kind == "entry":
            self.register_entry(text, crop)
        else:
            self._queue.append((text, crop))
            self._next_exit()
        if not self.license.can_recognize():
            QTimer.singleShot(300, lambda: self.show_activation(trial_over=True))

    def manual(self, kind):
        d = PlateDialog("ثبت دستی ورود" if kind == "entry" else "ثبت دستی خروج", self.letters, parent=self.window)
        if d.exec():
            if kind == "entry":
                self.register_entry(d.plate, None, manual=True)
            else:
                self._queue.append((d.plate, None))
                self._next_exit()

    def register_entry(self, text, crop=None, manual=False):
        if self.db.open_session(text):
            self.log_event("warn", f"{plates.display(text)} از قبل داخل پارکینگ است", text)
            return None
        inside = len(self.db.open_sessions())
        img = self.db.save_image(crop, "in") if crop is not None and crop.size else None
        s = self.db.add_entry(text, img=img, manual=manual)
        note = " (مشترک)" if s.subscriber else ""
        self.log_event("entry", f"ورود {plates.display(text)}{note}", text)
        if inside + 1 > int(self.settings["capacity"]):
            self.log_event("warn", "ظرفیت پارکینگ تکمیل است")
        self.changed()
        return s

    def _next_exit(self):
        if self._busy or not self._queue:
            return
        self._busy = True
        try:
            text, crop = self._queue.pop(0)
            s = self.db.open_session(text)
            if s is None:
                similar = self.db.similar_open_sessions(text, 2)
                if len(similar) == 1 and plates.edit_distance(text, similar[0].plate) <= 1:
                    s = similar[0]  # one misread character: same car
                else:
                    d = NotFoundDialog(text, similar[:5], self.letters, crop, self.window)
                    if not d.exec():
                        self.log_event("warn", f"خروج {plates.display(text)} لغو شد (ورود ثبت نشده)", text)
                        return
                    if d.choice == "retry":
                        self._queue.insert(0, (d.plate, crop))
                        return
                    s = self.db.session(d.choice)
            self._checkout(s, crop)
        finally:
            self._busy = False
            if self._queue:
                QTimer.singleShot(0, self._next_exit)

    def checkout_session(self, sid):
        s = self.db.session(sid)
        if s and s.exit_time is None and not self._busy:
            self._busy = True
            try:
                self._checkout(s, None)
            finally:
                self._busy = False

    def _checkout(self, s, crop):
        d = CheckoutDialog(s, self.tariff, self.settings["lot_name"], crop, self.window, self.letters)
        if d.exec():
            if d.corrected:
                self.db.update_plate(s.id, d.corrected)
            img = self.db.save_image(crop, "out") if crop is not None and crop.size else None
            self.db.close_session(s.id, d.fee, d.exit_time, img)
            fee = "رایگان" if d.fee == 0 else jalali.money(d.fee)
            self.log_event("exit", f"خروج {plates.display(d.plate)} · {jalali.duration(d.minutes)} · {fee}", d.plate)
            self.changed()


class MainWindow(QMainWindow):
    PAGES = [("dashboard", "داشبورد"), ("gate", "ورود و خروج"), ("report", "گزارش‌ها"), ("people", "مشترکین"),
             ("settings", "تنظیمات")]

    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        ctl.window = self
        self.setWindowTitle("پارک‌یار — مدیریت هوشمند پارکینگ")
        self.setMinimumSize(1200, 760)
        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.dashboard = DashboardPage(ctl)
        self.gates = GatesPage(ctl)
        self.reports = ReportsPage(ctl)
        self.subs = SubscribersPage(ctl)
        self.settings = SettingsPage(ctl)
        for p in (self.dashboard, self.gates, self.reports, self.subs, self.settings):
            self.stack.addWidget(p)
        self._toast = QLabel(self)
        self._toast.setStyleSheet(f"background: {theme.NAVY}; color: white; border-radius: 12px; padding: 12px 20px; font-size: 10.5pt;")
        self._toast.hide()
        ctl.refreshed.connect(self.refresh_current)
        ctl.license_changed.connect(self.update_license)
        self.update_license()
        self.go(0)

    def _sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        v = QVBoxLayout(side)
        v.setContentsMargins(18, 26, 18, 22)
        v.setSpacing(6)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(theme.app_icon(96).scaled(44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        col = QVBoxLayout()
        col.setSpacing(0)
        t = QLabel("پارک‌یار")
        t.setObjectName("brand")
        s = QLabel("مدیریت هوشمند پارکینگ")
        s.setObjectName("brandSub")
        col.addWidget(t)
        col.addWidget(s)
        brand.addWidget(logo)
        brand.addLayout(col, 1)
        v.addLayout(brand)
        v.addSpacing(28)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for i, (ic, name) in enumerate(self.PAGES):
            b = QPushButton("  " + name)
            b.setObjectName("navButton")
            b.setCheckable(True)
            b.setIcon(theme.icon(ic, "#C9D3EE", 22))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.go(i))
            self.nav.addButton(b, i)
            v.addWidget(b)
        v.addStretch(1)
        box = QWidget()
        box.setStyleSheet(f"background: {theme.NAVY_2}; border-radius: 16px;")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(16, 14, 16, 14)
        self.clock = QLabel()
        self.clock.setObjectName("clock")
        self.clock_date = QLabel()
        self.clock_date.setObjectName("clockDate")
        off = QLabel("● پردازش آفلاین روی همین رایانه")
        off.setStyleSheet("color: #6EE7B7; font-size: 8.5pt;")
        for w in (self.clock, self.clock_date, off):
            w.setStyleSheet(w.styleSheet() + "background: transparent;")
            bl.addWidget(w)
        v.addWidget(box)
        self.lic_btn = QPushButton()
        self.lic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lic_btn.clicked.connect(lambda: self.ctl.show_activation())
        v.addSpacing(8)
        v.addWidget(self.lic_btn)
        ver = QLabel(f"نسخه {jalali.fa_digits(__version__)}")
        ver.setStyleSheet("color: #5F6F99; font-size: 8.5pt; padding-top: 8px;")
        v.addWidget(ver)
        mark = QLabel(f"© {AUTHOR}\n{CONTACT}")
        mark.setStyleSheet("color: #5F6F99; font-size: 8pt;")
        mark.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        mark.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(mark)
        self._tick()
        QTimer(self, interval=1000, timeout=self._tick).start()
        return side

    def update_license(self):
        lic = self.ctl.license
        if lic.activated:
            text, fg, bg = "✓  نسخه فعال‌شده", "#6EE7B7", "rgba(110,231,183,0.12)"
        elif lic.remaining:
            text, fg, bg = f"نسخه رایگان · {jalali.fa_digits(lic.remaining)} تشخیص باقی‌مانده", "#FFD166", "rgba(255,209,102,0.12)"
        else:
            text, fg, bg = "نسخه رایگان تمام شد · فعال‌سازی", "#FF8A8A", "rgba(255,138,138,0.14)"
        self.lic_btn.setText(text)
        self.lic_btn.setEnabled(not lic.activated)
        self.lic_btn.setStyleSheet(f"QPushButton {{ background: {bg}; color: {fg}; border-radius: 10px; padding: 8px;"
                                   f" font-size: 9pt; font-weight: 600; }} QPushButton:disabled {{ color: {fg}; background: {bg}; }}")
        if self.stack.currentWidget() is self.settings:
            self.settings.refresh()

    def _tick(self):
        now = time.time()
        self.clock.setText(jalali.time(now, True))
        self.clock_date.setText(jalali.date(now, weekday=True))

    def go(self, i):
        self.nav.button(i).setChecked(True)
        for j, b in enumerate(self.nav.buttons()):
            b.setIcon(theme.icon(self.PAGES[self.nav.id(b)][0], "#FFFFFF" if self.nav.id(b) == i else "#C9D3EE", 22))
        self.stack.setCurrentIndex(i)
        self.stack.currentWidget().refresh()

    def refresh_current(self):
        self.stack.currentWidget().refresh()

    def toast(self, text, ms=2600):
        self._toast.setText(text)
        self._toast.adjustSize()
        self._toast.move((self.width() - self._toast.width()) // 2 - 125, self.height() - self._toast.height() - 36)
        self._toast.show()
        self._toast.raise_()
        QTimer.singleShot(ms, self._toast.hide)

    def closeEvent(self, e):
        self.gates.stop_all()
        super().closeEvent(e)


def build_app(argv=None):
    ap = argparse.ArgumentParser(prog="parkyar")
    ap.add_argument("--demo", action="store_true", help="use a separate database filled with sample data")
    ap.add_argument("--data", help="data folder (default: the user's application-data folder)")
    ap.add_argument("--selftest", action="store_true", help="open every page, read a test image, print OK and exit")
    args, qt_args = ap.parse_known_args(argv)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName("ParkYar")
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    QLocale.setDefault(QLocale(QLocale.Language.Persian, QLocale.Country.Iran))  # Persian digits in number fields
    theme.load_fonts()
    app.setFont(theme.font(10))
    app.setStyleSheet(theme.STYLE)
    app.setWindowIcon(QIcon(theme.app_icon(256)))
    folder = Path(args.data) if args.data else data_dir()
    db = Database(folder / ("demo.sqlite" if args.demo else "parkyar.sqlite"))
    if args.demo:
        from .demo import seed

        seed(db)
    ctl = Controller(db, PlateEngine())
    win = MainWindow(ctl)
    win.selftest = args.selftest
    return app, win, ctl


def selftest(app, win, ctl):
    import numpy as np

    win.resize(1280, 800)
    win.show()
    for i in range(win.stack.count()):
        win.go(i)
        app.processEvents()
    img = np.full((720, 1280, 3), 90, np.uint8)
    ctl.engine.read(img)
    print(f"SELFTEST OK  pages={win.stack.count()}  engine={ctl.engine.last_ms:.0f} ms  db={ctl.db.path}", flush=True)
    return 0


def main(argv=None):
    app, win, ctl = build_app(argv)
    if win.selftest:
        code = selftest(app, win, ctl)
    else:
        win.resize(1440, 900)
        win.showMaximized()
        code = app.exec()
    ctl.db.con.close()
    sys.stdout.flush()
    # skip interpreter teardown: ONNX Runtime can abort while its threads are destroyed at exit
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
