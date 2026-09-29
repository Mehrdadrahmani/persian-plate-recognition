"""Pages: dashboard, gates (entry / exit cameras), reports, subscribers, settings."""
import csv
import datetime as dt
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QProgressBar, QScrollArea, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import billing, jalali, plates, theme
from .camera import read_image
from .dialogs import PlateDialog, SubscriberDialog
from .widgets import BarChart, CameraView, Card, Kpi, PlateWidget, bgr_to_qimage, button, label


def day_start(t=None):
    d = dt.datetime.fromtimestamp(t or time.time())
    return dt.datetime(d.year, d.month, d.day).timestamp()


class Page(QWidget):
    def __init__(self, title, subtitle=""):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("content")
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(32, 26, 32, 26)
        self.lay.setSpacing(18)
        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = label(title, "pageTitle")
        self.sub = label(subtitle, "pageSub")
        col.addWidget(self.title)
        col.addWidget(self.sub)
        head.addLayout(col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(10)
        head.addLayout(self.actions)
        self.lay.addLayout(head)

    def refresh(self):
        pass


def table(headers, stretch_col=None, plate_h=None):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setShowGrid(False)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    h = t.horizontalHeader()
    h.setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_col is not None:
        h.setSectionResizeMode(stretch_col, QHeaderView.ResizeMode.Stretch)
    if plate_h:  # cell widgets are not measured by ResizeToContents
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        t.setColumnWidth(0, int(plate_h * 4.4) + 24)
    return t


def cell(text, color=None, bold=False, align=None):
    it = QTableWidgetItem(text)
    it.setTextAlignment(align or (Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
    if color:
        it.setForeground(QColor(color))
    if bold:
        it.setFont(theme.font(10.5, QFont.Weight.Bold))
    return it


def pill(text, fg, bg):
    lb = QLabel(text)
    lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lb.setStyleSheet(f"background: {bg}; color: {fg}; border-radius: 10px; padding: 4px 12px; font-weight: 600;")
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(4, 0, 4, 0)
    h.addWidget(lb)
    h.addStretch(1)
    return w


def plate_cell(text, h=36):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(6, 4, 6, 4)
    lay.addWidget(PlateWidget(text, h))
    lay.addStretch(1)
    return w


STATUS = {
    "inside": ("داخل پارکینگ", theme.BLUE, theme.BLUE_SOFT),
    "paid": ("پرداخت شد", theme.GREEN, theme.GREEN_SOFT),
    "sub": ("مشترک", "#B7791F", "#FFF4DB"),
    "free": ("رایگان", theme.GREEN, theme.GREEN_SOFT),
    "unpaid": ("پرداخت نشده", theme.RED, theme.RED_SOFT),
}


def status_of(s):
    if s.exit_time is None:
        return "inside"
    if s.subscriber:
        return "sub"
    if not s.paid:
        return "unpaid"
    return "free" if not s.fee else "paid"


# ====================================================================== dashboard
class DashboardPage(Page):
    def __init__(self, ctl):
        super().__init__("داشبورد")
        self.ctl = ctl
        grid = QGridLayout()
        grid.setSpacing(18)
        self.k_inside = Kpi("car", "خودرو داخل پارکینگ")
        self.k_free = Kpi("parking", "جای خالی", theme.GREEN, theme.GREEN_SOFT)
        self.occ = QProgressBar()
        self.occ.setTextVisible(False)
        self.occ.setFixedHeight(10)
        self.k_free.lay.addWidget(self.occ)
        self.k_entries = Kpi("login", "ورود امروز", "#7C4DDB", "#EFE8FC")
        self.k_rev = Kpi("wallet", "درآمد امروز", "#D98A00", "#FFF4DB")
        for i, k in enumerate((self.k_inside, self.k_free, self.k_entries, self.k_rev)):
            grid.addWidget(k, 0, i)
        self.lay.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(18)
        chart_card = Card("درآمد ۷ روز اخیر")
        self.chart_total = label("", "muted")
        chart_card.header.addWidget(self.chart_total)
        self.chart = BarChart()
        chart_card.lay.addWidget(self.chart)
        inside_card = Card("خودروهای داخل پارکینگ")
        self.search = QLineEdit()
        self.search.setPlaceholderText("جستجوی پلاک…")
        self.search.addAction(theme.icon("search", theme.MUTED), QLineEdit.ActionPosition.LeadingPosition)
        self.search.setFixedWidth(220)
        self.search.textChanged.connect(self.refresh)
        inside_card.header.addWidget(self.search)
        self.table = table(["پلاک", "ورود", "مدت توقف", "هزینه تا این لحظه", ""], 3, 36)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 110)
        self.table.setMinimumHeight(420)
        inside_card.lay.addWidget(self.table)
        row.addWidget(inside_card, 3)
        row.addWidget(chart_card, 2)
        self.lay.addLayout(row, 1)
        self.timer = QTimer(self, interval=30_000, timeout=self.refresh)
        self.timer.start()

    def refresh(self):
        db, s = self.ctl.db, self.ctl.settings
        tariff = billing.Tariff.from_settings(s)
        now = time.time()
        today = day_start(now)
        self.sub.setText(f"{s['lot_name']}  ·  {jalali.date(now, weekday=True)}")
        open_ = db.open_sessions()
        cap = int(s["capacity"])
        self.k_inside.value.setText(jalali.fa_digits(len(open_)))
        self.k_free.value.setText(jalali.fa_digits(max(0, cap - len(open_))))
        self.k_free.caption.setText(f"جای خالی از {jalali.fa_digits(cap)}")
        self.occ.setMaximum(max(1, cap))
        self.occ.setValue(min(cap, len(open_)))
        self.k_entries.value.setText(jalali.fa_digits(db.entries_between(today, today + 86400)))
        rev, _ = db.revenue_between(today, today + 86400)
        self.k_rev.value.setText(jalali.money(rev, False))
        self.k_rev.caption.setText("درآمد امروز (تومان)")
        days = []
        for i in range(6, -1, -1):
            t0 = today - i * 86400
            days.append((jalali.WEEKDAYS[dt.datetime.fromtimestamp(t0 + 3600).weekday()] if i else "امروز",
                         db.revenue_between(t0, t0 + 86400)[0]))
        self.chart.set_data(days, lambda v: jalali.fa_digits(f"{v / 1e6:.1f}".replace(".", "٫")) + " م" if v >= 1e6 else jalali.fa_digits(f"{v // 1000}") + " ه")
        self.chart_total.setText("مجموع " + jalali.money(sum(v for _, v in days)) + "  (م = میلیون تومان)")
        q = plates.normalize(self.search.text())
        rows = [x for x in open_ if q in x.plate]
        self.table.setRowCount(len(rows))
        for r, x in enumerate(rows):
            self.table.setCellWidget(r, 0, plate_cell(x.plate))
            self.table.setItem(r, 1, cell(jalali.time(x.entry_time) if x.entry_time >= today else jalali.datetime(x.entry_time)))
            self.table.setItem(r, 2, cell(jalali.duration(x.minutes)))
            f = billing.fee(x.minutes, tariff)
            fee = "مشترک" if x.subscriber else jalali.money(f) if f else "رایگان"
            self.table.setItem(r, 3, cell(fee, bold=True))
            b = button("خروج", "logout", "ghost")
            b.clicked.connect(lambda _=False, sid=x.id: self.ctl.checkout_session(sid))
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 2, 4, 2)
            h.addWidget(b)
            self.table.setCellWidget(r, 4, w)
            self.table.setRowHeight(r, 52)


# ====================================================================== gates
class GatePanel(Card):
    SOURCES = [("وب‌کم ۱", "0"), ("وب‌کم ۲", "1"), ("فایل ویدیو یا تصویر…", "file"), ("دوربین شبکه (RTSP)…", "url")]

    def __init__(self, ctl, kind):
        self.ctl, self.kind = ctl, kind
        entry = kind == "entry"
        super().__init__("دوربین ورودی" if entry else "دوربین خروجی")
        color = theme.GREEN if entry else theme.RED
        badge = QLabel()
        badge.setPixmap(theme.pixmap("login" if entry else "logout", color, 22))
        self.header.insertWidget(0, badge)
        self.state = QLabel("خاموش")
        self._set_state("off")
        self.header.addWidget(self.state)
        self.view = CameraView("دوربین خاموش است — منبع تصویر را انتخاب و «شروع» را بزنید")
        self.view.setMinimumHeight(300)
        self.lay.addWidget(self.view, 1)

        ctr = QHBoxLayout()
        self.src = QComboBox()
        for name, _ in self.SOURCES:
            self.src.addItem(name)
        self.custom = None
        self.src.setMinimumWidth(170)
        self.src.activated.connect(self._source_picked)
        self.start_btn = button("شروع", "play", "primary")
        self.start_btn.clicked.connect(self.toggle)
        photo = button("عکس", "image")
        photo.setToolTip("خواندن پلاک از یک عکس")
        photo.clicked.connect(self.photo)
        manual = button("ثبت دستی", "edit")
        manual.clicked.connect(lambda: self.ctl.manual(self.kind))
        ctr.addWidget(self.src)
        ctr.addWidget(self.start_btn)
        ctr.addStretch(1)
        ctr.addWidget(photo)
        ctr.addWidget(manual)
        self.lay.addLayout(ctr)

        last = QHBoxLayout()
        self.last_plate = PlateWidget("", 50)
        self.last_info = label("هنوز پلاکی ثبت نشده", "muted")
        self.auto = QCheckBox("ثبت خودکار")
        self.auto.setChecked(bool(ctl.settings["auto_entry" if entry else "auto_exit"]))
        self.auto.toggled.connect(lambda v: ctl.save_settings({"auto_entry" if entry else "auto_exit": v}))
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("آخرین پلاک", "muted"))
        col.addWidget(self.last_info)
        last.addWidget(self.last_plate)
        last.addLayout(col, 1)
        last.addWidget(self.auto)
        self.lay.addLayout(last)
        self.worker = None
        src = str(ctl.settings["entry_source" if entry else "exit_source"])
        self._select_source(src)

    def _set_state(self, st):
        text, fg, bg = {"off": ("خاموش", theme.MUTED, "#EEF1F7"), "on": ("در حال پایش", theme.GREEN, theme.GREEN_SOFT),
                        "err": ("خطا", theme.RED, theme.RED_SOFT)}[st]
        self.state.setText(("● " if st == "on" else "") + text)
        self.state.setStyleSheet(f"background: {bg}; color: {fg}; border-radius: 10px; padding: 4px 12px; font-weight: 600;")

    def _select_source(self, s):
        for i, (_, v) in enumerate(self.SOURCES):
            if v == s:
                self.src.setCurrentIndex(i)
                return
        self.custom = s
        name = s if "://" in s else s.replace("\\", "/").split("/")[-1]
        if self.src.count() > len(self.SOURCES):
            self.src.setItemText(len(self.SOURCES), name)
        else:
            self.src.addItem(name)
        self.src.setCurrentIndex(len(self.SOURCES))

    def _source_picked(self, i):
        if i >= len(self.SOURCES):
            return
        v = self.SOURCES[i][1]
        if v == "file":
            f, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل", "", "ویدیو یا تصویر (*.mp4 *.avi *.mkv *.mov *.jpg *.jpeg *.png *.bmp)")
            if not f:
                return self._select_source(self.source())
            self._select_source(f)
        elif v == "url":
            u, ok = QInputDialog.getText(self, "دوربین شبکه", "نشانی جریان تصویر (مثلاً rtsp://user:pass@192.168.1.10/stream):")
            if not ok or not u.strip():
                return self._select_source(self.source())
            self._select_source(u.strip())
        self.ctl.save_settings({"entry_source" if self.kind == "entry" else "exit_source": self.source()})
        if self.worker:
            self.stop()
            self.start()

    def source(self):
        i = self.src.currentIndex()
        return self.custom if i >= len(self.SOURCES) else self.SOURCES[i][1]

    def toggle(self):
        self.stop() if self.worker else self.start()

    def start(self):
        if not self.ctl.require_license():
            return
        self.worker = self.ctl.make_worker(self.source())
        self.worker.frame.connect(self.view.set_frame)
        self.worker.confirmed.connect(self._confirmed)
        self.worker.failed.connect(self._failed)
        self.worker.start()
        self.start_btn.setText("توقف")
        self.start_btn.setIcon(theme.icon("stop", "#FFFFFF", 20))
        self._set_state("on")

    def stop(self):
        if self.worker:
            w, self.worker = self.worker, None
            w.frame.disconnect()
            w.stop()
        self.view.clear()
        self.start_btn.setText("شروع")
        self.start_btn.setIcon(theme.icon("play", "#FFFFFF", 20))
        self._set_state("off")

    def _failed(self, msg):
        self.stop()
        self._set_state("err")
        self.view.placeholder = msg
        self.view.update()

    def _confirmed(self, plate, frame):
        self.show_plate(plate.text)
        if self.auto.isChecked():
            self.ctl.plate_seen(self.kind, plate.text, plate.crop)

    def show_plate(self, text, note=None):
        self.last_plate.set_plate(text)
        self.last_info.setText(note or f"ساعت {jalali.time(time.time(), True)}")

    def photo(self):
        if not self.ctl.require_license():
            return
        f, _ = QFileDialog.getOpenFileName(self, "انتخاب عکس", "", "تصویر (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not f:
            return
        img = read_image(f)
        if img is None:
            return
        found = self.ctl.engine.read(img, self.ctl.conf())
        if self.worker:
            self.stop()
        self.view.set_frame(bgr_to_qimage(img), found)
        if not found:
            self.last_info.setText("پلاکی در تصویر پیدا نشد")
            return
        best = max(found, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
        self.show_plate(best.text)
        self.ctl.plate_seen(self.kind, best.text, best.crop, review=best.conf < 0.5 or not plates.is_valid(best.text))


class GatesPage(Page):
    def __init__(self, ctl):
        super().__init__("ورود و خروج", "پلاک خودروها در ورودی و خروجی خوانده و به‌طور خودکار ثبت می‌شود")
        self.ctl = ctl
        row = QHBoxLayout()
        row.setSpacing(18)
        self.entry = GatePanel(ctl, "entry")
        self.exit = GatePanel(ctl, "exit")
        row.addWidget(self.entry)
        row.addWidget(self.exit)
        self.lay.addLayout(row, 3)
        log = Card("رویدادهای اخیر")
        self.events = QListWidget()
        self.events.setMinimumHeight(170)
        log.lay.addWidget(self.events)
        self.lay.addWidget(log, 1)

    def add_event(self, kind, text, plate=None):
        color, ic = {"entry": (theme.GREEN, "login"), "exit": (theme.BLUE, "logout"), "warn": (theme.RED, "clock")}[kind]
        it = QListWidgetItem(theme.icon(ic, color, 20), f"{jalali.time(time.time(), True)}   {text}")
        self.events.insertItem(0, it)
        while self.events.count() > 200:
            self.events.takeItem(self.events.count() - 1)
        panel = self.entry if kind == "entry" else self.exit if kind == "exit" else None
        if panel and plate:
            panel.show_plate(plate, text)

    def stop_all(self):
        self.entry.stop()
        self.exit.stop()


# ====================================================================== reports
class ReportsPage(Page):
    RANGES = [("امروز", 1), ("۷ روز اخیر", 7), ("۳۰ روز اخیر", 30), ("همه", 0)]

    def __init__(self, ctl):
        super().__init__("گزارش‌ها", "همه ورود و خروج‌ها، درآمد و مدت توقف")
        self.ctl = ctl
        self.range = QComboBox()
        for n, _ in self.RANGES:
            self.range.addItem(n)
        self.range.setCurrentIndex(1)
        self.range.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("جستجوی پلاک…")
        self.search.addAction(theme.icon("search", theme.MUTED), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self.refresh)
        exp = button("خروجی اکسل (CSV)", "download", "primary")
        exp.clicked.connect(self.export)
        for w in (self.search, self.range, exp):
            self.actions.addWidget(w)
        grid = QGridLayout()
        grid.setSpacing(18)
        self.k_count = Kpi("car", "مراجعه")
        self.k_rev = Kpi("wallet", "درآمد (تومان)", "#D98A00", "#FFF4DB")
        self.k_avg = Kpi("clock", "میانگین توقف", "#7C4DDB", "#EFE8FC")
        self.k_sub = Kpi("people", "مراجعه مشترکین", theme.GREEN, theme.GREEN_SOFT)
        for i, k in enumerate((self.k_count, self.k_rev, self.k_avg, self.k_sub)):
            grid.addWidget(k, 0, i)
        self.lay.addLayout(grid)
        card = Card("فهرست مراجعات")
        self.table = table(["پلاک", "ورود", "خروج", "مدت توقف", "مبلغ", "وضعیت"], 5, 32)
        self.table.setMinimumHeight(460)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        card.lay.addWidget(self.table)
        card.lay.addWidget(label("برای اصلاح پلاک یا حذف یک ردیف، روی آن راست‌کلیک کنید.", "muted"))
        self.lay.addWidget(card, 1)
        self.rows = []

    def _sessions(self):
        days = self.RANGES[self.range.currentIndex()][1]
        t1 = day_start() + 86400
        t0 = t1 - days * 86400 if days else 0
        return self.ctl.db.sessions_between(t0, t1, self.search.text())

    def refresh(self):
        rows = self.rows = self._sessions()
        closed = [s for s in rows if s.exit_time]
        self.k_count.value.setText(jalali.fa_digits(len(rows)))
        self.k_rev.value.setText(jalali.money(sum(s.fee or 0 for s in closed if s.paid), False))
        avg = sum(s.minutes for s in closed) / len(closed) if closed else 0
        self.k_avg.value.setText(jalali.duration(avg) if closed else "—")
        self.k_sub.value.setText(jalali.fa_digits(sum(1 for s in rows if s.subscriber)))
        shown = rows[:500]
        self.table.setRowCount(len(shown))
        for r, s in enumerate(shown):
            self.table.setCellWidget(r, 0, plate_cell(s.plate, 32))
            self.table.setItem(r, 1, cell(jalali.datetime(s.entry_time)))
            self.table.setItem(r, 2, cell(jalali.datetime(s.exit_time) if s.exit_time else "—"))
            self.table.setItem(r, 3, cell(jalali.duration(s.minutes)))
            self.table.setItem(r, 4, cell(jalali.money(s.fee) if s.fee else "—", bold=bool(s.fee)))
            self.table.setCellWidget(r, 5, pill(*STATUS[status_of(s)]))
            self.table.setRowHeight(r, 46)

    def _menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0 or r >= len(self.rows):
            return
        s = self.rows[r]
        m = QMenu(self)
        fix = m.addAction(theme.icon("edit", theme.BLUE, 18), "اصلاح پلاک")
        dele = m.addAction(theme.icon("delete", theme.RED, 18), "حذف این ردیف")
        a = m.exec(self.table.viewport().mapToGlobal(pos))
        if a == fix:
            d = PlateDialog("اصلاح پلاک", self.ctl.letters, s.plate, crop=s.entry_img, parent=self)
            if d.exec():
                self.ctl.db.update_plate(s.id, d.plate)
                self.ctl.changed()
        elif a == dele:
            if QMessageBox.question(self, "حذف", "این ردیف برای همیشه حذف شود؟") == QMessageBox.StandardButton.Yes:
                self.ctl.db.delete_session(s.id)
                self.ctl.changed()

    def export(self):
        f, _ = QFileDialog.getSaveFileName(self, "ذخیره گزارش", f"parking_report_{time.strftime('%Y%m%d')}.csv", "CSV (*.csv)")
        if not f:
            return
        letters, lat = self.ctl.letters, self.ctl.engine.cfg["letters_latin"]
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel shows Persian correctly
            w = csv.writer(fh)
            w.writerow(["پلاک", "plate_latin", "تاریخ ورود", "ساعت ورود", "تاریخ خروج", "ساعت خروج", "مدت (دقیقه)", "مبلغ (تومان)", "وضعیت"])
            for s in self._sessions():
                w.writerow([plates.display(s.plate), plates.latin(s.plate, letters, lat), jalali.short_date(s.entry_time),
                            jalali.time(s.entry_time), jalali.short_date(s.exit_time) if s.exit_time else "",
                            jalali.time(s.exit_time) if s.exit_time else "", int(round(s.minutes)), s.fee or 0,
                            STATUS[status_of(s)][0]])
        self.ctl.toast(f"گزارش ذخیره شد: {f}")


# ====================================================================== subscribers
class SubscribersPage(Page):
    def __init__(self, ctl):
        super().__init__("مشترکین", "خودروهای مشترک بدون پرداخت وارد و خارج می‌شوند")
        self.ctl = ctl
        self.search = QLineEdit()
        self.search.setPlaceholderText("جستجوی پلاک یا نام…")
        self.search.addAction(theme.icon("search", theme.MUTED), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self.refresh)
        add = button("مشترک جدید", "add", "primary")
        add.clicked.connect(lambda: self.edit(None))
        self.actions.addWidget(self.search)
        self.actions.addWidget(add)
        card = Card("فهرست مشترکین")
        self.table = table(["پلاک", "نام", "تلفن", "اعتبار تا", "وضعیت", ""], 1, 34)
        for col, w in ((4, 150), (5, 110)):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, w)
        self.table.setMinimumHeight(520)
        card.lay.addWidget(self.table)
        self.lay.addWidget(card, 1)

    def refresh(self):
        subs = self.ctl.db.subscribers(self.search.text())
        self.table.setRowCount(len(subs))
        for r, s in enumerate(subs):
            self.table.setCellWidget(r, 0, plate_cell(s.plate, 34))
            self.table.setItem(r, 1, cell(s.name or "—", bold=True))
            self.table.setItem(r, 2, cell(jalali.fa_digits(s.phone) or "—"))
            self.table.setItem(r, 3, cell(jalali.date(s.valid_until)))
            left = (s.valid_until - time.time()) / 86400
            if left < 0:
                st = ("منقضی شده", theme.RED, theme.RED_SOFT)
            elif left < 7:
                st = (f"{jalali.fa_digits(int(left) + 1)} روز مانده", "#B7791F", "#FFF4DB")
            else:
                st = ("فعال", theme.GREEN, theme.GREEN_SOFT)
            self.table.setCellWidget(r, 4, pill(*st))
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 2, 4, 2)
            e = button("", "edit", "ghost")
            e.setToolTip("ویرایش")
            e.clicked.connect(lambda _=False, s=s: self.edit(s))
            d = button("", "delete", "ghost", theme.RED)
            d.setToolTip("حذف")
            d.clicked.connect(lambda _=False, s=s: self.delete(s))
            h.addWidget(e)
            h.addWidget(d)
            self.table.setCellWidget(r, 5, w)
            self.table.setRowHeight(r, 50)

    def edit(self, sub):
        d = SubscriberDialog(self.ctl.letters, sub, self)
        if d.exec():
            try:
                self.ctl.db.save_subscriber(d.inp.text(), d.name.text().strip(), plates.normalize(d.phone.text()), d.valid_until,
                                            sub.id if sub else None)
            except Exception:
                QMessageBox.warning(self, "مشترکین", "این پلاک قبلاً ثبت شده است.")
            self.ctl.changed()

    def delete(self, sub):
        if QMessageBox.question(self, "حذف مشترک", f"مشترک {plates.display(sub.plate)} حذف شود؟") == QMessageBox.StandardButton.Yes:
            self.ctl.db.delete_subscriber(sub.id)
            self.ctl.changed()


# ====================================================================== settings
class SettingsPage(Page):
    def __init__(self, ctl):
        super().__init__("تنظیمات")
        self.ctl = ctl
        save = button("ذخیره تغییرات", "check", "primary")
        save.clicked.connect(self.save)
        self.actions.addWidget(save)

        def spin(lo, hi, step, suffix=""):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setGroupSeparatorShown(True)
            s.setSuffix(suffix)
            s.setMinimumWidth(200)
            s.setAlignment(Qt.AlignmentFlag.AlignLeft)
            return s

        row = QHBoxLayout()
        row.setSpacing(18)
        lot = Card("پارکینگ")
        g = QGridLayout()
        g.setVerticalSpacing(14)
        self.name = QLineEdit()
        self.capacity = spin(1, 100000, 10, "  خودرو")
        g.addWidget(label("نام پارکینگ (روی رسید چاپ می‌شود)", "muted"), 0, 0)
        g.addWidget(self.name, 0, 1)
        g.addWidget(label("ظرفیت", "muted"), 1, 0)
        g.addWidget(self.capacity, 1, 1)
        lot.lay.addLayout(g)
        lot.lay.addStretch(1)

        tariff = Card("تعرفه")
        g = QGridLayout()
        g.setVerticalSpacing(14)
        self.free = spin(0, 240, 5, "  دقیقه")
        self.first = spin(0, 100_000_000, 5000, "  تومان")
        self.extra = spin(0, 100_000_000, 5000, "  تومان")
        self.cap = spin(0, 1_000_000_000, 10000, "  تومان")
        for i, (k, w) in enumerate((("توقف رایگان تا", self.free), ("ساعت اول", self.first),
                                    ("هر ساعت بعد", self.extra), ("سقف هر شبانه‌روز (۰ = بدون سقف)", self.cap))):
            g.addWidget(label(k, "muted"), i, 0)
            g.addWidget(w, i, 1)
            w.valueChanged.connect(self._example)
        tariff.lay.addLayout(g)
        self.example = label("", wrap=True)
        self.example.setStyleSheet(f"background: {theme.BLUE_SOFT}; border-radius: 10px; padding: 10px 12px;")
        tariff.lay.addWidget(self.example)
        row.addWidget(lot)
        row.addWidget(tariff)
        self.lay.addLayout(row)

        det = Card("دوربین‌ها و تشخیص پلاک")
        g = QGridLayout()
        g.setVerticalSpacing(14)
        self.cooldown = spin(5, 3600, 5, "  ثانیه")
        self.sens = QSlider(Qt.Orientation.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setSingleStep(10)
        self.sens.setMinimumWidth(260)
        g.addWidget(label("فاصله زمانی ثبت دوباره یک پلاک در همان دوربین", "muted"), 0, 0)
        g.addWidget(self.cooldown, 0, 1)
        g.addWidget(label("حساسیت تشخیص (بیشتر: پلاک‌های دورتر و کوچک‌تر)", "muted"), 1, 0)
        g.addWidget(self.sens, 1, 1)
        det.lay.addLayout(g)
        self.lay.addWidget(det)

        lic = Card("مجوز استفاده")
        self.lic_text = label("", wrap=True)
        lic.lay.addWidget(self.lic_text)
        h = QHBoxLayout()
        self.lic_btn = button("فعال‌سازی", "check", "primary")
        self.lic_btn.clicked.connect(lambda: self.ctl.show_activation())
        h.addWidget(self.lic_btn)
        h.addStretch(1)
        lic.lay.addLayout(h)
        self.lay.addWidget(lic)

        data = Card("داده‌ها")
        data.lay.addWidget(label(f"همه اطلاعات روی همین رایانه ذخیره می‌شود:  {ctl.db.path.parent}", "muted", wrap=True))
        h = QHBoxLayout()
        backup = button("پشتیبان‌گیری", "download")
        backup.clicked.connect(self.backup)
        h.addWidget(backup)
        h.addStretch(1)
        data.lay.addLayout(h)
        self.lay.addWidget(data)
        self.lay.addStretch(1)

    def refresh(self):
        s = self.ctl.settings
        self.name.setText(s["lot_name"])
        self.capacity.setValue(int(s["capacity"]))
        self.free.setValue(int(s["free_minutes"]))
        self.first.setValue(int(s["first_hour"]))
        self.extra.setValue(int(s["extra_hour"]))
        self.cap.setValue(int(s["daily_cap"]))
        self.cooldown.setValue(int(s["cooldown_s"]))
        self.sens.setValue(int(s["sensitivity"]))
        self._example()
        from .licensing import CONTACT, FREE_RECOGNITIONS

        lic = self.ctl.license
        if lic.activated:
            self.lic_text.setText(f"این نسخه روی این رایانه فعال شده است (شناسه {lic.mid}).")
        else:
            self.lic_text.setText(f"نسخه رایگان: {jalali.fa_digits(lic.remaining)} از {jalali.fa_digits(FREE_RECOGNITIONS)} تشخیص پلاک باقی مانده است. "
                                  f"برای فعال‌سازی نامحدود با {CONTACT} تماس بگیرید.")
        self.lic_btn.setVisible(not lic.activated)

    def _tariff(self):
        return billing.Tariff(self.free.value(), self.first.value(), self.extra.value(), self.cap.value())

    def _example(self):
        t = self._tariff()
        ex = [(25, "۲۵ دقیقه"), (150, "۲ ساعت و ۳۰ دقیقه"), (600, "۱۰ ساعت")]
        self.example.setText("نمونه:  " + "   ·   ".join(f"{k}: {jalali.money(billing.fee(m, t))}" for m, k in ex))

    def save(self):
        self.ctl.save_settings({"lot_name": self.name.text().strip() or "پارکینگ", "capacity": self.capacity.value(),
                                "free_minutes": self.free.value(), "first_hour": self.first.value(),
                                "extra_hour": self.extra.value(), "daily_cap": self.cap.value(),
                                "cooldown_s": self.cooldown.value(), "sensitivity": self.sens.value()})
        self.ctl.toast("تنظیمات ذخیره شد")

    def backup(self):
        f, _ = QFileDialog.getSaveFileName(self, "پشتیبان", f"parkyar_backup_{time.strftime('%Y%m%d')}.sqlite", "SQLite (*.sqlite)")
        if f:
            import sqlite3

            dst = sqlite3.connect(f)
            with dst:
                self.ctl.db.con.backup(dst)
            dst.close()
            self.ctl.toast("پشتیبان ذخیره شد")
