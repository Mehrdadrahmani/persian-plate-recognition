"""Dialogs: checkout (fee + payment + receipt), plate entry / correction, plate not found, subscriber editor."""
import math
import time

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QFont, QTextDocument
from PySide6.QtWidgets import (QDateEdit, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QVBoxLayout, QWidget)

from . import billing, jalali, plates, theme
from .widgets import Card, PlateInput, PlateWidget, button, crop_pixmap, hline, label


class _Dialog(QDialog):
    def __init__(self, title, parent=None, width=560):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setMinimumWidth(width)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(26, 22, 26, 22)
        self.lay.setSpacing(14)
        self.lay.addWidget(label(title, "pageTitle"))


def fee_note(minutes, t: billing.Tariff, subscriber):
    if subscriber:
        return "مشترک — بدون هزینه"
    if minutes <= t.free_minutes:
        return f"کمتر از {jalali.fa_digits(t.free_minutes)} دقیقه — رایگان"
    hours = max(1, math.ceil(minutes / 60))
    if hours > 24 or (t.daily_cap and billing.fee(minutes, t) >= t.daily_cap and hours > 1):
        return "محاسبه با سقف روزانه"
    if hours == 1:
        return f"ساعت اول: {jalali.money(t.first_hour)}"
    return f"ساعت اول {jalali.money(t.first_hour, False)} + {jalali.fa_digits(hours - 1)} ساعت × {jalali.money(t.extra_hour, False)} تومان"


class CheckoutDialog(_Dialog):
    """Shows the stay and the fee; accept() = paid and the car leaves."""

    def __init__(self, session, tariff, lot_name, exit_crop=None, parent=None, letters=()):
        super().__init__("تسویه و خروج", parent, 720)
        self.session, self.letters = session, letters
        self.exit_time = time.time()
        self.minutes = (self.exit_time - session.entry_time) / 60
        self.fee = 0 if session.subscriber else billing.fee(self.minutes, tariff)
        self.lot_name, self.plate = lot_name, session.plate
        self.corrected = None

        card = Card(padding=22)
        top = QHBoxLayout()
        self.plate_w = PlateWidget(session.plate, 76)
        top.addWidget(self.plate_w)
        top.addStretch(1)
        for path, cap in ((session.entry_img, "ورود"), (exit_crop, "خروج")):
            pm = crop_pixmap(path, 44, 130)
            if pm:
                col = QVBoxLayout()
                col.setSpacing(2)
                im = QLabel()
                im.setPixmap(pm)
                im.setStyleSheet("border-radius: 8px;")
                col.addWidget(im)
                col.addWidget(label(cap, "muted"), 0, Qt.AlignmentFlag.AlignCenter)
                top.addLayout(col)
        card.lay.addLayout(top)
        card.lay.addWidget(hline())
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        rows = [("زمان ورود", f"{jalali.date(session.entry_time)}  ·  {jalali.time(session.entry_time)}"),
                ("زمان خروج", f"{jalali.date(self.exit_time)}  ·  {jalali.time(self.exit_time)}"),
                ("مدت توقف", jalali.duration(self.minutes))]
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, "muted"), i, 0)
            grid.addWidget(label(v, size=11.5, weight=QFont.Weight.DemiBold), i, 1)
        grid.setColumnStretch(1, 1)
        card.lay.addLayout(grid)
        self.lay.addWidget(card)

        pay = Card(padding=22)
        pay.setStyleSheet(f"#card {{ background: {theme.BLUE_SOFT}; border: none; }}")
        row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("مبلغ قابل پرداخت", "muted"))
        col.addWidget(label(fee_note(self.minutes, tariff, session.subscriber), "muted"))
        row.addLayout(col, 1)
        amount = label("رایگان" if self.fee == 0 else jalali.money(self.fee), "bigMoney")
        if self.fee == 0:
            amount.setStyleSheet(f"color: {theme.GREEN};")
        row.addWidget(amount)
        pay.lay.addLayout(row)
        self.lay.addWidget(pay)

        btns = QHBoxLayout()
        ok = button("پرداخت شد، خروج" if self.fee else "ثبت خروج", "check", "success")
        ok.clicked.connect(self.accept)
        pr = button("چاپ رسید", "print")
        pr.clicked.connect(self.print_receipt)
        fix = button("اصلاح پلاک", "edit", "ghost")
        fix.clicked.connect(self._fix)
        cancel = button("انصراف", kind="ghost")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(pr)
        btns.addStretch(1)
        btns.addWidget(fix)
        btns.addWidget(cancel)
        self.lay.addLayout(btns)
        ok.setDefault(True)

    def _fix(self):
        d = PlateDialog("اصلاح پلاک", self.letters, self.plate, parent=self)
        if d.exec():
            self.plate = self.corrected = d.plate
            self.plate_w.set_plate(self.plate)

    def receipt_html(self):
        s = self.session
        amount = "رایگان" if self.fee == 0 else jalali.money(self.fee)
        return f"""<div dir="rtl" style="font-family: Vazirmatn; font-size: 11pt;">
        <h2 style="text-align:center; margin:0;">{self.lot_name}</h2>
        <p style="text-align:center; color:#666; margin:2px;">رسید پارکینگ</p><hr/>
        <table width="100%" cellpadding="4">
        <tr><td>پلاک</td><td align="left"><b>{plates.display(self.plate)}</b></td></tr>
        <tr><td>ورود</td><td align="left">{jalali.datetime(s.entry_time)}</td></tr>
        <tr><td>خروج</td><td align="left">{jalali.datetime(self.exit_time)}</td></tr>
        <tr><td>مدت توقف</td><td align="left">{jalali.duration(self.minutes)}</td></tr>
        </table><hr/>
        <h2 style="text-align:center;">{amount}</h2>
        <p style="text-align:center; color:#666;">شماره رسید {jalali.fa_digits(s.id)} · از حضور شما سپاسگزاریم</p>
        <p style="text-align:center; color:#999; font-size:8pt;" dir="ltr">ParkYar · © Mehrdad Rahmani · mehrdad.rahmani100@gmail.com</p></div>"""

    def print_receipt(self):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        printer = QPrinter()
        dlg = QPrintDialog(printer, self)
        if dlg.exec():
            doc = QTextDocument()
            doc.setDefaultFont(theme.font(11))
            doc.setHtml(self.receipt_html())
            doc.print_(printer)


class PlateDialog(_Dialog):
    """Type or correct a plate number."""

    def __init__(self, title, letters, text="", crop=None, hint=None, parent=None):
        super().__init__(title, parent, 520)
        if hint:
            self.lay.addWidget(label(hint, "muted", wrap=True))
        pm = crop_pixmap(crop, 60)
        if pm:
            im = QLabel()
            im.setPixmap(pm)
            self.lay.addWidget(im, 0, Qt.AlignmentFlag.AlignHCenter)
        self.preview = PlateWidget(text, 70)
        self.lay.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.inp = PlateInput(letters)
        self.inp.set_text(text)
        self.lay.addWidget(self.inp, 0, Qt.AlignmentFlag.AlignHCenter)
        self.inp.changed.connect(self._changed)
        btns = QHBoxLayout()
        self.ok = button("تأیید", "check", "primary")
        self.ok.clicked.connect(self.accept)
        c = button("انصراف", kind="ghost")
        c.clicked.connect(self.reject)
        btns.addWidget(self.ok)
        btns.addStretch(1)
        btns.addWidget(c)
        self.lay.addLayout(btns)
        self._changed()

    def _changed(self):
        self.ok.setEnabled(self.inp.valid())
        self.preview.set_plate(self.inp.text() if self.inp.valid() else "")

    @property
    def plate(self):
        return self.inp.text()


class NotFoundDialog(_Dialog):
    """Exit of a plate with no open session: pick the right car from the ones inside, or correct the plate."""

    def __init__(self, plate, candidates, letters, crop=None, parent=None):
        super().__init__("پلاک در پارکینگ پیدا نشد", parent, 560)
        self.letters, self.choice, self.plate = letters, None, plate
        self.lay.addWidget(label("برای این پلاک ورودی ثبت نشده است. اگر خودرو یکی از موارد زیر است آن را انتخاب کنید "
                                 "یا شماره پلاک را اصلاح کنید.", "muted", wrap=True))
        pm = crop_pixmap(crop, 56)
        if pm:
            im = QLabel()
            im.setPixmap(pm)
            self.lay.addWidget(im, 0, Qt.AlignmentFlag.AlignHCenter)
        self.list = QListWidget()
        self.list.setMinimumHeight(min(320, 90 + 74 * len(candidates)))
        for s in candidates:
            it = QListWidgetItem()
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(6, 4, 6, 4)
            h.addWidget(PlateWidget(s.plate, 44))
            h.addStretch(1)
            h.addWidget(label(f"ورود {jalali.time(s.entry_time)} · {jalali.duration(s.minutes)} پیش", "muted"))
            it.setSizeHint(w.sizeHint())
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            self.list.addItem(it)
            self.list.setItemWidget(it, w)
        if candidates:
            self.lay.addWidget(label("خودروهای مشابه داخل پارکینگ", "cardTitle"))
            self.lay.addWidget(self.list)
        self.list.itemDoubleClicked.connect(lambda _: self._pick())
        btns = QHBoxLayout()
        pick = button("انتخاب خودرو", "check", "primary")
        pick.setEnabled(bool(candidates))
        pick.clicked.connect(self._pick)
        fix = button("اصلاح پلاک", "edit")
        fix.clicked.connect(self._fix)
        c = button("انصراف", kind="ghost")
        c.clicked.connect(self.reject)
        btns.addWidget(pick)
        btns.addWidget(fix)
        btns.addStretch(1)
        btns.addWidget(c)
        self.lay.addLayout(btns)
        if candidates:
            self.list.setCurrentRow(0)

    def _pick(self):
        it = self.list.currentItem()
        if it:
            self.choice = it.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def _fix(self):
        d = PlateDialog("اصلاح پلاک", self.letters, self.plate, parent=self)
        if d.exec():
            self.plate = d.plate
            self.choice = "retry"
            self.accept()


class SubscriberDialog(_Dialog):
    def __init__(self, letters, sub=None, parent=None):
        super().__init__("ویرایش مشترک" if sub else "مشترک جدید", parent, 520)
        grid = QGridLayout()
        grid.setVerticalSpacing(12)
        self.inp = PlateInput(letters)
        self.name, self.phone = QLineEdit(), QLineEdit()
        self.name.setPlaceholderText("نام و نام خانوادگی")
        self.phone.setPlaceholderText("۰۹۱۲…")
        self.until = QDateEdit()
        self.until.setCalendarPopup(True)
        self.until.setDisplayFormat("yyyy/MM/dd")
        self.until.setDate(QDate.currentDate().addMonths(1))
        self.until_fa = label("", "muted")
        self.until.dateChanged.connect(self._until_changed)
        for i, (k, w) in enumerate((("پلاک", self.inp), ("نام", self.name), ("تلفن", self.phone), ("اعتبار تا (میلادی)", self.until))):
            grid.addWidget(label(k, "muted"), i, 0)
            grid.addWidget(w, i, 1)
        grid.addWidget(self.until_fa, 4, 1)
        self.lay.addLayout(grid)
        if sub:
            self.inp.set_text(sub.plate)
            self.name.setText(sub.name)
            self.phone.setText(sub.phone)
            t = time.localtime(sub.valid_until)
            self.until.setDate(QDate(t.tm_year, t.tm_mon, t.tm_mday))
        self._until_changed()
        btns = QHBoxLayout()
        self.ok = button("ذخیره", "check", "primary")
        self.ok.clicked.connect(self.accept)
        c = button("انصراف", kind="ghost")
        c.clicked.connect(self.reject)
        btns.addWidget(self.ok)
        btns.addStretch(1)
        btns.addWidget(c)
        self.lay.addLayout(btns)
        self.inp.changed.connect(lambda: self.ok.setEnabled(self.inp.valid()))
        self.ok.setEnabled(self.inp.valid())

    def _until_changed(self):
        self.until_fa.setText("معادل " + jalali.date(self.valid_until))

    @property
    def valid_until(self):
        d = self.until.date()
        return time.mktime((d.year(), d.month(), d.day(), 23, 59, 59, 0, 0, -1))


class ActivationDialog(_Dialog):
    """Shows this computer's ID and accepts the activation key sent by the author."""

    def __init__(self, lic, trial_over=False, parent=None):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices, QGuiApplication
        from PySide6.QtWidgets import QPlainTextEdit

        from .licensing import AUTHOR, CONTACT, FREE_RECOGNITIONS

        super().__init__("فعال‌سازی پارک‌یار", parent, 600)
        self.lic = lic
        if trial_over:
            msg = f"{jalali.fa_digits(FREE_RECOGNITIONS)} تشخیص رایگان این نسخه استفاده شده است. برای ادامه، پارک‌یار را فعال کنید."
        else:
            msg = f"نسخه رایگان شامل {jalali.fa_digits(FREE_RECOGNITIONS)} تشخیص پلاک است؛ " \
                  f"{jalali.fa_digits(lic.remaining)} تشخیص باقی مانده است."
        self.lay.addWidget(label(msg, wrap=True))
        self.lay.addWidget(label(f"برای دریافت کد فعال‌سازی، شناسه این رایانه را به {CONTACT} بفرستید.", "muted", wrap=True))
        card = Card(padding=16)
        row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("شناسه این رایانه", "muted"))
        mid = label(lic.mid, size=16, weight=QFont.Weight.Bold)
        mid.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mid.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        col.addWidget(mid)
        row.addLayout(col, 1)
        copy = button("کپی", "edit")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(lic.mid))
        mail = button("ارسال ایمیل", "logout", "primary")
        subject = "ParkYar activation"
        body = f"Computer ID: {lic.mid}"
        mail.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"mailto:{CONTACT}?subject={subject}&body={body}")))
        row.addWidget(copy)
        row.addWidget(mail)
        card.lay.addLayout(row)
        self.lay.addWidget(card)
        self.lay.addWidget(label("کد فعال‌سازی", "cardTitle"))
        self.key = QPlainTextEdit()
        self.key.setPlaceholderText("XXXXX-XXXXX-…")
        self.key.setFixedHeight(90)
        self.key.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.lay.addWidget(self.key)
        self.err = label("", color=theme.RED)
        self.lay.addWidget(self.err)
        btns = QHBoxLayout()
        ok = button("فعال‌سازی", "check", "success")
        ok.clicked.connect(self._activate)
        c = button("بستن", kind="ghost")
        c.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addStretch(1)
        btns.addWidget(c)
        self.lay.addLayout(btns)
        self.lay.addWidget(label(f"© {AUTHOR} · {CONTACT}", "muted"), 0, Qt.AlignmentFlag.AlignHCenter)

    def _activate(self):
        if self.lic.activate(self.key.toPlainText()):
            self.accept()
        else:
            self.err.setText("کد فعال‌سازی برای این رایانه معتبر نیست.")
