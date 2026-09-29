"""Persian UI pieces for the web app: texts, CSS (RTL, Vazirmatn) and Iranian-plate HTML cards."""
import html

from platereader.plates import plate_parts, to_persian_digits

TITLE = "سامانه هوشمند پلاک‌خوان"
SUBTITLE = "تشخیص و خواندن خودکار پلاک خودروهای ایرانی با هوش مصنوعی (YOLO26 + شبکه عصبی کانولوشنی)"

T = {
    "tab_image": "📷 خواندن از تصویر",
    "tab_batch": "🗂️ پردازش گروهی",
    "tab_history": "🕘 تاریخچه",
    "tab_about": "ℹ️ درباره سامانه",
    "input": "تصویر خودرو (بارگذاری، دوربین یا چسباندن)",
    "read": "خواندن پلاک",
    "clear": "پاک کردن",
    "annotated": "نتیجه روی تصویر",
    "settings": "⚙️ تنظیمات",
    "conf": "حداقل اطمینان تشخیص پلاک",
    "conf_info": "مقدار بیشتر = پلاک‌های اشتباه کمتر، مقدار کمتر = پلاک‌های کوچک و دور بیشتر",
    "min_text": "حداقل اطمینان خوانش (کمتر از این با ⚠️ علامت می‌خورد)",
    "files": "چند تصویر را انتخاب کنید",
    "run_batch": "پردازش همه تصاویر",
    "download_csv": "دانلود نتایج (CSV)",
    "download_json": "دانلود نتایج (JSON)",
    "gallery": "تصاویر پردازش‌شده",
    "history_clear": "پاک کردن تاریخچه",
    "no_plate": "پلاکی در تصویر پیدا نشد. تصویر واضح‌تر یا نزدیک‌تری امتحان کنید، یا آستانه تشخیص را کمتر کنید.",
    "no_image": "ابتدا یک تصویر انتخاب کنید.",
}
TABLE_HEADERS = ["ردیف", "پلاک (حروف لاتین)", "اطمینان تشخیص", "اطمینان خوانش", "وضعیت"]
BATCH_HEADERS = ["فایل", "پلاک (حروف لاتین)", "اطمینان تشخیص", "اطمینان خوانش", "وضعیت"]

CSS = """
@font-face { font-family: 'VazirmatnLocal'; src: url('{font_url}') format('truetype'); font-weight: 400 900; }
.gradio-container, .gradio-container * { font-family: 'Vazirmatn', 'VazirmatnLocal', Tahoma, sans-serif !important; }
.gradio-container { direction: rtl; max-width: 1400px !important; margin: auto; }
.gradio-container input[type=range] { direction: ltr; }
#app-header { text-align: center; padding: 18px 12px 6px; }
#app-header h1 { font-size: 2.1em; font-weight: 800; margin: 0; color: #1f3b73; }
#app-header p { font-size: 1.05em; opacity: .8; margin: 6px 0 0; }
.plates-wrap { direction: rtl; display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; padding: 8px 0; }
.plate-card { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.ir-plate { direction: ltr; display: flex; align-items: stretch; height: 64px; border: 3px solid #111; border-radius: 8px;
            background: #fff; color: #111; box-shadow: 0 2px 8px rgba(0,0,0,.18); overflow: hidden; }
.ir-plate .flag { width: 34px; background: #1d4ea3; color: #fff; display: flex; flex-direction: column;
                  justify-content: flex-end; align-items: center; font-size: 8px; line-height: 1.1; padding-bottom: 5px;
                  font-family: Arial, sans-serif !important; }
.ir-plate .flag .stripes { width: 20px; height: 13px; margin-bottom: 7px;
                           background: linear-gradient(#239f40 0 33%, #fff 33% 66%, #da0000 66%); }
.ir-plate .main { display: flex; align-items: center; gap: 10px; padding: 0 14px; font-size: 38px; font-weight: 800; }
.ir-plate .main .letter { font-size: 34px; }
.ir-plate .region { border-left: 3px solid #111; display: flex; flex-direction: column; align-items: center;
                    justify-content: center; padding: 0 10px; font-weight: 800; }
.ir-plate .region small { font-size: 11px; font-weight: 600; }
.ir-plate .region span { font-size: 30px; line-height: 1; }
.plate-meta { direction: rtl; unicode-bidi: plaintext; font-size: 14px; opacity: .85; text-align: center; }
.plate-meta.warn { color: #c62828; opacity: 1; font-weight: 700; }
.msg { direction: rtl; text-align: center; padding: 18px; font-size: 1.05em; opacity: .85; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 10px 0; }
.stat { border-radius: 12px; padding: 14px; background: rgba(31,59,115,.07); text-align: center; }
.stat b { display: block; font-size: 1.7em; color: #1f3b73; }
"""


def pct(x):
    return to_persian_digits(f"{x * 100:.0f}") + "٪"


def plate_html(text):
    parts = plate_parts(text)
    if parts is None:
        return f"<div class='ir-plate'><div class='main'>{html.escape(to_persian_digits(text))}</div></div>"
    a, letter, b, region = parts
    return ("<div class='ir-plate'><div class='flag'><div class='stripes'></div>I.R.<br>IRAN</div>"
            f"<div class='main'><span>{a}</span><span class='letter'>{html.escape(letter)}</span><span>{b}</span></div>"
            f"<div class='region'><small>ایران</small><span>{region}</span></div></div>")


def cards_html(results, min_text):
    if not results:
        return f"<div class='msg'>{T['no_plate']}</div>"
    items = []
    for i, r in enumerate(results, 1):
        warn = r.text_conf < min_text
        meta = (f"پلاک {to_persian_digits(i)} | اطمینان تشخیص {pct(r.det_conf)} | اطمینان خوانش {pct(r.text_conf)}"
                + (" | ⚠️ خوانش نامطمئن" if warn else ""))
        items.append(f"<div class='plate-card'>{plate_html(r.text)}<div class='plate-meta{' warn' if warn else ''}'>{meta}</div></div>")
    return "<div class='plates-wrap'>" + "".join(items) + "</div>"


def message_html(text):
    return f"<div class='msg'>{html.escape(text)}</div>"
