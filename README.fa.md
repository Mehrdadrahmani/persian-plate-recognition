<p align="center"><img src="docs/screenshots/banner.png" width="100%" alt="پلاک‌خوان"></p>

<div dir="rtl">

<h1 align="center">سامانه هوشمند پلاک‌خوان</h1>

<p align="center">
تشخیص و خواندن خودکار پلاک خودروهای ایرانی، از آموزش و ارزیابی مدل‌ها تا محصول نهایی:
اپلیکیشن اندروید آفلاین، برنامه وب فارسی و نرم‌افزار مدیریت پارکینگ برای ویندوز.
</p>

<p align="center">
  <a href="https://github.com/Mehrdadrahmani/persian-plate-recognition/releases/download/v2.2.0/PlateReader-2.2.0-android.apk"><img src="https://img.shields.io/badge/Download-Android%20app-3DDC84?style=for-the-badge&logo=android&logoColor=white" alt="دانلود اپلیکیشن اندروید"></a>
  &nbsp;
  <a href="https://github.com/Mehrdadrahmani/persian-plate-recognition/releases/download/v2.2.0/ParkYar-1.1.0-windows-x64-setup.exe"><img src="https://img.shields.io/badge/Download-ParkYar%20for%20Windows-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="دانلود پارک‌یار برای ویندوز"></a>
</p>

<p align="center"><a href="README.md">English</a> · <a href="docs/TECHNICAL_REPORT.md">گزارش فنی</a> · <a href="parkyar/README.fa.md">پارک‌یار</a></p>

## معرفی

این سامانه در دو مرحله کار می‌کند:

1. **پیدا کردن پلاک:** مدل YOLO26n همه پلاک‌های داخل تصویر را پیدا می‌کند.
2. **خواندن پلاک:** یک شبکه عصبی کانولوشنی که از صفر آموزش دیده است، ۸ نویسه پلاک را می‌خواند: دو رقم، یک حرف، سه رقم و کد دو رقمی استان.

حجم هر دو مدل پس از بهینه‌سازی INT8 روی هم ۱۱ مگابایت است. این مدل‌ها بدون نیاز به اینترنت روی گوشی و رایانه اجرا می‌شوند.

| محصول | سکو | توضیح |
|---|---|---|
| **پلاک‌خوان** | اندروید ۸ و بالاتر | اسکن با دوربین و گالری، اسکن زنده، تاریخچه با تاریخ شمسی، خروجی CSV و حالت تاریک؛ کاملاً آفلاین |
| **پارک‌یار** | ویندوز ۱۰ و ۱۱ | مدیریت پارکینگ: ثبت خودکار ورود و خروج، محاسبه هزینه، چاپ رسید، داشبورد، گزارش‌گیری و مدیریت مشترکین |
| **برنامه وب** | مرورگر (پایتون) | رابط فارسی راست‌به‌چپ با بارگذاری عکس، وب‌کم و پردازش گروهی |

</div>

<p align="center"><img src="docs/screenshots/showcase.png" width="100%" alt="صفحه‌های اپلیکیشن"></p>

<div dir="rtl">

## نتایج

همه اعداد روی داده آزمون به دست آمده‌اند. این داده هرگز در آموزش یا انتخاب مدل استفاده نشده است.

| بخش | نتیجه |
|---|---|
| تشخیص پلاک (YOLO26n) | دقت **mAP50 برابر ۰٫۹۸۸** |
| خواندن پلاک (شبکه کانولوشنی) | **۹۴٫۲٪** پلاک‌ها کاملاً درست و **۹۸٫۹٪** نویسه‌ها درست |
| سامانه کامل | **۹۶۹ از ۱٬۰۹۱ پلاک** کاملاً درست خوانده شد (۸۸٫۸٪)؛ حدود ۶۶ میلی‌ثانیه برای هر تصویر |
| نسخه موبایل (INT8) | ۹۵۹ از ۱٬۰۹۱ پلاک؛ **۲٫۶ برابر سریع‌تر** و حجم ۱۱ مگابایت به جای ۴۰ مگابایت |

روش کار به‌طور کامل در [گزارش فنی](docs/TECHNICAL_REPORT.md) آمده است.

## دریافت

* **اندروید:** [PlateReader-2.2.0-android.apk](https://github.com/Mehrdadrahmani/persian-plate-recognition/releases/download/v2.2.0/PlateReader-2.2.0-android.apk). فایل را روی گوشی باز کنید و در صورت درخواست، نصب از این منبع را مجاز کنید.
* **ویندوز:** [ParkYar-1.1.0-windows-x64-setup.exe](https://github.com/Mehrdadrahmani/persian-plate-recognition/releases/download/v2.2.0/ParkYar-1.1.0-windows-x64-setup.exe). اگر پیام SmartScreen نمایش داده شد، «More info» و سپس «Run anyway» را بزنید.

## مجوز استفاده

* **پلاک‌خوان (اندروید)** کاملاً رایگان است.
* **پارک‌یار (ویندوز)** شامل ۱۰ تشخیص رایگان پلاک برای آزمایش است. برای استفاده نامحدود، شناسه رایانه را که در پنجره فعال‌سازی نرم‌افزار نمایش داده می‌شود به [mehrdad.rahmani100@gmail.com](mailto:mehrdad.rahmani100@gmail.com) بفرستید تا کد فعال‌سازی برای همان رایانه ارسال شود.
* برای مجوز تجاری، راه‌اندازی اختصاصی یا پشتیبانی با توسعه‌دهنده تماس بگیرید.

## اجرا از کد منبع

</div>

```bash
git clone https://github.com/Mehrdadrahmani/persian-plate-recognition.git
cd persian-plate-recognition
pip install -r requirements.txt
python webapp/app.py                  # برنامه وب فارسی: http://127.0.0.1:7860
python scripts/predict.py car.jpg  # خواندن پلاک از خط فرمان
```

<div dir="rtl">

* مدل‌های آموزش‌دیده در پوشه `models/` قرار دارند و به آموزش دوباره نیازی نیست.
* مراحل بازتولید نتایج، از دریافت داده تا بهینه‌سازی برای موبایل، در [README انگلیسی](README.md) آمده است.

## پارک‌یار

[پارک‌یار](parkyar/README.fa.md) این مدل‌ها را به یک محصول کامل برای پارکینگ تبدیل می‌کند:

* دوربین‌های ورودی و خروجی پلاک خودروها را می‌خوانند.
* مدت توقف و هزینه هر خودرو طبق تعرفه خودکار محاسبه می‌شود.
* متصدی پرداخت را تأیید می‌کند و رسید چاپ می‌کند.

</div>

<p align="center"><img src="parkyar/docs/screenshots/hero.png" width="100%" alt="پارک‌یار"></p>

<div dir="rtl">

## مجوز

این پروژه با مجوز [AGPL-3.0](LICENSE) منتشر شده است. داده‌های آموزشی متعلق به دوره هستند و در این مخزن منتشر نمی‌شوند.

<p align="center"><sub>توسعه‌دهنده: <a href="https://github.com/Mehrdadrahmani">مهرداد رحمانی</a> · رهنما کالج، ۱۴۰۵</sub></p>

</div>
