"""سامانه هوشمند پلاک‌خوان: Persian web app for Iranian licence plate recognition (Gradio).

    python webapp/app.py                      # http://127.0.0.1:7860
    python webapp/app.py --share              # temporary public link
    python webapp/app.py --host 0.0.0.0       # reachable from phones on the same Wi-Fi
    python webapp/app.py --backend torch      # PyTorch instead of ONNX Runtime
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

import ui  # noqa: E402
from platereader.pipeline import PlateRecognizer  # noqa: E402
from platereader.plates import to_persian_digits  # noqa: E402
from platereader.viz import FONT_DIR  # noqa: E402


def load_metrics():
    p = ROOT / "reports" / "end_to_end" / "end_to_end_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def about_html(reader):
    m = load_metrics()
    rows = [("مدل تشخیص پلاک", "YOLO26n (Ultralytics)، ورودی ۹۶۰ پیکسل"),
            ("مدل خواندن پلاک", "شبکه کانولوشنی باقیمانده (ScratchLPRNet) با ۸ خروجی: ۷ رقم + ۱ حرف"),
            ("موتور اجرا", "ONNX Runtime" if reader.backend == "onnx" else "PyTorch"),
            ("حروف پشتیبانی‌شده", " ".join(reader.letters))]
    html = "<div dir='rtl'><h3>درباره سامانه</h3><table style='width:100%'>" + "".join(
        f"<tr><td style='padding:6px 10px'><b>{k}</b></td><td style='padding:6px 10px'>{v}</td></tr>" for k, v in rows) + "</table>"
    if m:
        e = m["end_to_end"]["best_f1_threshold"]
        stats = [("پلاک‌های آزمون", to_persian_digits(e["gt_plates"])),
                 ("خوانده‌شده کاملاً درست", ui.pct(e["e2e_recall"])),
                 ("دقت تشخیص (mAP50)", ui.pct(m["detector_test_map"].get("mAP50", 0)) if m["detector_test_map"] else "-"),
                 ("دقت خوانش پلاک‌های تشخیص‌داده‌شده", ui.pct(e["recognition_acc_on_detected"])),
                 ("خطای نویسه (CER)", ui.pct(e["cer_on_detected"])),
                 ("زمان پردازش هر تصویر", to_persian_digits(f"{m['speed_ms_per_image']['total_ms']:.0f}") + " میلی‌ثانیه")]
        html += ("<h3>نتایج روی داده آزمون (دیده‌نشده)</h3><div class='stat-grid'>"
                 + "".join(f"<div class='stat'><b>{v}</b>{k}</div>" for k, v in stats) + "</div>")
    html += ("<p>پلاک به ترتیب نوشته‌شده روی آن نمایش داده می‌شود: دو رقم، حرف، سه رقم و کد شهر (ایران). "
             "«اطمینان خوانش» کمترین احتمال میان ۸ نویسه است؛ مقدار پایین یعنی باید پلاک را با چشم هم بررسی کرد.</p></div>")
    return html


def build(reader, default_conf):
    def read_image(img, conf, min_text, history):
        if img is None:
            return None, ui.message_html(ui.T["no_image"]), [], None, history, history_rows(history)
        t0 = time.perf_counter()
        results = reader.predict(img[:, :, ::-1].copy(), conf=conf)  # gradio gives RGB, pipeline expects BGR
        ms = (time.perf_counter() - t0) * 1000
        annotated = reader.draw(img[:, :, ::-1].copy(), results)
        table = [[to_persian_digits(i), r.latin, ui.pct(r.det_conf), ui.pct(r.text_conf),
                  "⚠️ نامطمئن" if r.text_conf < min_text else "✅"] for i, r in enumerate(results, 1)]
        stamp = time.strftime("%H:%M:%S")
        history = history + [{"time": stamp, "text": r.text, "persian": r.persian, "latin": r.latin,
                              "det_conf": r.det_conf, "text_conf": r.text_conf} for r in results]
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"processing_ms": round(ms, 1), "plates": [r.to_dict() for r in results]}, f, ensure_ascii=False, indent=2)
        f.close()
        return annotated, ui.cards_html(results, min_text), table, f.name, history, history_rows(history)

    def read_batch(files, conf, min_text):
        if not files:
            return [], [], None
        rows, csv_rows, gallery = [], [], []
        for fp in files:
            path = fp if isinstance(fp, str) else fp.name
            try:
                results = reader.predict(path, conf=conf)
            except Exception as e:  # unreadable file
                rows.append([Path(path).name, "—", "—", "—", f"خطا: {e}"])
                csv_rows.append([Path(path).name, "", "", "", "", f"error: {e}"])
                continue
            gallery.append((reader.draw(path, results), Path(path).name))
            if not results:
                rows.append([Path(path).name, "—", "—", "—", "پلاکی پیدا نشد"])
                csv_rows.append([Path(path).name, "", "", "", "", "no plate"])
            for r in results:
                rows.append([Path(path).name, r.latin, ui.pct(r.det_conf), ui.pct(r.text_conf),
                             "⚠️ نامطمئن" if r.text_conf < min_text else "✅"])
                csv_rows.append([Path(path).name, r.text, r.latin, r.det_conf, r.text_conf,
                                 "uncertain" if r.text_conf < min_text else "ok"])
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
        w = csv.writer(f)
        w.writerow(["file", "plate", "latin", "det_conf", "text_conf", "status"])
        w.writerows(csv_rows)
        f.close()
        return rows, gallery, f.name

    def history_rows(history):
        return [[h["time"], h["latin"], ui.pct(h["text_conf"])] for h in reversed(history)]

    def history_csv(history):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
        w = csv.writer(f)
        w.writerow(["time", "plate", "persian", "latin", "det_conf", "text_conf"])
        for h in history:
            w.writerow([h["time"], h["text"], h["persian"], h["latin"], h["det_conf"], h["text_conf"]])
        f.close()
        return f.name

    examples = []
    test_list = ROOT / "models" / "splits" / "lpd_test_split.txt"
    if test_list.exists() and (ROOT / "data" / "LPD" / "images").exists():
        examples = [str(ROOT / "data" / "LPD" / "images" / f) for f in test_list.read_text().split()[:6]]

    with gr.Blocks(title=ui.TITLE, fill_width=True) as demo:
        history = gr.State([])
        gr.HTML(f"<div id='app-header'><h1>🚗 {ui.TITLE}</h1><p>{ui.SUBTITLE}</p></div>")
        with gr.Accordion(ui.T["settings"], open=False):
            with gr.Row():
                conf = gr.Slider(0.05, 0.95, value=default_conf, step=0.01, label=ui.T["conf"], info=ui.T["conf_info"])
                min_text = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label=ui.T["min_text"])
        with gr.Tab(ui.T["tab_image"]):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label=ui.T["input"], height=420)
                    with gr.Row():
                        btn = gr.Button(ui.T["read"], variant="primary", size="lg")
                        gr.ClearButton([inp], value=ui.T["clear"])
                    if examples:
                        gr.Examples(examples, inputs=inp, label="نمونه تصاویر آزمون")
                with gr.Column(scale=1):
                    out_img = gr.Image(label=ui.T["annotated"], height=420, interactive=False)
            cards = gr.HTML()
            table = gr.Dataframe(headers=ui.TABLE_HEADERS, interactive=False, wrap=True, elem_classes=["plate-table"])
            json_file = gr.File(label=ui.T["download_json"])
        with gr.Tab(ui.T["tab_batch"]):
            files = gr.File(file_count="multiple", file_types=["image"], label=ui.T["files"])
            run_b = gr.Button(ui.T["run_batch"], variant="primary")
            b_table = gr.Dataframe(headers=ui.BATCH_HEADERS, interactive=False, wrap=True, elem_classes=["plate-table"])
            b_csv = gr.File(label=ui.T["download_csv"])
            b_gallery = gr.Gallery(label=ui.T["gallery"], columns=3, height=520, object_fit="contain")
        with gr.Tab(ui.T["tab_history"]):
            h_table = gr.Dataframe(headers=["زمان", "پلاک (حروف لاتین)", "اطمینان خوانش"], interactive=False, elem_classes=["plate-table"])
            with gr.Row():
                h_dl = gr.Button(ui.T["download_csv"])
                h_clear = gr.Button(ui.T["history_clear"])
            h_file = gr.File(label=ui.T["download_csv"])
        with gr.Tab(ui.T["tab_about"]):
            gr.HTML(about_html(reader))

        outs = [out_img, cards, table, json_file, history, h_table]
        btn.click(read_image, [inp, conf, min_text, history], outs)
        inp.change(read_image, [inp, conf, min_text, history], outs)  # upload, webcam, paste or example
        run_b.click(read_batch, [files, conf, min_text], [b_table, b_gallery, b_csv])
        h_dl.click(history_csv, history, h_file)
        h_clear.click(lambda: ([], []), None, [history, h_table])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    reader = PlateRecognizer(args.models_dir, backend=args.backend)
    m = load_metrics()
    default_conf = float(m["recommended_threshold"]) if m else reader.conf
    demo = build(reader, default_conf)
    font = FONT_DIR / "Vazirmatn-Regular.ttf"
    data = ROOT / "data" / "LPD" / "images"  # example images (may be a symlink to the dataset folder)
    allowed = [str(FONT_DIR), str(data), str(data.resolve())]
    css = ui.CSS.replace("{font_url}", f"/gradio_api/file={font}")
    head = ("<link rel='preconnect' href='https://fonts.googleapis.com'>"
            "<link href='https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap' rel='stylesheet'>")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, css=css, head=head,
                theme=gr.themes.Soft(primary_hue="blue"), allowed_paths=allowed)


if __name__ == "__main__":
    main()
