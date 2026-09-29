# LPD + LPR artifact bundle (Phase 1 & 2 -> Phase 3)

Produced by `License_Plate_Detection_Recognition_cloud.ipynb` on **kaggle** (cuda), 2026-09-26 08:57.


## Files
| file | what | how it was produced |
|---|---|---|
| `lpd_best.pt` | YOLO26 plate detector (1 class `license_plate`) | Ultralytics fine-tune of `yolo26n.pt`, imgsz 960, best epoch by val fitness |
| `lpd_data.yaml` | dataset yaml used for training (`path` is platform specific) | notebook 1.4 |
| `lpd_train/val/test_split.txt` | LPD image file names per split; **test is held out** (never used for training/selection) | grouped-by-plate 70/15/15 split, seed 42 |
| `lpd_metrics.json` | val + test P / R / mAP50 / mAP50-95, recommended conf threshold | `model.val(split=...)` |
| `lpr_scratch_best.pt`, `lpr_pretrained_best.pt` | `{"state_dict", "arch", "n_letters", "letter_vocab", "input_size", "backbone"}`, loadable with `torch.load(..., weights_only=True)` | best val full-plate accuracy |
| `lpr_best.ts` | TorchScript of the best recognizer (**scratch**), traced on CPU and verified against eager | notebook export cell |
| `lpr_models.py` | model classes, `preprocess`, `crop_with_padding`, `normalize_label`, `decode`, `decode_batch` (torch/torchvision/numpy/PIL only) | same file the notebook trained with |
| `lpr_config.json` | vocabularies (Persian + Latin), input size, normalization, layout, crop padding, YOLO threshold | notebook export cell |
| `plate_reader.py`, `requirements.txt`, `APP_README.md`, optional `lpd_best.onnx` / `lpr_best.onnx` | **inference package for apps**: `PlateReader(...).read(image)` returns plate text + boxes; also a command-line tool | notebook section *Inference package for an app* |
| `lpr_comparison.csv`, `lpr_metrics.json` | test metrics, params, size, latency of both recognizers | notebook 2.10-2.11 |
| `outputs/*.csv`, `outputs/figures/*.png`, `outputs/run_log_*.txt` | per-epoch histories, per-sample test predictions, per-letter / per-slot / per-size accuracy, detector error analysis, all figures, text log | written throughout the notebook |
| `lpr_train/val/test_split.csv` | LPR splits (`filename,label`); LPR plates that appear in LPD-test images are all in `lpr_test` | grouped-by-plate, letter-stratified |

## Key metrics
* Detector (yolo26n.pt): test mAP50 **0.9880**, mAP50-95 **0.7850**, P 0.9569, R 0.9761; recommended conf **0.618**
* Recognizer test full-plate / per-char accuracy: scratch **0.9417** / 0.9890, pretrained **0.8637** / 0.9699 -> best = **scratch**

## Notes for Phase 3
* **LPD annotations include plate text: YES.** `plate_labels.csv` (in the LPD dataset, not in this bundle) has one row per image; multiple plates are space-separated. Their order relative to the boxes in the label .txt is not documented, so match per image (e.g. by string). Normalize ground truth with `lpr_models.normalize_label` before comparing.
* Recognizer input: RGB crop -> `preprocess()` (letterbox 64x256, ImageNet norm). For cv2 (BGR) arrays use `preprocess(crop, bgr=True)`.
* Crop the YOLO box enlarged by `crop_padding` = 0.05 of its size per side (`crop_with_padding`); training used +/-8% crop jitter.

## Load and read one plate
```python
import json, cv2, torch, lpr_models as M            # run inside the extracted bundle folder
from ultralytics import YOLO
cfg = json.load(open("lpr_config.json", encoding="utf-8"))
det, rec = YOLO("lpd_best.pt"), torch.jit.load("lpr_best.ts").eval()
img = cv2.imread("car.jpg")                                             # BGR
boxes = det.predict(img, conf=cfg["yolo_conf_threshold"], imgsz=cfg["yolo_imgsz"])[0].boxes
xyxy = boxes.xyxy[boxes.conf.argmax()].tolist()                         # most confident plate
crop = M.crop_with_padding(img, xyxy, cfg["crop_padding"])
with torch.no_grad(): out = rec(M.preprocess(crop, bgr=True)[None])
print(M.decode(out, cfg["letter_vocab"]))                               # e.g. '12ب34567'
```
