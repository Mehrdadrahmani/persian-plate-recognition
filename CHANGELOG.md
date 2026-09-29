# Changelog

All notable changes to the applications in this repository.

## [2.2.0] — Plate Reader 2.2.0 (Android) · ParkYar 1.1.0 (Windows)
### Added
- ParkYar: evaluation mode with 10 free plate recognitions, and per-computer activation keys for unlimited use.
### Changed
- ParkYar: plate labels on the camera view follow the printed order of the plate; checkout layout handles wide
  plate snapshots.
### Fixed
- ParkYar: clean shutdown on exit.

## [2.1.0] — Plate Reader 2.1.0 · ParkYar 1.0.0
### Added
- ParkYar 1.0.0: Windows parking management with entry and exit cameras, fee calculation, receipts, a dashboard,
  reports with CSV export and subscriber management.
### Changed
- Plate Reader: streamlined result, history and settings screens; CSV export limited to plate, Latin form, date and
  time.

## [2.0.0] — Plate Reader 2.0.0
### Added
- Live scanning with two-frame confirmation.
- History with Jalali dates, search, swipe to delete and CSV export; settings screen; dark mode.
### Changed
- Redesigned interface (Material 3, Vazirmatn typeface, intro and result screens).
- INT8 models selected on the validation split: 11 MB instead of 40 MB; about 0.1 s per photo in the Android
  emulator (previously 2.5 s).

## [1.0.0] — Initial release
- YOLO26n detector, residual CNN recognizer and end-to-end pipeline.
- Persian web app (Gradio) and offline Android app.
