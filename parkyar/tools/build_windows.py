"""Build the Windows installer (ParkYar-Setup-<version>.exe) on any OS — Windows, macOS or Linux.

Uses pynsist: an embeddable Windows Python + Windows wheels from PyPI + an NSIS installer with Start-menu and
desktop shortcuts. Nothing is compiled, so it can be cross-built.

    pip install pynsist          # and NSIS: `brew install makensis` / `apt install nsis` / nsis.sourceforge.io
    python tools/build_windows.py
"""
import glob
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
WHEELS = BUILD / "wheels"
PY = "3.12.10"
REQS = ["PySide6-Essentials", "onnxruntime", "opencv-python-headless", "numpy", "pillow"]


def version():
    ns = {}
    exec((ROOT / "parkyar" / "__init__.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def icon():
    """Render the app icon to build/parkyar.ico (Qt draws it, Pillow writes the multi-size .ico)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PIL import Image
    from PySide6.QtWidgets import QApplication

    sys.path.insert(0, str(ROOT))
    from parkyar import theme

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("ParkYar")
    png = BUILD / "icon.png"
    theme.app_icon(256).save(str(png))
    ico = BUILD / "parkyar.ico"
    Image.open(png).save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def main():
    ver = version()
    shutil.rmtree(BUILD / "nsis", ignore_errors=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--platform", "win_amd64",
                           "--python-version", PY.rsplit(".", 1)[0], "--implementation", "cp", "-d", str(WHEELS), *REQS])
    for c in (ROOT / "parkyar").rglob("__pycache__"):
        shutil.rmtree(c)
    ico = icon()
    lic = ROOT.parent / "LICENSE"  # the repository licence
    files = f"files={lic}" if lic.exists() else ""
    cfg = BUILD / "installer.cfg"
    cfg.write_text(f"""[Application]
name=ParkYar
version={ver}
publisher=ParkYar
entry_point=parkyar.app:main
icon={ico}
console=false

[Python]
version={PY}
bitness=64

[Include]
local_wheels={WHEELS}/*.whl
packages=parkyar
{files}

[Build]
installer_name=ParkYar-Setup-{ver}.exe
directory={BUILD / 'nsis'}
""", encoding="utf-8")
    subprocess.check_call([sys.executable, "-m", "nsist", str(cfg), "--no-makensis"])
    pkgs = BUILD / "nsis" / "pkgs"
    # onnxruntime needs the MSVC C++ runtime; PySide6 ships it, so a PC without the VC++ redistributable still works
    for dll in glob.glob(str(pkgs / "PySide6" / "msvcp140*.dll")) + glob.glob(str(pkgs / "PySide6" / "vcruntime140*.dll")):
        shutil.copy(dll, pkgs / "onnxruntime" / "capi")
    # drop parts of Qt the app never uses (smaller installer)
    for pattern in ("PySide6/Qt6WebEngine*", "PySide6/qml", "PySide6/translations", "PySide6/Qt6Quick*", "PySide6/Qt6Qml*",
                    "PySide6/Qt6Pdf*", "PySide6/Qt6Designer*", "PySide6/*.exe", "PySide6/examples", "PySide6/include"):
        for p in glob.glob(str(pkgs / pattern)):
            shutil.rmtree(p) if Path(p).is_dir() else Path(p).unlink()
    subprocess.check_call(["makensis", "-V2", str(BUILD / "nsis" / "installer.nsi")])
    out = BUILD / "nsis" / f"ParkYar-Setup-{ver}.exe"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    shutil.copy(out, dist / out.name)
    print(f"\nInstaller: {dist / out.name}  ({(dist / out.name).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
