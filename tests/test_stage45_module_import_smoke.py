"""
Stage 45 — Comprehensive Module Import Smoke Test
Validates that every single Python module in engine/, gui_qt/, and gui/
can be imported without syntax errors, missing symbols, or unhandled exceptions.
"""

import os
import glob
import importlib
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["POTATOSTICS_TESTING"] = "1"


def get_all_package_modules(package_dir: str):
    modules = []
    for f in sorted(glob.glob(f"{package_dir}/*.py")):
        basename = os.path.basename(f)
        if basename.startswith(".") or basename == "__pycache__":
            continue
        mod_name = f"{package_dir}.{basename[:-3]}"
        modules.append(mod_name)
    return modules


ENGINE_MODULES = get_all_package_modules("engine")
GUI_QT_MODULES = get_all_package_modules("gui_qt")
GUI_MODULES = get_all_package_modules("gui")


@pytest.mark.parametrize("module_name", ENGINE_MODULES)
def test_import_engine_module(module_name):
    """Every engine module must import cleanly."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", GUI_QT_MODULES)
def test_import_gui_qt_module(module_name):
    """Every Qt GUI module must import cleanly."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", GUI_MODULES)
def test_import_gui_module(module_name):
    """Every GUI module must import cleanly, skipping gracefully if optional customtkinter is not installed."""
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        pytest.skip("Optional legacy GUI dependency 'customtkinter' is not installed in this environment")
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_import_main_entrypoint():
    """main.py must compile and import cleanly."""
    import main
    assert hasattr(main, "main")
