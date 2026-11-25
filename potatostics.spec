# -*- mode: python ; coding: utf-8 -*-
import os

gui_assets = [(os.path.join('gui', 'assets', f), os.path.join('gui', 'assets')) for f in os.listdir('gui/assets')] if os.path.exists('gui/assets') else []
gui_qt_assets = [(os.path.join('gui_qt', 'assets', f), os.path.join('gui_qt', 'assets')) for f in os.listdir('gui_qt/assets')] if os.path.exists('gui_qt/assets') else []
engine_assets = [(os.path.join('engine', 'assets', 'data', f), os.path.join('engine', 'assets', 'data')) for f in os.listdir('engine/assets/data')] if os.path.exists('engine/assets/data') else []
datas = [('sample_data.csv', '.'), ('LICENSE', '.')] + gui_assets + gui_qt_assets + engine_assets

block_cipher = None

is_core_edition = (os.environ.get('POTATOSTICS_EDITION', '').lower() == 'core')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'scipy.stats',
        'scipy.special',
        'pandas',
        'numpy',
        'matplotlib',
        'customtkinter',
        'PIL',
        'docx',
        'docx.oxml',
        'engine.dataset',
        'engine.formatter',
        'engine.estimation',
        'engine.statistics',
        'engine.diagnostics',
        'engine.timeseries',
        'engine.causal',
        'engine.copilot',
        'engine.report_generator',
        'engine.plugins_manager',
        'engine.interpreter',
        'engine.estimates',
        'gui.app',
        'gui.data_editor',
        'gui.script_editor',
        'gui.plot_window',
        'gui.dialogs',
        'gui.branding',
        'gui_qt.theme',
        'gui_qt.data_grid',
        'gui_qt.console',
        'gui_qt.command_bar',
        'gui_qt.history_pane',
        'gui_qt.variables_pane',
        'gui_qt.command_palette',
        'gui_qt.dialogs',
        'gui_qt.main_window',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'engine.parser',
        'engine.data_ops',
        'engine.macros',
        'engine.script_runner',
        'engine.advanced_econometrics',
        'engine.causal_modern',
        'engine.exporters',
        'engine.chart_studio',
        'engine.bigdata',
        'engine.copilot_egm',
        'engine.session_manager',
        'engine.citations',
        'engine.reference_log_parser',
        'engine.benchmarks',
    ] + ([] if is_core_edition else ['pyarrow', 'polars']),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unused Qt bindings (Potatostics strictly uses PySide6)
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtNetwork',
        'PyQt6.QtQml',
        'PyQt6.QtQuick',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        # Unused PySide6 submodules & multimedia/3D extensions
        'PySide6.QtNetwork',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtQuickWidgets',
        'PySide6.Qt3DCore',
        'PySide6.Qt3DRender',
        'PySide6.Qt3DInput',
        'PySide6.Qt3DLogic',
        'PySide6.Qt3DAnimation',
        'PySide6.Qt3DExtras',
        'PySide6.QtDesigner',
        'PySide6.QtBluetooth',
        'PySide6.QtSensors',
        'PySide6.QtPositioning',
        'PySide6.QtNfc',
        'PySide6.QtWebChannel',
        'PySide6.QtWebSockets',
        'PySide6.QtRemoteObjects',
        'PySide6.QtScxml',
        'PySide6.QtSerialPort',
        'PySide6.QtSpatialAudio',
        'PySide6.QtStateMachine',
        'PySide6.QtPdf',
        'PySide6.QtPdfWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.QtTest',
        'PySide6.QtXml',
        'PySide6.QtSql',
        'PySide6.QtHelp',
        # Unused database drivers & server backends
        'asyncpg',
        'psycopg2',
        'psycopg2_binary',
        'mysql',
        'cx_Oracle',
        'sqlalchemy',
        # Test frameworks & developer tools
        'pytest',
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'pandas.tests',
        # Unused heavy machine learning & GIS frameworks
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'keras',
        'pyogrio',
        'pyproj',
        'shapely',
        'Shapely',
        'altair',
        'bokeh',
        'seaborn',
        'pydantic',
        'pydantic_core',
        'notebook',
        'jupyter',
        'IPython',
        'nbformat',
        'jedi',
        'pygments',
        'cryptography'
    ] + (['polars', 'pyarrow'] if os.environ.get('POTATOSTICS_EDITION', '').lower() == 'core' else []),
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Prune unused Qt binaries and translation blobs
# Qt on Windows uses the ICU interface supplied by Windows. A private ICU
# found on PATH can have the same filename and different exported symbols.
private_icu_dirs = {
    os.path.dirname(source)
    for dest, source, _ in a.binaries
    if os.name == 'nt' and dest.lower() == 'icuuc.dll'
}
filtered_binaries = []
for b in a.binaries:
    name_lower = b[0].lower()
    if os.name == 'nt' and name_lower.startswith(('api-ms-win-', 'ext-ms-win-')):
        continue
    if os.path.dirname(b[1]) in private_icu_dirs and (
        name_lower == 'icuuc.dll' or (name_lower.startswith('icudt') and name_lower.endswith('.dll'))
    ):
        continue
    if any(k in name_lower for k in ('pyqt6', 'qt6qml', 'qt6quick', 'qt63d', 'qt6network', 'qt6designer', 'qt6multimedia', 'qt6pdf', 'qt6sensors', 'qt6bluetooth')):
        continue
    filtered_binaries.append(b)
a.binaries = filtered_binaries

filtered_datas = []
for d in a.datas:
    p_lower = d[0].lower()
    if 'translations' in p_lower and p_lower.endswith('.qm') and not any(k in p_lower for k in ('en', 'us')):
        continue
    filtered_datas.append(d)
a.datas = filtered_datas

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Potatostics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gui/assets/potatostics.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Potatostics',
)
