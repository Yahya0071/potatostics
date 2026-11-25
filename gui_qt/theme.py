"""
gui_qt/theme.py - Modern Dark & Light Stylesheets for Potatostics PySide6 IDE
Provides tailored styling for:
- Main Window, DockWidgets, and Splitters
- Virtualized Data Grid (QTableView)
- Monospaced Results Console (QPlainTextEdit)
- Variables & History Trees
- Command Bar & Spotlight Command Palette (Ctrl+K)
- Dialogs and Action Toolbars
"""

DARK_THEME_QSS = """
QWidget#toolbarSpace { background: transparent; }
QLabel#workspaceBrand {
    background: transparent;
    color: #78D7C1;
    font-size: 13pt;
    font-weight: 700;
    padding-right: 8px;
}
QWidget#commandBar {
    border-top: 1px solid #424854;
}
QMainWindow {
    background-color: #1E1E1E;
}

QWidget {
    background-color: #1E1E1E;
    color: #D4D4D4;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 10pt;
}

/* Menu Bar */
QMenuBar {
    background-color: #252526;
    color: #CCCCCC;
    border-bottom: 1px solid #3E3E42;
    padding: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #37373D;
    color: #FFFFFF;
}
QMenu {
    background-color: #252526;
    color: #CCCCCC;
    border: 1px solid #3E3E42;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #094771;
    color: #FFFFFF;
}
QMenu::separator {
    height: 1px;
    background: #3E3E42;
    margin: 4px 8px;
}

/* ToolBar */
QToolBar {
    background-color: #252526;
    border-bottom: 1px solid #3E3E42;
    spacing: 6px;
    padding: 4px 8px;
}
QToolButton {
    background-color: #2D2D30;
    color: #E0E0E0;
    border: 1px solid #3E3E42;
    border-radius: 5px;
    padding: 5px 10px;
    font-weight: 500;
}
QToolButton:hover {
    background-color: #3E3E42;
    border-color: #007ACC;
    color: #FFFFFF;
}
QToolButton:pressed {
    background-color: #007ACC;
    color: #FFFFFF;
}

/* Dock Widgets */
QDockWidget {
    color: #D4D4D4;
    font-weight: 600;
}
QDockWidget::title {
    background-color: #252526;
    padding: 6px 10px;
    border-bottom: 1px solid #3E3E42;
    text-align: left;
}
QDockWidget::close-button, QDockWidget::float-button {
    background-color: transparent;
    border-radius: 3px;
    padding: 2px;
}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #3E3E42;
}

/* Tab Widget & Bar */
QTabWidget::pane {
    border: 1px solid #3E3E42;
    background-color: #1E1E1E;
}
QTabBar::tab {
    background-color: #2D2D30;
    color: #969696;
    border: 1px solid #3E3E42;
    border-bottom: none;
    padding: 6px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #1E1E1E;
    color: #4EC9B0;
    font-weight: bold;
    border-bottom: 2px solid #007ACC;
}
QTabBar::tab:hover:!selected {
    background-color: #37373D;
    color: #D4D4D4;
}

/* Virtualized Table View (Data Editor) */
QTableView {
    background-color: #1E1E1E;
    alternate-background-color: #252526;
    color: #D4D4D4;
    gridline-color: #2E2E32;
    border: none;
    selection-background-color: #094771;
    selection-color: #FFFFFF;
    font-family: 'Consolas', monospace;
    font-size: 10pt;
}
QHeaderView::section {
    background-color: #252526;
    color: #9CDCFE;
    font-weight: bold;
    border: 1px solid #3E3E42;
    padding: 4px 6px;
}
QHeaderView::section:hover {
    background-color: #37373D;
}

/* Console (Results Window) */
QPlainTextEdit, QTextEdit {
    background-color: #1E1E1E;
    color: #D4D4D4;
    border: none;
    font-family: 'Consolas', 'Cascadia Code', monospace;
    font-size: 10pt;
    selection-background-color: #264F78;
}

/* Trees & Lists (Review & Variables) */
QTreeWidget, QListWidget {
    background-color: #252526;
    color: #CCCCCC;
    border: 1px solid #3E3E42;
    border-radius: 4px;
}
QTreeWidget::item, QListWidget::item {
    padding: 4px 6px;
    border-radius: 3px;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #2A2D2E;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #094771;
    color: #FFFFFF;
}

/* Command Bar Input */
QLineEdit {
    background-color: #252526;
    color: #FFFFFF;
    border: 1px solid #3E3E42;
    border-radius: 5px;
    padding: 6px 10px;
    font-family: 'Consolas', monospace;
    font-size: 10.5pt;
}
QLineEdit:focus {
    border: 1px solid #007ACC;
    background-color: #1F1F1F;
}

/* Buttons */
QPushButton {
    background-color: #0E639C;
    color: #FFFFFF;
    border: 1px solid #1177BB;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1177BB;
}
QPushButton:pressed {
    background-color: #094771;
}
QPushButton:disabled {
    background-color: #2D2D30;
    color: #656565;
    border-color: #3E3E42;
}

/* Splitters & Scrollbars */
QSplitter::handle {
    background-color: #2B2D30;
}
QSplitter::handle:hover {
    background-color: #007ACC;
}
QScrollBar:vertical {
    border: none;
    background-color: #1E1E1E;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #424242;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background-color: #4F4F4F;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal { background: #252526; height: 12px; }
QScrollBar::handle:horizontal { background: #535359; min-width: 22px; border-radius: 5px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""

LIGHT_THEME_QSS = """
QWidget#toolbarSpace { background: transparent; }
QLabel#workspaceBrand {
    background: transparent;
    color: #125E50;
    font-size: 13pt;
    font-weight: 700;
    padding-right: 8px;
}
QWidget#commandBar {
    border-top: 1px solid #CED4DA;
}
QMainWindow {
    background-color: #F3F3F3;
}

QWidget {
    background-color: #F3F3F3;
    color: #1E1E1E;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 10pt;
}

/* Menu Bar */
QMenuBar {
    background-color: #FFFFFF;
    color: #333333;
    border-bottom: 1px solid #D0D0D0;
    padding: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #E5E5E5;
    color: #000000;
}
QMenu {
    background-color: #FFFFFF;
    color: #333333;
    border: 1px solid #D0D0D0;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #0066B8;
    color: #FFFFFF;
}
QMenu::separator {
    height: 1px;
    background: #E5E5E5;
    margin: 4px 8px;
}

/* ToolBar */
QToolBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #D0D0D0;
    spacing: 6px;
    padding: 4px 8px;
}
QToolButton {
    background-color: #F8F9FA;
    color: #212529;
    border: 1px solid #CED4DA;
    border-radius: 5px;
    padding: 5px 10px;
    font-weight: 500;
}
QToolButton:hover {
    background-color: #E9ECEF;
    border-color: #0066B8;
    color: #000000;
}
QToolButton:pressed {
    background-color: #0066B8;
    color: #FFFFFF;
}

/* Dock Widgets */
QDockWidget {
    color: #1E1E1E;
    font-weight: 600;
}
QDockWidget::title {
    background-color: #EAEAEA;
    padding: 6px 10px;
    border-bottom: 1px solid #D0D0D0;
    text-align: left;
}
QDockWidget::close-button, QDockWidget::float-button {
    background-color: transparent;
    border-radius: 3px;
    padding: 2px;
}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #D0D0D0;
}

/* Tab Widget & Bar */
QTabWidget::pane {
    border: 1px solid #D0D0D0;
    background-color: #FFFFFF;
}
QTabBar::tab {
    background-color: #E4E4E4;
    color: #555555;
    border: 1px solid #D0D0D0;
    border-bottom: none;
    padding: 6px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #0066B8;
    font-weight: bold;
    border-bottom: 2px solid #0066B8;
}
QTabBar::tab:hover:!selected {
    background-color: #EDEDED;
    color: #000000;
}

/* Virtualized Table View (Data Editor) */
QTableView {
    background-color: #FFFFFF;
    alternate-background-color: #F9FAFB;
    color: #212529;
    gridline-color: #E5E7EB;
    border: none;
    selection-background-color: #CCE8FF;
    selection-color: #000000;
    font-family: 'Consolas', monospace;
    font-size: 10pt;
}
QHeaderView::section {
    background-color: #F3F4F6;
    color: #1F2937;
    font-weight: bold;
    border: 1px solid #E5E7EB;
    padding: 4px 6px;
}
QHeaderView::section:hover {
    background-color: #E5E7EB;
}

/* Console (Results Window) */
QPlainTextEdit, QTextEdit {
    background-color: #FFFFFF;
    color: #1E1E1E;
    border: none;
    font-family: 'Consolas', 'Cascadia Code', monospace;
    font-size: 10pt;
    selection-background-color: #ADD6FF;
}

/* Trees & Lists */
QTreeWidget, QListWidget {
    background-color: #FFFFFF;
    color: #212529;
    border: 1px solid #D0D0D0;
    border-radius: 4px;
}
QTreeWidget::item, QListWidget::item {
    padding: 4px 6px;
    border-radius: 3px;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #F0F4F8;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #CCE8FF;
    color: #000000;
}

/* Command Bar Input */
QLineEdit {
    background-color: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #CED4DA;
    border-radius: 5px;
    padding: 6px 10px;
    font-family: 'Consolas', monospace;
    font-size: 10.5pt;
}
QLineEdit:focus {
    border: 1px solid #0066B8;
    background-color: #FFFFFF;
}

/* Buttons */
QPushButton {
    background-color: #0066B8;
    color: #FFFFFF;
    border: 1px solid #005A9E;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #0078D4;
}
QPushButton:pressed {
    background-color: #005A9E;
}
QPushButton:disabled {
    background-color: #E0E0E0;
    color: #888888;
    border-color: #D0D0D0;
}

/* Splitters & Scrollbars */
QSplitter::handle {
    background-color: #D0D0D0;
}
QSplitter::handle:hover {
    background-color: #0066B8;
}
QScrollBar:vertical {
    border: none;
    background-color: #F3F3F3;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #C1C1C1;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background-color: #A8A8A8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal { background: #F3F3F3; height: 12px; }
QScrollBar::handle:horizontal { background: #A8A8A8; min-width: 22px; border-radius: 5px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""
