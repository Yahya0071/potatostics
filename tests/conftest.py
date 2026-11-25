"""
tests/conftest.py - Pytest Fixtures and Teardown for Clean Qt Session Termination
Prevents segmentation faults and access violations on Windows and CI runners.
"""

import gc
import pytest
from PySide6 import QtWidgets, QtCore

@pytest.fixture(scope="session", autouse=True)
def qapp_session():
    """Ensure a single offscreen QApplication instance for the entire test session."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["--platform", "offscreen"])
    yield app
    app.closeAllWindows()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()
    gc.collect()

@pytest.fixture(autouse=True)
def clean_qt_per_test():
    """Tear down any lingering widgets or event queues after each test."""
    yield
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.closeAllWindows()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()

def pytest_sessionfinish(session, exitstatus):
    """Flushes Qt event loop and closes all windows before Python process termination."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.closeAllWindows()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
        gc.collect()
