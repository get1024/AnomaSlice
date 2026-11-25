"""AnomaSlice desktop entry point."""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app_window import MainWindow
from app_version import APP_VERSION


def main():
    app = QApplication(sys.argv)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        def check_runtime():
            try:
                from runtime_check import verify_runtime
                verify_runtime(window)
            except Exception:
                if sys.stderr is not None:
                    import traceback
                    traceback.print_exc()
                app.exit(1)
            else:
                app.quit()
        QTimer.singleShot(0, check_runtime)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
