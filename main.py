"""AnomaSlice desktop entry point. Runtime dependency: PySide6 Essentials only."""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        def check_runtime():
            try:
                from runtime_check import verify_runtime
                verify_runtime()
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
