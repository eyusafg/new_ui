import sys
from Signal_Ui import IAP
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    UI = IAP()
    UI.show()
    sys.exit(app.exec())
