from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox

class CustomDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入信息")
        self.layout = QVBoxLayout()

        # 公司名输入
        self.label_company_name = QLabel("请输入公司名：")
        self.layout.addWidget(self.label_company_name)
        self.input_company_name = QLineEdit()
        self.layout.addWidget(self.input_company_name)

        # 序列号输入
        self.label_serial_number = QLabel("请输入序列号：")
        self.layout.addWidget(self.label_serial_number)
        self.input_serial_number = QLineEdit()
        self.layout.addWidget(self.input_serial_number)

        # 机器类型选择
        self.label_table_type = QLabel("请选择要查询的机器类型:")
        self.layout.addWidget(self.label_table_type)
        self.combo_table_type = QComboBox()
        self.combo_table_type.addItems(("圆领", "罗纹下摆"))
        self.combo_table_type.setEditable(False)
        self.layout.addWidget(self.combo_table_type)

        # 确认按钮
        self.button_confirm = QPushButton("确定")
        self.button_confirm.clicked.connect(self.accept)
        self.layout.addWidget(self.button_confirm)

        # 取消按钮
        self.button_cancel = QPushButton("取消")
        self.button_cancel.clicked.connect(self.reject)
        self.layout.addWidget(self.button_cancel)

        self.setLayout(self.layout)

    def get_values(self):
        return self.input_company_name.text(), self.input_serial_number.text(), self.combo_table_type.currentText()

class YourMainWindowClass:
    def get_company_name_and_other_info(self):
        dialog = CustomDialog(self)
        if dialog.exec():
            company_name, serial_number, table_type = dialog.get_values()
            print('company_name', company_name)
            print('serial_number', serial_number)
            print('table_type', table_type)
            self.Func.company_name_received_signal.emit(company_name, serial_number, table_type)


class Func:
    def __init__(self):
        from PyQt6.QtCore import pyqtSignal
        self.company_name_received_signal = pyqtSignal(str, str, str)

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    func = Func()
    main_window = YourMainWindowClass()
    main_window.Func = func
    main_window.get_company_name_and_other_info()

    sys.exit(app.exec())
