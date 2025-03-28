from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox

class QueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择查询条件")
        self.resize(300, 150)
        # 创建布局
        self.layout = QVBoxLayout(self)

        # 创建机器类型选择框
        self.machine_type_label = QLabel("选择机器类型:")
        self.machine_type_combo = QComboBox()
        self.machine_type_combo.addItems(["圆领", "罗纹下摆"])
        self.machine_type_combo.setCurrentIndex(0)

        # 创建查询方式选择框
        self.query_type_label = QLabel("选择查询方式:")
        self.query_type_combo = QComboBox()
        self.query_type_combo.addItems(["全量查询", "条件查询"])
        self.query_type_combo.setCurrentIndex(0)

        # 创建查询条件输入框
        # self.query_condition_label = QLabel("请输入查询条件:")
        self.query_column_name_combo = QComboBox()
        self.query_column_name_combo.addItems(["uid", "company_name", "serial_number"])
        self.query_column_name_combo.setCurrentIndex(0)
        self.query_column_name_combo.setVisible(False)  # 初始状态下不显示

        self.query_condition_edit = QLineEdit()
        self.query_condition_edit.setPlaceholderText("请输入查询条件")
        self.query_condition_edit.setVisible(False)  # 初始状态下不显示

        # 根据查询类型设置查询条件输入框的可见性
        self.query_type_combo.currentIndexChanged.connect(lambda: self.query_condition_edit.setVisible(self.query_type_combo.currentIndex() == 1))
        self.query_type_combo.currentIndexChanged.connect(lambda: self.query_column_name_combo.setVisible(self.query_type_combo.currentIndex() == 1))
        
        # 添加按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)

        # 将控件添加到布局中
        self.layout.addWidget(self.machine_type_label)
        self.layout.addWidget(self.machine_type_combo)
        self.layout.addWidget(self.query_type_label)
        self.layout.addWidget(self.query_type_combo)
        # self.layout.addWidget(self.query_condition_label)
        self.layout.addWidget(self.query_column_name_combo)
        self.layout.addWidget(self.query_condition_edit)
        self.layout.addWidget(self.ok_button)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

    def get_values(self):
        machine_type = self.machine_type_combo.currentText()
        query_type = self.query_type_combo.currentText()
        query_column_name = self.query_column_name_combo.currentText()
        query_condition = self.query_condition_edit.text()
        # print("查询条件：")
        # print('query_column_name',query_column_name, 'query_condition', query_condition)
        return machine_type, query_type, query_column_name, query_condition

        # return self.machine_type_combo.currentText(), self.query_type_combo.currentText()
