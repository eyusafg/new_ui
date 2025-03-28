from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QLabel, QLineEdit, QPushButton, QHBoxLayout)
from PyQt6.QtCore import Qt


class DeviceInfoDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设备信息")
        self.setGeometry(100, 100, 800, 400)
        
        # 创建表格部件
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["ID", "公司", "机器码", "软件版本", "硬件版本", '机器序列号', 'HMI_code', "创建时间"])
        
        # 设置列宽策略
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 50)    # ID
        self.table.setColumnWidth(1, 100)   # 公司
        self.table.setColumnWidth(2, 100)   # 机器码
        self.table.setColumnWidth(3, 100)   # 软件版本
        self.table.setColumnWidth(4, 100)   # 硬件版本
        self.table.setColumnWidth(5, 100)   # 机器序列号
        self.table.setColumnWidth(6, 100)   # HMI_code
        self.table.setColumnWidth(7, 200)   # 创建时间
        
        # 填充原始数据
        self.original_data = data
        self.populate_table(data)
        
        # 创建搜索组件
        self.search_label = QLabel("搜索:")
        self.search_input = QLineEdit()
        self.search_button = QPushButton("查询")
        self.search_button.clicked.connect(self.search_table)
        
        # 设置布局
        layout = QVBoxLayout()
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.show()

    def populate_table(self, data):
        self.table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, item in enumerate(row_data):
                table_item = QTableWidgetItem(str(item))
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, col_idx, table_item)

    def search_table(self, text=None):
        search_text = self.search_input.text().strip().lower()
        if not search_text:
            self.populate_table(self.original_data)
            return
        
        # 过滤数据
        filtered_data = []
        for row_data in self.original_data:
            if any(search_text in str(item).lower() for item in row_data):
                filtered_data.append(row_data)
        
        # 填充过滤后的数据
        self.populate_table(filtered_data)

