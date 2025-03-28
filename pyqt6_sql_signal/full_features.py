
from PyQt6 import QtCore, QtGui, QtWidgets
import tkinter as tk
from tkinter import messagebox

def setup_tab4(ui):
    ui.tab_4 = QtWidgets.QWidget()
    ui.tab_4.setObjectName("tab_4")
    ui.tab_4_container = QtWidgets.QWidget(parent=ui.tab_4)
    ui.tab_4_container.setGeometry(QtCore.QRect(0, 0, 871, 541))
    ui.save_parameter = QtWidgets.QPushButton(parent=ui.tab_4_container)
    ui.save_parameter.setGeometry(QtCore.QRect(260, 200, 171, 61))
    font = QtGui.QFont()
    font.setPointSize(12)
    font.setBold(True)
    ui.save_parameter.setFont(font)
    ui.save_parameter.setStyleSheet("border-radius:10px;background-color:lightblue")
    ui.save_parameter.setObjectName("save_parameter")

    ui.UpdatawithAPP_3 = QtWidgets.QPushButton(parent=ui.tab_4_container)
    ui.UpdatawithAPP_3.setGeometry(QtCore.QRect(410, 10, 171, 61))
    font = QtGui.QFont()
    font.setPointSize(12)
    font.setBold(True)
    ui.UpdatawithAPP_3.setFont(font)
    ui.UpdatawithAPP_3.setStyleSheet("border-radius:10px;background-color:lightblue")
    ui.UpdatawithAPP_3.setObjectName("UpdatawithAPP_3")

    ui.UpdatawithAPP_4 = QtWidgets.QPushButton(parent=ui.tab_4_container)
    ui.UpdatawithAPP_4.setGeometry(QtCore.QRect(570, 200, 171, 61))
    font = QtGui.QFont()
    font.setPointSize(12)
    font.setBold(True)
    ui.UpdatawithAPP_4.setFont(font)
    ui.UpdatawithAPP_4.setStyleSheet("border-radius:10px;background-color:lightblue")
    ui.UpdatawithAPP_4.setObjectName("UpdatawithAPP_4")

    ui.UID_label = QtWidgets.QLabel(parent=ui.tab_4_container)
    ui.UID_label.setGeometry(QtCore.QRect(140, 100, 81, 31))
    font = QtGui.QFont()
    font.setPointSize(11)
    font.setBold(True)
    ui.UID_label.setFont(font)
    ui.UID_label.setScaledContents(False)
    ui.UID_label.setObjectName("Updata_progress_7")
    ui.Rec_ID_3 = QtWidgets.QLineEdit(parent=ui.tab_4_container)
    ui.Rec_ID_3.setGeometry(QtCore.QRect(220, 100, 631, 31))
    font = QtGui.QFont()
    font.setPointSize(11)
    font.setBold(True)
    ui.Rec_ID_3.setFont(font)
    ui.Rec_ID_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    ui.Rec_ID_3.setObjectName("Rec_ID_3")
    ui.Rec_ID_3.setStyleSheet("background-color: blue; color: white;") 
    ui.tabWidget.addTab(ui.tab_4, "")
    ui.tab_4_container.setVisible(True)

def show_file_info(filename):
    root =tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 显示询问对话框
    if 'collar_machine' in filename:
        response = messagebox.askyesno("提示", "这是一个圆领固件文件，是否要进行烧录？")
    else:
        response = messagebox.askyesno("提示", "这是一个罗纹下摆固件文件，是否要进行烧录？")
    return response
