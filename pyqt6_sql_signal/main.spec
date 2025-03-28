# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('Signal_Ui.pyd', '.'), ('iap.pyd', '.'), ('Bin_Unpacking.pyd', '.'), ('test_sql.pyd', '.'), ('Bootloader_func.pyd', '.'), ('data_QDialog.pyd', '.'), ('full_features.pyd', '.'), ('get_company_QDialog.pyd', '.'), ('lite_features.pyd', '.'), ('logger.pyd', '.'), ('query_dialog.pyd', '.'), ('Verify.pyd', '.')],  # 添加所有.pyd文件
    hiddenimports=['Signal_Ui','iap','Bin_Unpacking', 'test_sql', 'Bootloader_func', 'data_QDialog', 'full_features', 'get_company_QDialog', 'lite_features', 'logger', 'query_dialog', 'Verify', 'sys', 'time', 'os', 'datetime', 'threading', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.sip', 'serial', 'serial.tools.list_ports', 'win32api', 'win32con', 'thinter', 'tkinter.filedialog', 'Crypto.Cipher.AES', 'Crypto.Util.Padding', 'Crypto.Util.Padding.unpad', 'Crypto.Util.Padding.pad', 'Verify.data_verify', 'logging', 'ctypes', 'atexit', 'logging.FileHandler', 'pymysql', 'pymysql.err.IntegrityError', 'crcmod', 'binascii'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='IAP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    version='file_version_info.txt',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
