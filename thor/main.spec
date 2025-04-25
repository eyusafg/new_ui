# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('utils/*.pyd', 'utils'), ('thor_barycenter.pyd', '.')],  # 添加所有.pyd文件
    hiddenimports=['Crypto.Util.Padding','pandas','queue','logging.handlers','logging','threading','threading.Thread','pathlib','wmi','hmac','psutil','uuid','hashlib','Crypto.Util.Padding.pad','Crypto.Util.Padding.unpad','Crypto.Cipher.AES','datetime','json','PyCameraList.camera_device','shutil','struct','cv2','numpy','serial','serial.tools.list_ports','PyQt5.QtCore','PyQt5.QtGui','PyQt5.QtWidgets','onnxruntime','yaml'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    # optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='雷神',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='雷神',
)
