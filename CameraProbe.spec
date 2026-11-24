# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['camera_probe.py'],
    pathex=[],
    binaries=[],
    datas=[('app_icon.ico', '.'), ('C:/Users/Chenzhichao/AppData/Roaming/Python/Python313/site-packages/hik_camera/MvCameraNode-CH.csv', 'hik_camera')],
    hiddenimports=['hik_camera', 'boxx'],
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
    name='CameraProbe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
