# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/hs-real-anime-y11n-640-fp32.onnx', 'models'),
        ('models/mobile_sam_image_encoder.onnx', 'models'),
        ('models/sam_mask_decoder_single.onnx', 'models'),
        ('models/MODEL_SOURCES.md', 'models'),
        ('models/licenses/MobileSAM-LICENSE.txt', 'models/licenses'),
        ('models/licenses/Hotscreen-YOLO-LICENSE.txt', 'models/licenses'),
        ('assets/app_icon_girl.ico', 'assets'),
    ],
    hiddenimports=[],
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
    name='PixivSafeMosaic',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon_girl.ico',
)
