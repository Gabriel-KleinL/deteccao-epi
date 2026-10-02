"""Gera o aplicativo no projeto; --registrar associa apenas i9-epi:// neste Mac."""
import argparse
from pathlib import Path
import plistlib
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / '.macos/I9 EPI Local.app'
LSREGISTER = '/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister'


def build():
    APP.parent.mkdir(exist_ok=True)
    escaped_root = str(ROOT).replace('\\', '\\\\').replace('"', '\\"')
    source = (ROOT / 'macos/I9EPILocal.applescript').read_text().replace('__PROJECT_ROOT__', escaped_root)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'launcher.applescript'
        path.write_text(source)
        subprocess.run(['/usr/bin/osacompile', '-o', str(APP), str(path)], check=True)
    plist_path = APP / 'Contents/Info.plist'
    with plist_path.open('rb') as file:
        info = plistlib.load(file)
    info.update(CFBundleIdentifier='br.com.in9automacao.epi.local',
                CFBundleName='I9 EPI Local', CFBundleDisplayName='I9 EPI Local',
                CFBundleShortVersionString='1.0', CFBundleVersion='1', LSUIElement=True,
                CFBundleURLTypes=[{'CFBundleURLName':'Sistema EPI neste Mac',
                                   'CFBundleTypeRole':'Viewer', 'CFBundleURLSchemes':['i9-epi']}])
    with plist_path.open('wb') as file:
        plistlib.dump(info, file)
    # A alteração do Info.plist invalida a assinatura gerada pelo osacompile.
    subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(APP)], check=True)
    return APP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registrar', action='store_true')
    args = parser.parse_args()
    app = build()
    if args.registrar:
        subprocess.run([LSREGISTER, '-f', str(app)], check=True)
    print(('Registrado' if args.registrar else 'Aplicativo preparado') + ': ' + str(app))
