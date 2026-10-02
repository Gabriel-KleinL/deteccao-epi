"""Inicia uma única instância local e abre a interface após HTTP e WS estarem prontos."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
LOCAL_URL = 'http://localhost:8080/interface.html'
CONFIG_URL = 'http://127.0.0.1:8080/api/config'
LAUNCH_URLS = {'i9-epi://abrir', 'i9-epi://abrir/'}


class StartupError(Exception):
    pass


def port_in_use(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def ready():
    try:
        # Este acesso é sempre direto ao Mac, mesmo se houver proxy configurado.
        with build_opener(ProxyHandler({})).open(CONFIG_URL, timeout=1) as response:
            config = json.load(response)
        if not (isinstance(config, dict) and config.get('http_port') == 8080
                and config.get('ws_port') == 8765 and config.get('camera_servidor') is False):
            return False
        # Valida o protocolo, sem encerrar uma conexão TCP antes do handshake.
        with connect('ws://127.0.0.1:8765', open_timeout=1, close_timeout=1, proxy=None):
            return True
    except (URLError, OSError, ValueError, WebSocketException):
        return False


@contextmanager
def startup_lock():
    folder = ROOT / 'logs'
    folder.mkdir(mode=0o700, exist_ok=True)
    with (folder / 'abertura-mac.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def start_server():
    env = os.environ.copy()
    env.update(PORT='8080', WS_PORT='8765', BIND_HOST='127.0.0.1',
               CAMERA_SERVIDOR='0', ABRIR_BROWSER='0', PRESERVAR_HISTORICO='1')
    logfile = ROOT / 'logs/servidor-mac.log'
    fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a') as log:
        return subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'servidor.py')], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )


def ensure_server(timeout=60):
    with startup_lock():
        if ready():
            return False
        if port_in_use(8080) or port_in_use(8765):
            raise StartupError('As portas 8080 ou 8765 estão ocupadas, mas o sistema local '
                               'não respondeu corretamente. Nenhum processo foi encerrado.')
        process = start_server()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise StartupError('O servidor não conseguiu iniciar. Consulte logs/servidor-mac.log.')
            if ready():
                return True
            time.sleep(0.25)
        raise StartupError('O servidor ainda não respondeu. Consulte logs/servidor-mac.log '
                           'e tente abrir novamente.')


def main(args=None):
    args = sys.argv[1:] if args is None else args
    # O link não pode fornecer comandos, arquivos, parâmetros ou destinos externos.
    if len(args) > 1 or (args and args[0] not in LAUNCH_URLS):
        raise StartupError('Link de abertura inválido.')
    ensure_server()
    subprocess.run(['/usr/bin/open', LOCAL_URL], check=True)


if __name__ == '__main__':
    try:
        main()
    except (StartupError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
