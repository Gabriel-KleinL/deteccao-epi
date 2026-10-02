from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from macos import abrir_sistema as launcher


class LocalLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = patch.object(launcher, 'ROOT', Path(self.tmp.name))
        root.start()
        self.addCleanup(root.stop)

    def test_existing_server_is_reused(self):
        with patch.object(launcher, 'ready', return_value=True), \
                patch.object(launcher, 'start_server') as start:
            self.assertFalse(launcher.ensure_server())
            start.assert_not_called()

    def test_http_alone_is_not_ready_without_websocket(self):
        config = b'{"http_port":8080,"ws_port":8765,"camera_servidor":false}'
        for available in (False, True):
            with self.subTest(websocket=available), \
                    patch.object(launcher, 'build_opener') as opener, \
                    patch.object(launcher, 'connect') as connection:
                opener.return_value.open.return_value = io.BytesIO(config)
                if not available:
                    connection.side_effect = OSError('WebSocket indisponível')
                self.assertEqual(launcher.ready(), available)

    def test_concurrent_clicks_start_only_one_server(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(launcher, 'ready', side_effect=[False, True, True]), \
                patch.object(launcher, 'port_in_use', return_value=False), \
                patch.object(launcher, 'start_server', return_value=process) as start:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: launcher.ensure_server(), range(2)))
        self.assertEqual(sorted(results), [False, True])
        start.assert_called_once()

    def test_busy_port_does_not_start_or_kill_a_process(self):
        with patch.object(launcher, 'ready', return_value=False), \
                patch.object(launcher, 'port_in_use', return_value=True), \
                patch.object(launcher, 'start_server') as start:
            with self.assertRaises(launcher.StartupError):
                launcher.ensure_server()
            start.assert_not_called()

    def test_failed_start_does_not_open_browser(self):
        process = Mock()
        process.poll.return_value = 1
        with patch.object(launcher, 'ready', return_value=False), \
                patch.object(launcher, 'port_in_use', return_value=False), \
                patch.object(launcher, 'start_server', return_value=process), \
                patch.object(launcher.subprocess, 'run') as browser:
            with self.assertRaises(launcher.StartupError):
                launcher.main(['i9-epi://abrir'])
            browser.assert_not_called()

    def test_timeout_is_reported_without_opening_browser(self):
        with patch.object(launcher, 'ready', return_value=False), \
                patch.object(launcher, 'port_in_use', return_value=False), \
                patch.object(launcher, 'start_server'), \
                patch.object(launcher.time, 'monotonic', side_effect=[100, 161]):
            with self.assertRaises(launcher.StartupError):
                launcher.ensure_server()

    def test_link_cannot_supply_commands_or_another_destination(self):
        with patch.object(launcher, 'ensure_server') as server, \
                patch.object(launcher.subprocess, 'run') as browser:
            for value in ['i9-epi://abrir?cmd=whoami', 'i9-epi://abrir;id',
                          'i9-epi://outro', 'https://example.org', 'file:///tmp/test']:
                with self.subTest(value=value), self.assertRaises(launcher.StartupError):
                    launcher.main([value])
            server.assert_not_called()
            browser.assert_not_called()
            launcher.main(['i9-epi://abrir'])
            server.assert_called_once()
            browser.assert_called_once_with(['/usr/bin/open', launcher.LOCAL_URL], check=True)

    def test_start_is_local_detached_and_preserves_history(self):
        (launcher.ROOT / 'logs').mkdir()
        with patch.object(launcher.subprocess, 'Popen') as process:
            launcher.start_server()
        args, kwargs = process.call_args
        self.assertEqual(args[0][-1], str(launcher.ROOT / 'servidor.py'))
        self.assertEqual(kwargs['env']['BIND_HOST'], '127.0.0.1')
        self.assertEqual(kwargs['env']['PRESERVAR_HISTORICO'], '1')
        self.assertEqual(kwargs['env']['CAMERA_SERVIDOR'], '0')
        self.assertTrue(kwargs['start_new_session'])


if __name__ == '__main__':
    unittest.main()
