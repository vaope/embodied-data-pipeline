"""Loopback-only range server. Media access is restricted to indexed playback files."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from .features.feature_store import FeatureStore

def make_server(artifact_root, port=8771):
    store = FeatureStore(artifact_root)
    allowed = {r['clip_id']: Path(r['playback_video_path']) for r in store.rows}
    root = store.root
    class Handler(SimpleHTTPRequestHandler):
        def send_head(self):
            self.remaining = None
            route = unquote(urlsplit(self.path).path)
            if route.startswith('/media/'):
                path = allowed.get(route[len('/media/'):])
                if path is None:
                    self.send_error(404)
                    return None
            else:
                path = Path(self.translate_path(self.path)).resolve()
                if not path.is_relative_to(root):
                    self.send_error(403)
                    return None
                if path.is_dir():
                    return super().send_head()
            if not path.is_file():
                self.send_error(404)
                return None
            size = path.stat().st_size
            value = self.headers.get('Range')
            start, end, status = 0, size-1, 200
            if value:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return None
                if not match[1]:
                    start = max(0, size-int(match[2]))
                else:
                    start = int(match[1])
                    if match[2]:
                        end = min(end, int(match[2]))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return None
                status = 206
            f = path.open('rb')
            f.seek(start)
            self.remaining = end-start+1
            self.send_response(status)
            self.send_header('Content-Type', self.guess_type(str(path)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(self.remaining))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            return f

        def copyfile(self, source, output):
            if self.remaining is None:
                return super().copyfile(source, output)
            try:
                while self.remaining:
                    block = source.read(min(1024*1024, self.remaining))
                    if not block:
                        break
                    output.write(block)
                    self.remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(root)))
