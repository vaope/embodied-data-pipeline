"""Loopback report server, restricted to HTML/JSON and rendered MP4 artifacts."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def make_server(root, port=8797):
    root = Path(root).resolve()

    class Handler(SimpleHTTPRequestHandler):
        def send_head(self):
            self.remaining = 0
            route = unquote(urlsplit(self.path).path).lstrip('/')
            path = (root / (route or 'index.html')).resolve()
            if not path.is_relative_to(root) or path.suffix not in ('.html','.json','.mp4'):
                self.send_error(403)
                return None
            if not path.is_file():
                self.send_error(404)
                return None
            size = path.stat().st_size
            start, end, status = 0,size-1,200
            value = self.headers.get('Range')
            if value:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)',value)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return None
                if not match[1]:
                    start = max(0,size-int(match[2]))
                else:
                    start = int(match[1])
                    if match[2]:
                        end = min(end,int(match[2]))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range',f'bytes */{size}')
                    self.end_headers()
                    return None
                status = 206
            stream = path.open('rb')
            stream.seek(start)
            self.remaining = end-start+1
            self.send_response(status)
            self.send_header('Content-Type', self.guess_type(str(path)))
            self.send_header('Content-Length',str(self.remaining))
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','no-cache')
            if status == 206:
                self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            return stream

        def copyfile(self, source, output):
            try:
                while self.remaining:
                    block = source.read(min(1024*1024,self.remaining))
                    if not block:
                        break
                    output.write(block)
                    self.remaining -= len(block)
            except (BrokenPipeError,ConnectionResetError):
                pass

    return ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,directory=str(root)))


def main():
    p = argparse.ArgumentParser(description='Serve sequence reports on loopback')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--port',type=int,default=8797)
    args = p.parse_args()
    server = make_server(args.root,args.port)
    print(f'http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
