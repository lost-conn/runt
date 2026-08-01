#!/usr/bin/env python3
"""Minimal client for rinch-debug's IPC server.

The `rinch-test` CLI that ships with rinch speaks newline-delimited JSON, but
`rinch-debug`'s server speaks a length-prefixed framed protocol with a
handshake, so the two cannot talk to each other. This speaks the server's
protocol.

  rinchctl.py <port> screenshot <out.png>
  rinchctl.py <port> click <x> <y>
  rinchctl.py <port> dom [max_depth]
  rinchctl.py <port> query <selector>
  rinchctl.py <port> text <node_id>
  rinchctl.py <port> scroll <x> <y> <dx> <dy>
  rinchctl.py <port> mouse_down <x> <y> [button]
  rinchctl.py <port> mouse_move <x> <y>
  rinchctl.py <port> mouse_up <x> <y> [button]
  rinchctl.py <port> wait
"""
import base64
import json
import socket
import struct
import sys


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.next_id = 1
        self._send({"protocol": "rinch-debug", "version": 1})
        self.hello = self._recv()

    def _send(self, obj):
        payload = json.dumps(obj).encode()
        self.sock.sendall(struct.pack(">I", len(payload)) + payload)

    def _recv(self):
        header = self._read_exact(4)
        (length,) = struct.unpack(">I", header)
        return json.loads(self._read_exact(length))

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("connection closed")
            buf += chunk
        return buf

    def call(self, method, **params):
        request = {"id": self.next_id, "method": method}
        if params:
            request["params"] = params
        self.next_id += 1
        self._send(request)
        return self._recv()


def main():
    port = int(sys.argv[1])
    verb = sys.argv[2]
    client = Client(port)

    if verb == "screenshot":
        out = sys.argv[3]
        response = client.call("screenshot")
        if response.get("type") != "bytes":
            print(json.dumps(response)[:400])
            sys.exit(1)
        png = base64.b64decode(response["data"])
        with open(out, "wb") as f:
            f.write(png)
        print(f"wrote {out} ({len(png)} bytes)")
    elif verb == "click":
        print(client.call("click", x=float(sys.argv[3]), y=float(sys.argv[4])))
    elif verb == "mouse_down":
        button = sys.argv[5] if len(sys.argv) > 5 else "left"
        print(client.call("mouse_down", x=float(sys.argv[3]), y=float(sys.argv[4]), button=button))
    elif verb == "mouse_move":
        print(client.call("mouse_move", x=float(sys.argv[3]), y=float(sys.argv[4])))
    elif verb == "mouse_up":
        button = sys.argv[5] if len(sys.argv) > 5 else "left"
        print(client.call("mouse_up", x=float(sys.argv[3]), y=float(sys.argv[4]), button=button))
    elif verb == "scroll":
        print(client.call("scroll", x=float(sys.argv[3]), y=float(sys.argv[4]),
                          delta_x=float(sys.argv[5]), delta_y=float(sys.argv[6])))
    elif verb == "dom":
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        print(json.dumps(client.call("dom_tree", max_depth=depth), indent=1)[:20000])
    elif verb == "query":
        print(json.dumps(client.call("query_selector", selector=sys.argv[3]))[:8000])
    elif verb == "text":
        print(json.dumps(client.call("get_text_content", id=int(sys.argv[3])))[:8000])
    elif verb == "wait":
        print(client.call("wait_frame"))
    else:
        raise SystemExit(f"unknown verb {verb}")


if __name__ == "__main__":
    main()
