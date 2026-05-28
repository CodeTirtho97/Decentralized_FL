"""
shared/net.py  --  Socket communication utilities
                   Length-prefixed binary protocol used by all FL communication.
"""

import struct
import socket


def send_data(sock, data_bytes):
    sock.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_data(sock):
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None
    total = struct.unpack('>I', raw_len)[0]
    return recv_exact(sock, total)


def make_server_socket(ip, port, backlog=10):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((ip, port))
    srv.listen(backlog)
    return srv
