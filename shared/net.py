"""
shared/net.py  --  Socket communication utilities
                   Length-prefixed binary protocol used by all FL communication.
"""

import struct
import socket


def send_data(sock, data_bytes):
    """Send a length-prefixed payload: 4-byte big-endian length, then the bytes."""
    sock.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)


def recv_exact(sock, n):
    """Read exactly n bytes from sock, looping until done. Returns None if the peer closes early."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_data(sock):
    """Read one length-prefixed payload written by send_data(). Returns None on a closed connection."""
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None
    total = struct.unpack('>I', raw_len)[0]
    return recv_exact(sock, total)


def make_server_socket(ip, port, backlog=10):
    """Create a bound, listening TCP socket with SO_REUSEADDR set (so restarts don't hit 'address in use')."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((ip, port))
    srv.listen(backlog)
    return srv
