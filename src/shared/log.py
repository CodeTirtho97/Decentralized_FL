"""
shared/log.py  --  Logging utilities
                   All output formatting lives here. Import this everywhere.

Log structure:
    logs/YYYY_MM_DD/
    ├── centralized_iid/        server.log, client_1.log ... client_N.log
    ├── centralized_noniid/     server.log, client_1.log ... client_N.log
    ├── decentralized_iid/      node_0.log ... node_N.log
    ├── decentralized_noniid/   node_0.log ... node_N.log
    ├── centralized_spof/       server.log, client_1.log ... client_N.log
    └── decentralized_fault/    node_0.log ... node_N.log
"""

import os
import sys
import time

SEP  = "=" * 72
THIN = "-" * 72


def ts():
    return time.strftime('%H:%M:%S')


def log(msg):
    print(f"  [{ts()}]  {msg}", flush=True)


def log_section(title):
    print(f"\n{SEP}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{SEP}\n", flush=True)


def log_thin(title=""):
    if title:
        print(f"\n{THIN}", flush=True)
        print(f"  {title}", flush=True)
        print(f"{THIN}\n", flush=True)
    else:
        print(f"\n{THIN}\n", flush=True)


# ============================================================
# FILE LOGGING  --  Tee stdout to terminal + log file
# ============================================================
class _Tee:
    """Mirror all stdout writes to a file simultaneously."""
    def __init__(self, file):
        self._stdout = sys.stdout
        self._file   = file

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def fileno(self):
        return self._stdout.fileno()

    @property
    def encoding(self):
        return self._stdout.encoding

    @property
    def errors(self):
        return self._stdout.errors


def setup_file_logging(exp_label, node_label):
    """
    Create  logs/YYYY_MM_DD/<exp_label>/<node_label>.log
    and redirect sys.stdout so all print() calls go to both
    the terminal and the log file automatically.

    Re-running the same experiment on the same day overwrites
    the previous file so only the latest run is kept.

    Returns the full path of the log file created.
    """
    date_dir = time.strftime('%Y_%m_%d')
    log_dir  = os.path.join('logs', date_dir, exp_label)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{node_label}.log')

    f          = open(log_path, 'w', buffering=1, encoding='utf-8')
    sys.stdout = _Tee(f)
    return log_path
