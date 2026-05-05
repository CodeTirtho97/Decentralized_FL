"""
shared/log.py  --  Logging utilities
                   All output formatting lives here. Import this everywhere.
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
    Create  logs/<exp_label>_<YYYYMMDD_HHMM>/<node_label>.log
    and redirect sys.stdout so all print() calls go to both
    the terminal and the log file automatically.

    Timestamp is rounded to the minute so all nodes started
    within the same experiment run share the same folder name.

    Returns the full path of the log file created.
    """
    ts_dir   = time.strftime('%Y%m%d_%H%M')
    log_dir  = os.path.join('logs', f'{exp_label}_{ts_dir}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{node_label}.log')

    f          = open(log_path, 'w', buffering=1, encoding='utf-8')
    sys.stdout = _Tee(f)
    return log_path
