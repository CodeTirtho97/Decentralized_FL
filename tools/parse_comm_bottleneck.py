"""
parse_comm_bottleneck.py

Parse [COMM_SUMMARY] lines from experiment log files and print a markdown
table of per-round averages suitable for pasting into the thesis.

Usage:
    python tools/parse_comm_bottleneck.py --log_dir results/2026_05_07/centralized_iid/
    python tools/parse_comm_bottleneck.py --log_dir results/          # scans all subdirs
    python tools/parse_comm_bottleneck.py --log_dir results/ --arch ring fedstellar_ring

[COMM_SUMMARY] line format:
    [COMM_SUMMARY] round=X arch=<centralized|ring|fc|fedstellar_ring> node=<id|server>
      train_time=XX.XXs comm_time=XX.XXs round_time=XX.XXs
      bytes_sent=XXXXX bytes_recv=XXXXX
      push_retries=N timeout_hits=N neighbors_received=N fanout=N
      pre_blend_acc=XX.XX% post_blend_acc=XX.XX%
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

PATTERN = re.compile(
    r'\[COMM_SUMMARY\] '
    r'round=(?P<round>\d+) arch=(?P<arch>\S+) node=(?P<node>\S+) '
    r'train_time=(?P<train_time>[0-9.]+)s '
    r'comm_time=(?P<comm_time>[0-9.]+)s '
    r'round_time=(?P<round_time>[0-9.]+)s '
    r'bytes_sent=(?P<bytes_sent>\d+) '
    r'bytes_recv=(?P<bytes_recv>\d+) '
    r'(?:push_retries=(?P<push_retries>\d+) )?'
    r'(?:timeout_hits=(?P<timeout_hits>\d+) )?'
    r'(?:(?:neighbors|peers)_received=(?P<models_received>\d+) )?'
    r'(?:fanout=(?P<fanout>\d+) )?'
    r'pre_blend_acc=(?P<pre_acc>[0-9.N/A]+)% '
    r'post_blend_acc=(?P<post_acc>[0-9.N/A]+)%'
)

ARCH_DISPLAY = {
    'centralized':     'Centralized',
    'ring':            'Ring Gossip',
    'fc':              'FC Gossip',
    'fedstellar_ring': 'Fedstellar Ring',
}

TARGET_ACC = 50.0


def parse_file(path):
    records = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            pre  = m.group('pre_acc')
            post = m.group('post_acc')
            records.append({
                'round':           int(m.group('round')),
                'arch':            m.group('arch'),
                'node':            m.group('node'),
                'train_time':      float(m.group('train_time')),
                'comm_time':       float(m.group('comm_time')),
                'round_time':      float(m.group('round_time')),
                'bytes_sent':      int(m.group('bytes_sent')),
                'bytes_recv':      int(m.group('bytes_recv')),
                'push_retries':    int(m.group('push_retries'))    if m.group('push_retries')    else None,
                'timeout_hits':    int(m.group('timeout_hits'))    if m.group('timeout_hits')    else None,
                'models_received': int(m.group('models_received')) if m.group('models_received') else None,
                'fanout':          int(m.group('fanout'))          if m.group('fanout')          else None,
                'pre_acc':         float(pre)  if pre  not in ('N/A', '') else None,
                'post_acc':        float(post) if post not in ('N/A', '') else None,
            })
    return records


def scan(log_dir):
    files = glob.glob(os.path.join(log_dir, '**', '*.log'), recursive=True)
    if not files:
        files = glob.glob(os.path.join(log_dir, '*.log'))
    return sorted(files)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

def stddev(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def total(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)

def pct_nonzero(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return 100.0 * sum(1 for v in vals if v > 0) / len(vals)


def rounds_to_target(records, target=TARGET_ACC):
    by_node = defaultdict(list)
    for r in records:
        if r['post_acc'] is not None:
            by_node[r['node']].append((r['round'], r['post_acc']))

    first_rounds = []
    for node, rounds in by_node.items():
        rounds_sorted = sorted(rounds, key=lambda x: x[0])
        for rnd, acc in rounds_sorted:
            if acc >= target:
                first_rounds.append(rnd)
                break

    if not first_rounds:
        return None
    first_rounds.sort()
    mid = len(first_rounds) // 2
    return first_rounds[mid]


def summarise(records, arch_filter=None):
    by_arch = defaultdict(list)
    for r in records:
        if arch_filter and r['arch'] not in arch_filter:
            continue
        by_arch[r['arch']].append(r)

    summaries = {}
    for arch, recs in sorted(by_arch.items()):
        post_accs   = [r['post_acc'] for r in recs if r['post_acc'] is not None]
        has_retry   = any(r['push_retries'] is not None for r in recs)

        avg_comm    = mean([r['comm_time']  for r in recs])
        avg_round   = mean([r['round_time'] for r in recs])
        avg_tx      = mean([r['bytes_sent'] for r in recs])
        total_bytes = total(r['bytes_sent'] + r['bytes_recv'] for r in recs)
        final_acc   = post_accs[-1] if post_accs else None

        comm_times  = [r['comm_time'] for r in recs if r['comm_time'] > 0]
        tx_during   = [r['bytes_sent'] for r in recs if r['comm_time'] > 0]
        throughput  = mean([b / t for b, t in zip(tx_during, comm_times)]) / 1024 if comm_times else None

        comm_eff    = (total_bytes / 1024 / 1024) / final_acc if final_acc and final_acc > 0 else None

        summaries[arch] = {
            'n':               len(recs),
            'avg_train':       mean([r['train_time']  for r in recs]),
            'std_train':       stddev([r['train_time'] for r in recs]),
            'avg_comm':        avg_comm,
            'std_comm':        stddev([r['comm_time']  for r in recs]),
            'avg_round':       avg_round,
            'comm_overhead':   (avg_comm / avg_round * 100) if avg_round > 0 else 0,
            'avg_tx_kb':       avg_tx / 1024,
            'avg_rx_kb':       mean([r['bytes_recv']  for r in recs]) / 1024,
            'total_mb':        total_bytes / 1024 / 1024,
            'throughput_kbs':  throughput,
            'comm_efficiency': comm_eff,
            'rounds_to_target': rounds_to_target(recs, TARGET_ACC),
            'avg_retries':     mean([r['push_retries']    for r in recs]) if has_retry else None,
            'total_timeouts':  total(r['timeout_hits'] or 0 for r in recs) if has_retry else None,
            'timeout_pct':     pct_nonzero([r['timeout_hits'] for r in recs]) if has_retry else None,
            'avg_fanout':      mean([r['fanout'] for r in recs if r['fanout'] is not None]) or None,
            'final_acc':       final_acc,
            'best_acc':        max(post_accs) if post_accs else None,
        }
    return summaries


def fmt(val, fmt_str, missing='N/A', suffix=''):
    if val is None:
        return missing
    return format(val, fmt_str) + suffix


def print_table(summaries):
    if not summaries:
        print("No data to display.")
        return

    arches = list(summaries.keys())
    labels = [ARCH_DISPLAY.get(a, a) for a in arches]

    def row(label, fn):
        cells = [f"**{label}**"] + [fn(summaries[a]) for a in arches]
        print('| ' + ' | '.join(cells) + ' |')

    def sep(label=''):
        print(f"| **{label}** | " + " | ".join(['--'] * len(arches)) + " |")

    header = ['**Metric**'] + labels
    print()
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join([':---'] + [':---:'] * len(arches)) + '|')

    sep('TIMING')
    row('Avg train time / round',        lambda s: f"{s['avg_train']:.2f} s  (+-{s['std_train']:.2f})")
    row('Avg comm time / round',         lambda s: f"{s['avg_comm']:.2f} s  (+-{s['std_comm']:.2f})")
    row('Avg round duration',            lambda s: f"{s['avg_round']:.2f} s")
    row('Comm overhead (% of round)',    lambda s: f"{s['comm_overhead']:.1f}%")

    sep('COMMUNICATION BOTTLENECK')
    row('Throughput during comm window', lambda s: fmt(s['throughput_kbs'], '.1f', suffix=' KB/s'))
    row('Total data transferred',        lambda s: f"{s['total_mb']:.1f} MB")
    row('Avg bytes sent / round',        lambda s: f"{s['avg_tx_kb']:.1f} KB")
    row('Avg bytes recv / round',        lambda s: f"{s['avg_rx_kb']:.1f} KB")
    row('Comm efficiency (MB / % acc)',  lambda s: fmt(s['comm_efficiency'], '.2f', suffix=' MB/%'))
    row('Communication fanout',          lambda s: fmt(s['avg_fanout'], '.0f'))

    sep('FAULT / RELIABILITY')
    row('Avg push retries / round',      lambda s: fmt(s['avg_retries'], '.2f'))
    row('Total timeout events',          lambda s: fmt(s['total_timeouts'], 'd'))
    row('Rounds with timeout (%)',        lambda s: fmt(s['timeout_pct'], '.1f', suffix='%'))

    sep('CONVERGENCE')
    row(f'Rounds to {TARGET_ACC:.0f}% accuracy',  lambda s: fmt(s['rounds_to_target'], 'd'))
    row('Final accuracy',                lambda s: f"{s['final_acc']:.2f}%" if s['final_acc'] is not None else 'N/A')
    row('Best accuracy',                 lambda s: f"{s['best_acc']:.2f}%"  if s['best_acc']  is not None else 'N/A')

    sep()
    row('COMM_SUMMARY lines parsed',     lambda s: str(s['n']))
    print()

    print("### Bottleneck Verdict")
    for arch, s in summaries.items():
        label = ARCH_DISPLAY.get(arch, arch)
        co = s['comm_overhead']
        verdict = "COMMUNICATION-BOUND" if co > 40 else ("BALANCED" if co > 20 else "COMPUTE-BOUND")
        print(f"- **{label}**: comm overhead = {co:.1f}% -> **{verdict}**")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Parse [COMM_SUMMARY] log lines and print a markdown comparison table.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--log_dir', required=True,
                        help='Directory to scan recursively for .log files')
    parser.add_argument('--arch', nargs='+', metavar='ARCH',
                        help='Filter by arch name(s): centralized ring fc fedstellar_ring  (default: all)')
    parser.add_argument('--target_acc', type=float, default=TARGET_ACC,
                        help=f'Accuracy target for rounds_to_target metric (default: {TARGET_ACC}%%)')
    args = parser.parse_args()

    global TARGET_ACC
    TARGET_ACC = args.target_acc

    files = scan(args.log_dir)
    if not files:
        print(f"No .log files found under: {args.log_dir}")
        return

    print(f"Scanning {len(files)} log file(s) under: {args.log_dir}")
    all_records = []
    for path in files:
        recs = parse_file(path)
        rel  = os.path.relpath(path, args.log_dir)
        if recs:
            print(f"  {rel}: {len(recs)} COMM_SUMMARY line(s)")
        all_records.extend(recs)

    if not all_records:
        print("\nNo [COMM_SUMMARY] lines found.")
        print("Make sure you are running the updated scripts that emit [COMM_SUMMARY] lines.")
        return

    summaries = summarise(all_records, arch_filter=args.arch)
    print_table(summaries)


if __name__ == '__main__':
    main()
