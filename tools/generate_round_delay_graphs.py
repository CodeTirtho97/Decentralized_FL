#!/usr/bin/env python3
"""
Parse existing experiment logs and generate round-delay graphs.
Usage: python tools/generate_round_delay_graphs.py --results_dir results/2026_05_07/
Outputs: PNG files in figures/ directory suitable for thesis inclusion
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_round_durations(log_file):
    """
    Extract per-round total duration and comm time from a node log file.
    Returns: list of (round_num, train_time, comm_time, total_time, accuracy)
    """
    rounds = []
    round_pattern = re.compile(r'ROUND (\d+) / \d+')
    done_pattern = re.compile(r'Done\s*\|\s*Time:\s*([\d.]+)s\s*\|\s*Accuracy:\s*([\d.]+)%')
    comm_pattern = re.compile(r'Total comm time:\s*([\d.]+)s')

    current_round = None
    train_time = None
    comm_time = None
    acc = None

    with open(log_file) as f:
        for line in f:
            rm = round_pattern.search(line)
            if rm:
                current_round = int(rm.group(1))
                train_time = None
                comm_time = None
                acc = None

            dm = done_pattern.search(line)
            if dm and current_round:
                train_time = float(dm.group(1))
                acc = float(dm.group(2))

            cm = comm_pattern.search(line)
            if cm and current_round:
                comm_time = float(cm.group(1))
                if train_time is not None and comm_time is not None:
                    total = train_time + comm_time
                    rounds.append((current_round, train_time, comm_time, total, acc or 0.0))
                    train_time = None
                    comm_time = None

    return rounds


def plot_round_durations(exp_dir, exp_label, output_file, highlight_round=None):
    """
    For all node logs in exp_dir, plot average round duration per round.
    """
    all_rounds = {}
    for fname in sorted(os.listdir(exp_dir)):
        if fname.endswith('.log') and 'node' in fname:
            fpath = os.path.join(exp_dir, fname)
            rounds = parse_round_durations(fpath)
            for r, tr, co, tot, a in rounds:
                if r not in all_rounds:
                    all_rounds[r] = []
                all_rounds[r].append(tot)

    if not all_rounds:
        print(f"No round data found in {exp_dir}")
        return

    round_nums = sorted(all_rounds.keys())
    avg_durations = [np.mean(all_rounds[r]) for r in round_nums]
    std_durations = [np.std(all_rounds[r]) for r in round_nums]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(round_nums, avg_durations, 'b-o', markersize=3, linewidth=1.5,
            label='Avg round duration')
    ax.fill_between(round_nums,
                    [a - s for a, s in zip(avg_durations, std_durations)],
                    [a + s for a, s in zip(avg_durations, std_durations)],
                    alpha=0.2, color='blue', label='+-1 std dev (node spread)')
    if highlight_round:
        ax.axvline(x=highlight_round, color='red', linestyle='--', linewidth=1.5,
                   label=f'Node failure at Round {highlight_round}')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Round Duration (seconds)')
    ax.set_title(f'Round Duration per Round -- {exp_label}')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_fault_comm_times(fault_dir, output_file):
    """
    Plot comm time per round for Nodes 0, 2, 4 in fault experiment.
    Shows the cascade effect: Node 2 and 4 spike after Node 3 dies at R10.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {'node_0': 'steelblue', 'node_2': 'darkorange', 'node_4': 'green'}
    labels = {
        'node_0': 'Node 0 (distant)',
        'node_2': 'Node 2 (left neighbour of Node 3)',
        'node_4': 'Node 4 (right neighbour of Node 3)',
    }

    for node_name in ['node_0', 'node_2', 'node_4']:
        log_file = os.path.join(fault_dir, f'{node_name}.log')
        if not os.path.exists(log_file):
            log_file = os.path.join(fault_dir, f'{node_name.replace("_", "")}.log')
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping")
            continue
        rounds = parse_round_durations(log_file)
        if not rounds:
            continue
        rr = [r[0] for r in rounds]
        comm = [r[2] for r in rounds]
        ax.plot(rr, comm, '-o', markersize=3, linewidth=1.5,
                color=colors[node_name], label=labels[node_name])

    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5,
               label='Node 3 killed (Round 10)')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Communication Time (seconds)')
    ax.set_title('Per-Round Communication Time -- Decentralized Fault Experiment (Exp 5A)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate round-delay graphs from experiment log files.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--results_dir', default='results/',
                        help='Root results directory')
    parser.add_argument('--output_dir', default='figures/',
                        help='Directory for output PNG files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ring_iid = os.path.join(args.results_dir, 'decentralized_iid')
    if os.path.exists(ring_iid):
        plot_round_durations(ring_iid, 'Ring Gossip -- IID (Exp 3)',
                             os.path.join(args.output_dir, 'round_duration_ring_iid.png'))

    cent_iid = os.path.join(args.results_dir, 'centralized_iid')
    if os.path.exists(cent_iid):
        plot_round_durations(cent_iid, 'Centralised FedAvg -- IID (Exp 1)',
                             os.path.join(args.output_dir, 'round_duration_central_iid.png'))

    fault_dir = os.path.join(args.results_dir, 'decentralized_fault')
    if os.path.exists(fault_dir):
        plot_fault_comm_times(fault_dir,
                              os.path.join(args.output_dir, 'fault_comm_cascade.png'))
        plot_round_durations(fault_dir, 'Ring Gossip -- Fault Experiment (Exp 5A)',
                             os.path.join(args.output_dir, 'round_duration_fault.png'),
                             highlight_round=10)

    print("Done. Add generated PNGs to thesis figures/ and reference in LaTeX.")
