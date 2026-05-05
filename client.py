"""
client.py  --  Centralized FL Client  (Experiments 1, 2, 5-B)

Usage:
    python3 client.py <node_id> <server_ip> [--dist iid|non_iid] [--alpha F]
                      [--rounds N] [--fault-demo]

Examples:
    python3 client.py 1 172.31.21.108                            # Exp 1: IID
    python3 client.py 1 172.31.21.108 --dist non_iid             # Exp 2: Non-IID
    python3 client.py 1 172.31.21.108 --dist non_iid --fault-demo  # Exp 5-B: SPOF demo
"""

import argparse
import copy
import pickle
import socket
import time

from shared.log   import log, log_thin, SEP, THIN, setup_file_logging
from shared.model import CNNCifar
from shared.data  import get_loaders
from shared.net   import send_data, recv_data
from shared.train import train_local, evaluate

FAIL_ROUND = 16


# ============================================================
# SERVER COMMUNICATION
# ============================================================
def send_to_server(server_ip, port, weights_bytes, timeout=180):
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(30)
            s.connect((server_ip, port))
            send_data(s, weights_bytes)
            s.close()
            return True
        except Exception as e:
            log(f"      Upload attempt {attempt} failed  |  {e}"
                f"  |  {int(deadline - time.time())}s remaining")
            time.sleep(3)
    return False


def recv_from_server(server_ip, port, timeout=180):
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(30)
            s.connect((server_ip, port))
            raw = recv_data(s)
            s.close()
            return raw
        except Exception as e:
            log(f"      Download attempt {attempt} failed  |  {e}"
                f"  |  {int(deadline - time.time())}s remaining")
            time.sleep(3)
    return None


# ============================================================
# MAIN CLIENT LOOP
# ============================================================
def run_client(node_id, server_ip, distribution, alpha,
               num_rounds, local_epochs, batch_size,
               samples_per_node, port=9000, fault_demo=False):

    device = __import__('torch').device('cpu')

    # ---- Header ----
    print()
    print(SEP)
    print("  CENTRALIZED FL  --  CLIENT")
    print(SEP)
    print()
    print(f"  Node ID      : {node_id}")
    print(f"  Server       : {server_ip}:{port}  (Node 0)")
    print(f"  Distribution : {distribution.upper()}  (alpha={alpha})")
    print(f"  Rounds       : {num_rounds}  |  Epochs: {local_epochs}"
          f"  |  Batch: {batch_size}  |  Samples: {samples_per_node}")
    print(f"  Fault demo   : {'YES  --  Expect server to exit at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
    print(f"  Started at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Dataset ----
    log_thin("Loading CIFAR-10...")
    train_loader, test_loader, n_train, dist_label = get_loaders(
        node_id, distribution, alpha, samples_per_node, batch_size
    )
    log(f"Samples: {n_train}  ({dist_label})")
    log(f"Test set: 10,000  |  Model params: "
        f"{sum(p.numel() for p in CNNCifar().parameters()):,}")
    print()

    model   = CNNCifar().to(device)
    results = []

    for round_num in range(1, num_rounds + 1):

        # ---- Round header ----
        print()
        print(SEP)
        print(f"  ROUND {round_num:>2} / {num_rounds}"
              f"  --  Node {node_id}"
              f"  --  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(SEP)
        print()

        # ---- [1/3] Train ----
        log(f"[1/3]  LOCAL TRAINING  --  {local_epochs} epochs  |  SGD lr=0.01 momentum=0.5")
        print()

        t_train     = time.time()
        local_model = copy.deepcopy(model)
        local_model = train_local(local_model, train_loader, device, local_epochs)
        train_time  = time.time() - t_train
        acc_local   = evaluate(local_model, test_loader, device)

        print()
        log(f"      Done  |  Time: {train_time:.1f}s  |  Accuracy: {acc_local:.2f}%")
        print()

        # ---- [2/3] Upload to server ----
        log_thin(f"[2/3]  UPLOADING TO SERVER  ({server_ip}:{port})")

        weights_bytes = pickle.dumps(local_model.state_dict())
        log(f"      Model size: {len(weights_bytes):,} bytes  |  Connecting...")
        print()

        t_send = time.time()
        ok     = send_to_server(server_ip, port, weights_bytes)
        send_time = time.time() - t_send

        if not ok:
            print()
            print(SEP)
            print("  CRITICAL  --  SERVER UNREACHABLE")
            print(SEP)
            print()
            log(f"Cannot reach server at {server_ip}:{port}.")
            if fault_demo:
                log(f"")
                log(f"EXPERIMENT 5-B  --  SPOF CONFIRMED")
                log(f"")
                log(f"The central server has gone offline.")
                log(f"This client (Node {node_id}) cannot continue -- no alternative server exists.")
                log(f"Every other client is now in the same state.")
                log(f"The entire FL system has halted.")
                log(f"")
                log(f"In the decentralized experiment (Exp 5-A):")
                log(f"  When Node 3 died, all other 7 nodes continued training.")
                log(f"  Here, when the server dies, all 7 clients are stuck.")
                log(f"")
                log(f"CONCLUSION: Centralized FL has a Single Point of Failure.")
                log(f"            Decentralized FL does not.")
            print()
            print(SEP)
            break

        log(f"      Upload complete  |  Time: {send_time:.1f}s")
        print()

        # ---- [3/3] Download aggregated model ----
        log_thin("[3/3]  DOWNLOADING AGGREGATED MODEL FROM SERVER")

        log(f"      Waiting for server broadcast...")
        print()

        t_recv  = time.time()
        raw_agg = recv_from_server(server_ip, port)
        recv_time   = time.time() - t_recv
        total_comm  = send_time + recv_time

        if raw_agg is None:
            log(f"ERROR: Did not receive aggregated model. Stopping.")
            break

        model = CNNCifar().to(device)
        model.load_state_dict(pickle.loads(raw_agg))
        acc_agg = evaluate(model, test_loader, device)
        delta   = acc_agg - acc_local
        sign    = "+" if delta >= 0 else ""

        log(f"      Received: {len(raw_agg):,} bytes"
            f"  |  Upload: {send_time:.1f}s  |  Download: {recv_time:.1f}s"
            f"  |  Total comm: {total_comm:.1f}s")
        log(f"      Accuracy  local: {acc_local:.2f}%"
            f"  -->  aggregated: {acc_agg:.2f}%"
            f"  (change: {sign}{delta:.2f}%)")
        print()

        results.append({
            'round':     round_num,
            'local_acc': acc_local,
            'agg_acc':   acc_agg,
            'delta':     delta,
            'comm_time': total_comm,
        })

        if round_num < num_rounds:
            log(f"Waiting 15s before next round...")
            time.sleep(15)

    # ---- Final table ----
    if not results:
        return

    print()
    print(SEP)
    print(f"  CENTRALIZED FL  --  NODE {node_id}  --  RESULTS")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Local Acc':>12}  {'Aggregated':>12}"
          f"  {'Change':>10}  {'Comm(s)':>9}  {'Trend':>6}")
    print(f"  {THIN}")

    for r in results:
        sign  = "+" if r['delta'] >= 0 else ""
        arrow = "UP" if r['delta'] > 0.5 else ("DOWN" if r['delta'] < -0.5 else "FLAT")
        print(f"  {r['round']:<7} {r['local_acc']:>11.2f}%  "
              f"{r['agg_acc']:>11.2f}%  "
              f"{sign}{r['delta']:>9.2f}%  "
              f"{r['comm_time']:>8.1f}s  {arrow:>6}")

    completed  = len(results)
    final_acc  = results[-1]['agg_acc']
    best_acc   = max(r['agg_acc']  for r in results)
    best_rnd   = max(results, key=lambda r: r['agg_acc'])['round']
    avg_comm   = sum(r['comm_time'] for r in results) / completed

    print()
    print(f"  Rounds completed  : {completed} / {num_rounds}")
    print(f"  Final accuracy    : {final_acc:.2f}%  (Round {completed})")
    print(f"  Best  accuracy    : {best_acc:.2f}%  (Round {best_rnd})")
    print(f"  Avg comm / round  : {avg_comm:.1f}s  (upload + download)")
    print(f"  Finished at       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(SEP)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Centralized FL Client  (Experiments 1, 2, 5-B)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('node_id',   type=int,
                        help='Client node ID  (1-7)')
    parser.add_argument('server_ip',
                        help='Private IP of the server instance (Node 0)')
    parser.add_argument('--dist',    default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution  (default: iid)')
    parser.add_argument('--alpha',   type=float, default=0.5,
                        help='Dirichlet alpha for non_iid  (default: 0.5)')
    parser.add_argument('--rounds',  type=int, default=30,
                        help='Number of FL rounds  (default: 30)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 5-B: log SPOF confirmation when server dies')

    args = parser.parse_args()

    if args.fault_demo:
        exp_label = 'exp5b_centralized_noniid_spof'
    elif args.dist == 'non_iid':
        exp_label = 'exp2_centralized_noniid'
    else:
        exp_label = 'exp1_centralized_iid'
    log_path = setup_file_logging(exp_label, f'client_{args.node_id}')
    print(f"  Log file: {log_path}", flush=True)

    run_client(
        node_id          = args.node_id,
        server_ip        = args.server_ip,
        distribution     = args.dist,
        alpha            = args.alpha,
        num_rounds       = args.rounds,
        local_epochs     = 5,
        batch_size       = 64,
        samples_per_node = 2500,
        fault_demo       = args.fault_demo
    )
