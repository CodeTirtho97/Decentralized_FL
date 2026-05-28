"""
node.py  --  Decentralized FL Node  (Experiments 3, 4, 5-A)
             Ring topology, synchronous gossip weight sharing.

             Each round: Train -> Push to neighbors -> Receive from neighbors
                         -> Blend (equal-weight avg) -> Evaluate

Usage:
    PYTHONPATH=src python3 src/node.py <node_id> <my_ip> <left_ip> <right_ip>
                                        [--dist iid|non_iid] [--alpha F]
                                        [--rounds N] [--num-nodes N] [--fault-demo]

Examples:
    PYTHONPATH=src python3 src/node.py 0 192.168.1.10 192.168.1.17 192.168.1.11
    PYTHONPATH=src python3 src/node.py 0 192.168.1.10 192.168.1.17 192.168.1.11 --dist non_iid
    PYTHONPATH=src python3 src/node.py 0 192.168.1.10 192.168.1.17 192.168.1.11 --fault-demo
"""

import argparse
import pickle
import socket
import sys
import threading
import time

from shared.log   import log, log_thin, SEP, THIN, setup_file_logging
from shared.model import CNNCifar
from shared.data  import get_loaders
from shared.net   import send_data, recv_data, make_server_socket
from shared.train import train_local, evaluate, fedavg

BASE_PORT  = 8000    # node i listens on BASE_PORT + i
FAIL_ROUND = 10      # Experiment 5-A: node 3 exits at this round


# ============================================================
# PUSH TO ONE NEIGHBOR  --  retries until timeout
# ============================================================
def push_to_neighbor(label, ip, port, weights_bytes, results_dict, timeout=90):
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((ip, port))
            send_data(s, weights_bytes)
            s.close()
            log(f"      PUSH OK    -->  {label}  ({ip}:{port})"
                f"  |  {len(weights_bytes)/1024:.1f} KB  |  attempt {attempt}")
            results_dict[label] = 'ok'
            results_dict[label + '_attempts'] = attempt
            return
        except Exception as e:
            if time.time() < deadline:
                log(f"      PUSH retry {attempt}  -->  {label}"
                    f"  |  {e}  |  {int(deadline - time.time())}s left")
                time.sleep(2)
    log(f"      PUSH FAIL  -->  {label}  ({ip}:{port})  |  Neighbor may be down.")
    results_dict[label] = 'fail'
    results_dict[label + '_attempts'] = attempt


# ============================================================
# RECEIVE FROM NEIGHBORS  --  per-round blocking listener
# ============================================================
def receive_from_neighbors(my_ip, listen_port, expected_count,
                           results_dict, ready_event, timeout=120):
    srv = make_server_socket(my_ip, listen_port, backlog=expected_count + 1)
    ready_event.set()   # socket is bound -- caller may now push
    srv.settimeout(2.0)
    received = []
    deadline = time.time() + timeout

    while len(received) < expected_count and time.time() < deadline:
        try:
            conn, addr = srv.accept()
            conn.settimeout(60)
            raw = recv_data(conn)
            conn.close()
            if raw:
                received.append((addr[0], raw))
                log(f"      RECV OK    <--  {addr[0]}"
                    f"  |  {len(raw)/1024:.1f} KB"
                    f"  |  {len(received)}/{expected_count}")
        except socket.timeout:
            continue
        except Exception as e:
            if time.time() < deadline:
                log(f"      RECV ERR: {e}")

    srv.close()
    if len(received) < expected_count:
        log(f"      WARNING: Received {len(received)}/{expected_count} models "
            f"(neighbor down or timeout).")
    results_dict['received'] = received


# ============================================================
# MAIN NODE LOOP
# ============================================================
def run_node(node_id, my_ip, left_ip, right_ip,
             num_rounds, local_epochs, batch_size,
             samples_per_node, distribution, alpha, fault_demo, num_nodes):

    device      = __import__('torch').device('cpu')
    left_id     = (node_id - 1) % num_nodes
    right_id    = (node_id + 1) % num_nodes
    listen_port = BASE_PORT + node_id
    left_port   = BASE_PORT + left_id
    right_port  = BASE_PORT + right_id

    # ---- Header ----
    print()
    print(SEP)
    print("  DECENTRALIZED FL  --  NODE")
    print(SEP)
    print()
    print(f"  Node ID      : {node_id}  (listen port: {listen_port})")
    print(f"  Left  (Node {left_id})  : {left_ip}:{left_port}")
    print(f"  Right (Node {right_id})  : {right_ip}:{right_port}")
    print(f"  Distribution : {distribution.upper()}  (alpha={alpha})")
    print(f"  Rounds       : {num_rounds}  |  Epochs: {local_epochs}"
          f"  |  Batch: {batch_size}  |  Samples: {samples_per_node}")
    print(f"  Gossip       : Synchronous  (train -> push -> receive -> blend)")
    print(f"  Fault demo   : {'YES  --  Node 3 exits at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
    print(f"  Started at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Dataset ----
    log_thin("Loading CIFAR-10...")
    train_loader, test_loader, n_train, dist_label = get_loaders(
        node_id, distribution, alpha, samples_per_node, batch_size, num_nodes
    )
    log(f"Samples: {n_train}  ({dist_label})")
    log(f"Test set: 10,000  |  Model params: "
        f"{sum(p.numel() for p in CNNCifar().parameters()):,}")
    print()

    model          = CNNCifar().to(device)
    blends_total   = 0
    bytes_tx_total = 0
    bytes_rx_total = 0
    results        = []

    for round_num in range(1, num_rounds + 1):

        # ---- Experiment 5-A: Node 3 deliberately exits ----
        if fault_demo and node_id == 3 and round_num == FAIL_ROUND:
            print()
            print(SEP)
            print("  EXPERIMENT 5-A  --  NODE 3 FAILURE")
            print(SEP)
            print()
            log(f"Node 3 deliberately exiting at round {FAIL_ROUND}.")
            log(f"Neighbors (Node {left_id} left, Node {right_id} right)"
                f" will detect this on their next push and continue.")
            log(f"All remaining {num_nodes - 1} nodes will complete all {num_rounds} rounds.")
            log(f"This proves no single point of failure in decentralized design.")
            print()
            sys.exit(0)

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

        t_train        = time.time()
        model          = train_local(model, train_loader, device, local_epochs)
        train_time     = time.time() - t_train
        acc_post_train = evaluate(model, test_loader, device)

        print()
        log(f"      Done  |  Time: {train_time:.3f}s  |  Accuracy: {acc_post_train:.2f}%")
        print()

        # ---- [2/3] Exchange with neighbors (push + receive simultaneously) ----
        log_thin(f"[2/3]  EXCHANGE WITH RING NEIGHBORS")
        log(f"      Left  (Node {left_id})  :  {left_ip}:{left_port}")
        log(f"      Right (Node {right_id})  :  {right_ip}:{right_port}")
        print()

        weights_bytes = pickle.dumps(model.state_dict())
        log(f"      Model serialized: {len(weights_bytes)/1024:.1f} KB")
        log(f"      Opening receive listener on port {listen_port}...")
        print()

        recv_results = {}
        socket_ready = threading.Event()
        recv_t = threading.Thread(
            target=receive_from_neighbors,
            args=(my_ip, listen_port, 2, recv_results, socket_ready),
            daemon=True
        )
        recv_t.start()
        socket_ready.wait(timeout=10)

        push_results = {}
        t_comm       = time.time()
        push_l = threading.Thread(target=push_to_neighbor,
                                   args=(f"Node-{left_id}", left_ip, left_port,
                                         weights_bytes, push_results))
        push_r = threading.Thread(target=push_to_neighbor,
                                   args=(f"Node-{right_id}", right_ip, right_port,
                                         weights_bytes, push_results))
        push_l.daemon = push_r.daemon = True
        push_l.start()
        push_r.start()
        push_l.join(timeout=95)
        push_r.join(timeout=95)

        left_ok   = push_results.get(f"Node-{left_id}")  == 'ok'
        right_ok  = push_results.get(f"Node-{right_id}") == 'ok'
        push_fail = not left_ok or not right_ok

        left_attempts  = push_results.get(f"Node-{left_id}_attempts",  1)
        right_attempts = push_results.get(f"Node-{right_id}_attempts", 1)
        push_retries   = (left_attempts - 1) + (right_attempts - 1)
        timeout_hits   = (0 if left_ok else 1) + (0 if right_ok else 1)

        log(f"      Push done  |  Left: {'OK' if left_ok else 'FAIL'}"
            f"  |  Right: {'OK' if right_ok else 'FAIL'}"
            f"  |  retries={push_retries}  timeouts={timeout_hits}")
        if push_fail and fault_demo:
            log(f"      Push failure expected in Exp 5-A  --  neighbor node is down.")

        recv_t.join(timeout=130)
        comm_time = time.time() - t_comm
        received  = recv_results.get('received', [])
        recv_got  = len(received)

        print()
        log(f"      Exchange done  |  Total comm time: {comm_time:.3f}s"
            f"  |  Got {len(received)}/2 neighbor models")
        print()

        # ---- [3/3] Blend + Evaluate ----
        log_thin("[3/3]  BLEND + EVALUATE")

        bytes_tx_round = len(weights_bytes) * (int(left_ok) + int(right_ok))
        bytes_rx_round = sum(len(raw) for _, raw in received)

        if received:
            all_states = [model.state_dict()]
            for addr, raw in received:
                n_model = CNNCifar().to(device)
                n_model.load_state_dict(pickle.loads(raw))
                all_states.append(n_model.state_dict())
            model = CNNCifar().to(device)
            model.load_state_dict(fedavg(all_states))
            blends_total += len(received)
            log(f"      Blended {len(received)} neighbor model(s)"
                f"  |  Equal-weight avg ({len(all_states)} models)"
                f"  |  Total blends: {blends_total}")
        else:
            log(f"      No models received  --  keeping local model")

        acc_blended = evaluate(model, test_loader, device)
        delta = acc_blended - acc_post_train
        sign  = "+" if delta >= 0 else ""

        log(f"      Post-train accuracy  : {acc_post_train:.2f}%")
        log(f"      Post-blend accuracy  : {acc_blended:.2f}%")
        log(f"      Change               : {sign}{delta:.2f}%")
        print()

        bytes_tx_total += bytes_tx_round
        bytes_rx_total += bytes_rx_round

        results.append({
            'round':               round_num,
            'local_acc':           acc_post_train,
            'blend_acc':           acc_blended,
            'delta':               delta,
            'train_time':          train_time,
            'comm_time':           comm_time,
            'round_time':          train_time + comm_time,
            'bytes_pushed':        bytes_tx_round,
            'bytes_rx_round':      bytes_rx_round,
            'blends':              blends_total,
            'push_fail':           push_fail,
            'push_retries':        push_retries,
            'timeout_hits':        timeout_hits,
            'neighbors_received':  recv_got,
        })

        print(f"[COMM_SUMMARY] round={round_num} arch=ring node={node_id} "
              f"train_time={train_time:.2f}s comm_time={comm_time:.2f}s "
              f"round_time={train_time + comm_time:.2f}s "
              f"bytes_sent={bytes_tx_round} bytes_recv={bytes_rx_round} "
              f"push_retries={push_retries} timeout_hits={timeout_hits} "
              f"neighbors_received={recv_got} fanout=2 "
              f"pre_blend_acc={acc_post_train:.2f}% post_blend_acc={acc_blended:.2f}%")

        if round_num < num_rounds:
            log(f"Waiting 5s before next round...")
            time.sleep(5)

    log(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not results:
        return

    print()
    print(SEP)
    print(f"  DECENTRALIZED FL  --  NODE {node_id}  --  RESULTS")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Post-Train':>12}  {'Post-Blend':>12}  {'Change':>10}"
          f"  {'Train(s)':>10}  {'Comm(s)':>9}  {'Round(s)':>10}"
          f"  {'Pushed(KB)':>11}  {'Recvd(KB)':>10}  {'Blends':>7}"
          f"  {'Retries':>8}  {'T/O':>5}  {'Rcvd':>5}  {'Trend':>6}")
    print(f"  {THIN}")

    for r in results:
        sign  = "+" if r['delta'] >= 0 else ""
        arrow = "UP" if r['delta'] > 0.5 else ("DOWN" if r['delta'] < -0.5 else "FLAT")
        note  = "  [NEIGHBOR DOWN]" if r.get('push_fail') else ""
        print(f"  {r['round']:<7} {r['local_acc']:>11.2f}%  "
              f"{r['blend_acc']:>11.2f}%  "
              f"{sign}{r['delta']:>9.2f}%  "
              f"{r['train_time']:>9.3f}s  "
              f"{r['comm_time']:>8.3f}s  "
              f"{r['round_time']:>9.3f}s  "
              f"{r['bytes_pushed']/1024:>11.1f}  "
              f"{r['bytes_rx_round']/1024:>10.1f}  "
              f"{r['blends']:>7}  "
              f"{r.get('push_retries', 0):>8}  "
              f"{r.get('timeout_hits', 0):>5}  "
              f"{r.get('neighbors_received', '?'):>5}  "
              f"{arrow:>6}{note}")

    completed    = len(results)
    final_acc    = results[-1]['blend_acc']
    best_acc     = max(r['blend_acc']    for r in results)
    best_rnd     = max(results, key=lambda r: r['blend_acc'])['round']
    avg_train    = sum(r['train_time']   for r in results) / completed
    avg_comm     = sum(r['comm_time']    for r in results) / completed
    avg_round    = sum(r['round_time']   for r in results) / completed

    print()
    print(f"  Rounds completed   : {completed} / {num_rounds}")
    print(f"  Final accuracy     : {final_acc:.2f}%  (Round {completed})")
    print(f"  Best  accuracy     : {best_acc:.2f}%  (Round {best_rnd})")
    print(f"  Avg train / round  : {avg_train:.3f}s")
    print(f"  Avg comm  / round  : {avg_comm:.3f}s  (push + receive)")
    print(f"  Avg round duration : {avg_round:.3f}s")
    print(f"  Total pushed       : {bytes_tx_total/1024:.1f} KB  (sent to neighbors)")
    print(f"  Total received     : {bytes_rx_total/1024:.1f} KB  (blends received)")
    print(f"  Total comm         : {(bytes_tx_total + bytes_rx_total)/1024:.1f} KB")
    print(f"  Total blends       : {blends_total}")

    if fault_demo:
        failed = [r['round'] for r in results if r.get('push_fail')]
        print()
        print(f"  {THIN}")
        if node_id == 3:
            print(f"  This node (Node 3) exited at round {FAIL_ROUND} as planned.")
        elif failed:
            print(f"  Neighbor down first detected : Round {failed[0]}")
            print(f"  Rounds with push failure     : {failed}")
            print(f"  This node completed          : {completed} / {num_rounds} rounds")
            print(f"  RESULT: No SPOF. System continued without Node 3.")
        else:
            print(f"  No push failures detected (not adjacent to failed node).")
            print(f"  Completed all {completed} rounds unaffected.")

    print()
    print(SEP)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decentralized FL Node  (Experiments 3, 4, 5-A)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('node_id',  type=int,
                        help='Node ID  (0 to num_nodes-1, determines listen port: 8000 + node_id)')
    parser.add_argument('my_ip',
                        help='Private IP of this instance')
    parser.add_argument('left_ip',
                        help='Private IP of left ring neighbor')
    parser.add_argument('right_ip',
                        help='Private IP of right ring neighbor')
    parser.add_argument('--dist',      default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution  (default: iid)')
    parser.add_argument('--alpha',     type=float, default=0.5,
                        help='Dirichlet alpha for non_iid  (default: 0.5)')
    parser.add_argument('--rounds',    type=int, default=50,
                        help='Number of FL rounds  (default: 50)')
    parser.add_argument('--num-nodes', type=int, default=8,
                        help='Total number of nodes in the ring  (default: 8)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 5-A: Node 3 exits at round 10 to demonstrate no-SPOF')

    args = parser.parse_args()

    if args.fault_demo:
        exp_label = 'decentralized_fault'
    elif args.dist == 'non_iid':
        exp_label = 'decentralized_noniid'
    else:
        exp_label = 'decentralized_iid'
    log_path = setup_file_logging(exp_label, f'node_{args.node_id}')
    print(f"  Log file: {log_path}", flush=True)

    run_node(
        node_id          = args.node_id,
        my_ip            = args.my_ip,
        left_ip          = args.left_ip,
        right_ip         = args.right_ip,
        num_rounds       = args.rounds,
        local_epochs     = 5,
        batch_size       = 64,
        samples_per_node = 6250,
        distribution     = args.dist,
        alpha            = args.alpha,
        fault_demo       = args.fault_demo,
        num_nodes        = args.num_nodes,
    )
