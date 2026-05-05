"""
node.py  --  Decentralized Async FL Node  (Experiments 3, 4, 5-A)
             Ring topology, 8 nodes, gossip-based weight sharing

Usage:
    python3 node.py <node_id> <my_ip> <left_ip> <right_ip>
                   [--dist iid|non_iid] [--alpha F] [--rounds N] [--fault-demo]

Examples:
    python3 node.py 0 172.31.21.108 172.31.18.64 172.31.31.28              # Exp 3: IID
    python3 node.py 0 172.31.21.108 172.31.18.64 172.31.31.28 --dist non_iid  # Exp 4: Non-IID
    python3 node.py 0 172.31.21.108 172.31.18.64 172.31.31.28 --fault-demo    # Exp 5-A: fault demo
"""

import argparse
import copy
import pickle
import socket
import sys
import threading
import time

from shared.log   import log, log_thin, SEP, THIN, setup_file_logging
from shared.model import CNNCifar
from shared.data  import get_loaders, NUM_NODES
from shared.net   import send_data, recv_data, make_server_socket
from shared.train import train_local, evaluate, blend_models

BASE_PORT  = 8000    # node i listens on BASE_PORT + i
ALPHA      = 0.5     # gossip blend factor
FAIL_ROUND = 16      # Experiment 5-A: node 3 exits at this round


# ============================================================
# PUSH TO ONE NEIGHBOR  --  fire-and-forget thread target
# ============================================================
def push_to_neighbor(label, ip, port, weights_bytes, results_dict, timeout=30):
    t = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        send_data(s, weights_bytes)
        s.close()
        elapsed = time.time() - t
        log(f"      PUSH OK    -->  {label}  ({ip}:{port})"
            f"  |  {len(weights_bytes):,} bytes  |  {elapsed:.2f}s")
        results_dict[label] = 'ok'
    except Exception as e:
        elapsed = time.time() - t
        log(f"      PUSH FAIL  -->  {label}  ({ip}:{port})"
            f"  |  {elapsed:.2f}s  |  {e}")
        log(f"                      Neighbor may be down. Skipping.")
        results_dict[label] = 'fail'


# ============================================================
# BACKGROUND LISTENER  --  always-on blend receiver
# ============================================================
def listener_loop(node_id, my_ip, listen_port,
                   model_ref, model_lock, stats, stop_event):

    log(f"Listener starting on {my_ip}:{listen_port}...")
    srv = make_server_socket(my_ip, listen_port, backlog=10)
    srv.settimeout(1.0)
    log(f"Listener ready  --  accepting incoming model pushes on port {listen_port}")

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
            conn.settimeout(30)
            raw = recv_data(conn)
            conn.close()

            if raw:
                received_state = pickle.loads(raw)
                received_model = CNNCifar()
                received_model.load_state_dict(received_state)

                with model_lock:
                    model_ref[0] = blend_models(model_ref[0], received_model, ALPHA)
                    stats['blends'] += 1
                    stats['bytes']  += len(raw)
                    n = stats['blends']

                log(f"BLEND #{n}  <--  {addr[0]}"
                    f"  |  {len(raw):,} bytes  |  Total blends: {n}")

        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                log(f"WARNING: Listener error  --  {e}")

    srv.close()
    log("Listener shut down.")


# ============================================================
# MAIN NODE LOOP
# ============================================================
def run_node(node_id, my_ip, left_ip, right_ip,
             num_rounds, local_epochs, batch_size,
             samples_per_node, distribution, alpha, fault_demo):

    device      = __import__('torch').device('cpu')
    left_id     = (node_id - 1) % NUM_NODES
    right_id    = (node_id + 1) % NUM_NODES
    listen_port = BASE_PORT + node_id
    left_port   = BASE_PORT + left_id
    right_port  = BASE_PORT + right_id

    # ---- Header ----
    print()
    print(SEP)
    print("  DECENTRALIZED ASYNC FL  --  NODE")
    print(SEP)
    print()
    print(f"  Node ID      : {node_id}  (listen port: {listen_port})")
    print(f"  Left  (Node {left_id})  : {left_ip}:{left_port}")
    print(f"  Right (Node {right_id})  : {right_ip}:{right_port}")
    print(f"  Distribution : {distribution.upper()}  (alpha={alpha})")
    print(f"  Rounds       : {num_rounds}  |  Epochs: {local_epochs}"
          f"  |  Batch: {batch_size}  |  Samples: {samples_per_node}")
    print(f"  Blend factor : {ALPHA}  (equal weight gossip)")
    print(f"  Fault demo   : {'YES  --  Node 3 exits at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
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

    # ---- Initialize shared state ----
    model_ref  = [CNNCifar().to(device)]
    model_lock = threading.Lock()
    stop_event = threading.Event()
    stats      = {'blends': 0, 'bytes': 0}

    # ---- Start listener ----
    log_thin("Starting background listener thread...")
    listener_t = threading.Thread(
        target=listener_loop,
        args=(node_id, my_ip, listen_port, model_ref, model_lock, stats, stop_event),
        daemon=True
    )
    listener_t.start()

    # Staggered startup: node i waits i*3s so all listeners are ready
    stagger = node_id * 3
    log(f"Startup stagger: {stagger}s  (node {node_id} x 3s)"
        f"  --  ensuring all 8 listeners are up before first push")
    time.sleep(stagger)
    print()

    results = []

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
            log(f"All remaining 7 nodes will complete all {num_rounds} rounds.")
            log(f"This proves no single point of failure in decentralized design.")
            print()
            stop_event.set()
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

        with model_lock:
            local_copy = copy.deepcopy(model_ref[0])

        t_train    = time.time()
        local_copy = train_local(local_copy, train_loader, device, local_epochs)
        train_time = time.time() - t_train

        with model_lock:
            model_ref[0] = local_copy

        acc_post_train = evaluate(model_ref[0], test_loader, device)
        print()
        log(f"      Done  |  Time: {train_time:.1f}s  |  Accuracy: {acc_post_train:.2f}%")
        print()

        # ---- [2/3] Push to neighbors ----
        log_thin(f"[2/3]  ASYNC PUSH TO RING NEIGHBORS")
        log(f"      Left  (Node {left_id})  :  {left_ip}:{left_port}")
        log(f"      Right (Node {right_id})  :  {right_ip}:{right_port}")
        print()

        with model_lock:
            weights_bytes = pickle.dumps(model_ref[0].state_dict())

        log(f"      Model serialized: {len(weights_bytes):,} bytes  |  Dispatching...")
        print()

        push_results = {}
        t_push       = time.time()

        push_l = threading.Thread(target=push_to_neighbor,
                                   args=(f"Node-{left_id}", left_ip, left_port,
                                         weights_bytes, push_results))
        push_r = threading.Thread(target=push_to_neighbor,
                                   args=(f"Node-{right_id}", right_ip, right_port,
                                         weights_bytes, push_results))
        push_l.daemon = push_r.daemon = True
        push_l.start()
        push_r.start()
        push_l.join(timeout=35)
        push_r.join(timeout=35)
        push_time = time.time() - t_push

        left_ok   = push_results.get(f"Node-{left_id}")  == 'ok'
        right_ok  = push_results.get(f"Node-{right_id}") == 'ok'
        push_fail = not left_ok or not right_ok

        print()
        log(f"      Push done  |  Time: {push_time:.1f}s"
            f"  |  Left: {'OK' if left_ok else 'FAIL'}"
            f"  |  Right: {'OK' if right_ok else 'FAIL'}")

        if push_fail and fault_demo:
            log(f"      Push failure expected in Experiment 5-A  --  neighbor node is down.")
            log(f"      Continuing with reachable neighbors only.")
        print()

        # ---- [3/3] Evaluate ----
        log_thin("[3/3]  EVALUATION")
        log(f"      Evaluating current model (includes any blends received by listener)")
        print()

        with model_lock:
            acc_current = evaluate(model_ref[0], test_loader, device)
            blends_now  = stats['blends']
            bytes_rx    = stats['bytes']

        delta = acc_current - acc_post_train
        sign  = "+" if delta >= 0 else ""

        log(f"      Post-train accuracy  : {acc_post_train:.2f}%")
        log(f"      Post-blend accuracy  : {acc_current:.2f}%")
        log(f"      Change               : {sign}{delta:.2f}%")
        log(f"      Total blends so far  : {blends_now}  ({bytes_rx:,} bytes received)")
        print()

        results.append({
            'round':      round_num,
            'local_acc':  acc_post_train,
            'blend_acc':  acc_current,
            'delta':      delta,
            'push_time':  push_time,
            'blends':     blends_now,
            'push_fail':  push_fail,
        })

    # ---- Shutdown listener ----
    print()
    log("All rounds complete. Stopping listener...")
    stop_event.set()
    listener_t.join(timeout=5)
    log(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Final table ----
    if not results:
        return

    print()
    print(SEP)
    print(f"  DECENTRALIZED FL  --  NODE {node_id}  --  RESULTS")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Post-Train':>12}  {'Post-Blend':>12}"
          f"  {'Change':>10}  {'Push(s)':>8}  {'Blends':>7}  {'Trend':>6}")
    print(f"  {THIN}")

    for r in results:
        sign  = "+" if r['delta'] >= 0 else ""
        arrow = "UP" if r['delta'] > 0.5 else ("DOWN" if r['delta'] < -0.5 else "FLAT")
        note  = "  [NEIGHBOR DOWN]" if r.get('push_fail') else ""
        print(f"  {r['round']:<7} {r['local_acc']:>11.2f}%  "
              f"{r['blend_acc']:>11.2f}%  "
              f"{sign}{r['delta']:>9.2f}%  "
              f"{r['push_time']:>7.1f}s  "
              f"{r['blends']:>7}  {arrow:>6}{note}")

    completed = len(results)
    final_acc = results[-1]['blend_acc']
    best_acc  = max(r['blend_acc'] for r in results)
    best_rnd  = max(results, key=lambda r: r['blend_acc'])['round']

    print()
    print(f"  Rounds completed  : {completed} / {num_rounds}")
    print(f"  Final accuracy    : {final_acc:.2f}%  (Round {completed})")
    print(f"  Best  accuracy    : {best_acc:.2f}%  (Round {best_rnd})")
    print(f"  Total blends rx   : {stats['blends']}  ({stats['bytes']:,} bytes)")

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
        description='Decentralized Async FL Node  (Experiments 3, 4, 5-A)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('node_id',  type=int,
                        help='Node ID  (0-7, determines listen port: 8000 + node_id)')
    parser.add_argument('my_ip',
                        help='Private IP of this instance')
    parser.add_argument('left_ip',
                        help='Private IP of left ring neighbor')
    parser.add_argument('right_ip',
                        help='Private IP of right ring neighbor')
    parser.add_argument('--dist',   default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution  (default: iid)')
    parser.add_argument('--alpha',  type=float, default=0.5,
                        help='Dirichlet alpha for non_iid  (default: 0.5)')
    parser.add_argument('--rounds', type=int, default=30,
                        help='Number of FL rounds  (default: 30)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 5-A: Node 3 exits at round 16 to demonstrate no-SPOF')

    args = parser.parse_args()

    if args.fault_demo:
        exp_label = 'exp5a_decentralized_noniid_fault'
    elif args.dist == 'non_iid':
        exp_label = 'exp4_decentralized_noniid'
    else:
        exp_label = 'exp3_decentralized_iid'
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
        samples_per_node = 2500,
        distribution     = args.dist,
        alpha            = args.alpha,
        fault_demo       = args.fault_demo
    )
