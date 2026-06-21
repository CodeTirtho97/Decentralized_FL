"""
node_fc.py  --  Decentralized FL Node  (Experiments 6-A, 6-B, 6-C)
               Fully-Connected topology, 8 nodes, synchronous gossip weight sharing.

               Each node communicates with ALL other 7 nodes every round.
               Each round: Train → Push to 7 peers → Receive from 7 peers
                           → Blend (equal-weight avg over 8 models) → Evaluate

               Compare with node.py (ring) where each node only talks to 2 neighbors.
               Fully-connected is the theoretical upper bound for decentralized FL
               connectivity -- maximum information flow per round at maximum comm cost.

Usage:
    python3 node_fc.py <node_id> <ip_0> <ip_1> <ip_2> <ip_3> <ip_4> <ip_5> <ip_6> <ip_7>
                       [--dist iid|non_iid] [--alpha F] [--rounds N] [--fault-demo]

Examples:
    python3 node_fc.py 0 172.31.21.108 172.31.31.28 172.31.24.251 172.31.26.122 \\
                         172.31.24.136 172.31.22.247 172.31.20.96 172.31.18.64
    python3 node_fc.py 0 <ip_0..ip_7> --dist non_iid              # Exp 6-B: Non-IID
    python3 node_fc.py 0 <ip_0..ip_7> --dist non_iid --fault-demo # Exp 6-C: FC Fault Tolerance
"""

import argparse
import pickle
import socket
import sys
import threading
import time

from shared.log   import log, log_thin, SEP, THIN, setup_file_logging
from shared.model import CNNCifar
from shared.data  import get_loaders, NUM_NODES
from shared.net   import send_data, recv_data, make_server_socket
from shared.train import train_local, evaluate, fedavg

BASE_PORT  = 8000    # node i listens on BASE_PORT + i  (same scheme as node.py)
FAIL_ROUND = 10      # Experiment 7: node 3 exits at this round


# ============================================================
# PUSH TO ONE PEER  --  retries until timeout
# ============================================================
def push_to_peer(label, ip, port, weights_bytes, results_dict, timeout=90):
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
    log(f"      PUSH FAIL  -->  {label}  ({ip}:{port})  |  Peer may be down.")
    results_dict[label] = 'fail'
    results_dict[label + '_attempts'] = attempt


# ============================================================
# RECEIVE FROM ALL PEERS  --  per-round blocking listener
# ============================================================
def receive_from_peers(my_ip, listen_port, expected_count,
                       results_dict, ready_event, timeout=120):
    srv = make_server_socket(my_ip, listen_port, backlog=expected_count + 1)
    ready_event.set()   # socket is bound -- callers may now push
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
        log(f"      WARNING: Received {len(received)}/{expected_count} peer models "
            f"(peer down or timeout).")
    results_dict['received'] = received


# ============================================================
# MAIN NODE LOOP
# ============================================================
def run_node(node_id, all_ips,
             num_rounds, local_epochs, batch_size,
             samples_per_node, distribution, alpha, fault_demo):

    device      = __import__('torch').device('cpu')
    my_ip       = all_ips[node_id]
    listen_port = BASE_PORT + node_id

    # All peers: every node except self
    peers = [(j, all_ips[j], BASE_PORT + j)
             for j in range(NUM_NODES) if j != node_id]
    num_peers = len(peers)   # 7 for 8-node setup

    # ---- Header ----
    print()
    print(SEP)
    print("  DECENTRALIZED FL  --  FULLY-CONNECTED NODE")
    print(SEP)
    print()
    print(f"  Node ID      : {node_id}  (listen port: {listen_port})")
    print(f"  My IP        : {my_ip}")
    print(f"  Topology     : Fully Connected  (all {num_peers} peers per round)")
    for peer_id, peer_ip, peer_port in peers:
        print(f"  Peer Node {peer_id}  : {peer_ip}:{peer_port}")
    print(f"  Distribution : {distribution.upper()}  (alpha={alpha})")
    print(f"  Rounds       : {num_rounds}  |  Epochs: {local_epochs}"
          f"  |  Batch: {batch_size}  |  Samples: {samples_per_node}")
    print(f"  Gossip       : Synchronous  (train → push all → recv all → blend)")
    print(f"  Fault demo   : {'YES  --  Node 3 exits at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
    if fault_demo:
        print(f"  FC note      : ALL {num_peers} peers will detect the fault (vs only 2 in ring Exp 5-A)")
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

    model          = CNNCifar().to(device)
    blends_total   = 0
    bytes_tx_total = 0
    bytes_rx_total = 0
    results        = []

    for round_num in range(1, num_rounds + 1):

        # ---- Experiment 7: Node 3 deliberately exits ----
        if fault_demo and node_id == 3 and round_num == FAIL_ROUND:
            print()
            print(SEP)
            print("  EXPERIMENT 6-C  --  NODE 3 FAILURE  (FULLY CONNECTED)")
            print(SEP)
            print()
            log(f"Node 3 deliberately exiting at round {FAIL_ROUND}.")
            log(f"ALL {NUM_NODES - 1} remaining peers will detect this failure.")
            log(f"Each surviving node will push to Node 3 and hit the TCP timeout (90s).")
            log(f"Each surviving node will receive 6/7 peer models and blend with its own.")
            log(f"All {NUM_NODES - 1} surviving nodes will complete all {num_rounds} rounds.")
            log(f"Contrast with Exp 5-A (ring): only 2 adjacent nodes detected the failure.")
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

        # ---- [2/3] Exchange with ALL peers (push + receive simultaneously) ----
        log_thin(f"[2/3]  EXCHANGE WITH ALL PEERS  (Fully Connected: {num_peers} peers)")
        for peer_id, peer_ip, peer_port in peers:
            log(f"      Peer Node {peer_id}  :  {peer_ip}:{peer_port}")
        print()

        weights_bytes = pickle.dumps(model.state_dict())
        log(f"      Model serialized: {len(weights_bytes)/1024:.1f} KB")
        log(f"      Opening receive listener on port {listen_port}...")
        print()

        # Start receive listener first -- socket must be bound before peers push to us
        recv_results = {}
        socket_ready = threading.Event()
        recv_t = threading.Thread(
            target=receive_from_peers,
            args=(my_ip, listen_port, num_peers, recv_results, socket_ready),
            daemon=True
        )
        recv_t.start()
        socket_ready.wait(timeout=10)   # wait until our socket is bound

        # Push to all peers simultaneously -- one thread per peer
        push_results = {}
        t_comm       = time.time()
        push_threads = []
        for peer_id, peer_ip, peer_port in peers:
            t = threading.Thread(
                target=push_to_peer,
                args=(f"Node-{peer_id}", peer_ip, peer_port,
                      weights_bytes, push_results)
            )
            t.daemon = True
            t.start()
            push_threads.append(t)

        for t in push_threads:
            t.join(timeout=95)

        # Count successful pushes and aggregate retry metrics
        ok_count     = sum(1 for pid, _, _ in peers
                           if push_results.get(f"Node-{pid}") == 'ok')
        fail_count   = num_peers - ok_count
        push_retries = sum(
            push_results.get(f"Node-{pid}_attempts", 1) - 1
            for pid, _, _ in peers
        )
        timeout_hits = fail_count  # each failed push exhausted its deadline

        log(f"      Push done  |  {ok_count}/{num_peers} peers reached"
            + (f"  |  {fail_count} FAILED  (peer down -- TCP timeout)" if fail_count else "")
            + f"  |  retries={push_retries}  timeouts={timeout_hits}")
        if fail_count and fault_demo:
            log(f"      Push failure expected in Exp 6-C  --  Node 3 is down.")

        # Wait for receive to complete
        recv_t.join(timeout=130)
        comm_time = time.time() - t_comm
        received  = recv_results.get('received', [])
        recv_got  = len(received)

        print()
        log(f"      Exchange done  |  Total comm time: {comm_time:.3f}s"
            f"  |  Got {len(received)}/{num_peers} peer models")
        print()

        # ---- [3/3] Blend + Evaluate ----
        log_thin("[3/3]  BLEND + EVALUATE")

        bytes_tx_round = len(weights_bytes) * ok_count
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
            log(f"      Blended {len(received)} peer model(s)"
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

        any_push_fail = fail_count > 0
        results.append({
            'round':             round_num,
            'local_acc':         acc_post_train,
            'blend_acc':         acc_blended,
            'delta':             delta,
            'train_time':        train_time,
            'comm_time':         comm_time,
            'round_time':        train_time + comm_time,
            'bytes_pushed':      bytes_tx_round,
            'bytes_rx_round':    bytes_rx_round,
            'blends':            blends_total,
            'ok_count':          ok_count,
            'push_fail':         any_push_fail,
            'push_retries':      push_retries,
            'timeout_hits':      timeout_hits,
            'peers_received':    recv_got,
        })

        print(f"[COMM_SUMMARY] round={round_num} arch=fc node={node_id} "
              f"train_time={train_time:.2f}s comm_time={comm_time:.2f}s "
              f"round_time={train_time + comm_time:.2f}s "
              f"bytes_sent={bytes_tx_round} bytes_recv={bytes_rx_round} "
              f"push_retries={push_retries} timeout_hits={timeout_hits} "
              f"peers_received={recv_got} fanout={num_peers} "
              f"pre_blend_acc={acc_post_train:.2f}% post_blend_acc={acc_blended:.2f}%")

        if round_num < num_rounds:
            log(f"Waiting 5s before next round...")
            time.sleep(5)

    log(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Final table ----
    if not results:
        return

    print()
    print(SEP)
    print(f"  DECENTRALIZED FL (FULLY CONNECTED)  --  NODE {node_id}  --  RESULTS")
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
        note  = f"  [PUSH {num_peers - r['ok_count']} FAIL]" if r.get('push_fail') else ""
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
              f"{r.get('peers_received', '?'):>5}  "
              f"{arrow:>6}{note}")

    completed = len(results)
    final_acc = results[-1]['blend_acc']
    best_acc  = max(r['blend_acc']  for r in results)
    best_rnd  = max(results, key=lambda r: r['blend_acc'])['round']
    avg_train = sum(r['train_time'] for r in results) / completed
    avg_comm  = sum(r['comm_time']  for r in results) / completed
    avg_round = sum(r['round_time'] for r in results) / completed

    print()
    print(f"  Rounds completed   : {completed} / {num_rounds}")
    print(f"  Final accuracy     : {final_acc:.2f}%  (Round {completed})")
    print(f"  Best  accuracy     : {best_acc:.2f}%  (Round {best_rnd})")
    print(f"  Avg train / round  : {avg_train:.3f}s")
    print(f"  Avg comm  / round  : {avg_comm:.3f}s  (push all + receive all)")
    print(f"  Avg round duration : {avg_round:.3f}s")
    print(f"  Total pushed       : {bytes_tx_total/1024:.1f} KB  (sent to {num_peers} peers)")
    print(f"  Total received     : {bytes_rx_total/1024:.1f} KB  (blends received)")
    print(f"  Total comm         : {(bytes_tx_total + bytes_rx_total)/1024:.1f} KB")
    print(f"  Total blends       : {blends_total}")
    print(f"  Topology note      : Fully Connected  --  {num_peers}x comm vs ring (2 neighbors)")

    if fault_demo:
        failed_rounds = [r['round'] for r in results if r.get('push_fail')]
        print()
        print(f"  {THIN}")
        if node_id == 3:
            print(f"  This node (Node 3) exited at round {FAIL_ROUND} as planned.")
        elif failed_rounds:
            print(f"  Node 3 failure first detected : Round {failed_rounds[0]}")
            print(f"  Rounds with push failure      : {failed_rounds}")
            print(f"  This node completed           : {completed} / {num_rounds} rounds")
            print(f"  RESULT: No SPOF in FC topology.")
            print(f"  NOTE: ALL {NUM_NODES - 1} peers detected this failure (vs 2 in ring Exp 5-A).")
            print(f"        FC fault impact is global; ring fault impact is localized.")
        else:
            print(f"  No push failures detected.")
            print(f"  Completed all {completed} rounds unaffected.")

    print()
    print(SEP)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decentralized FL Node -- Fully Connected  (Experiments 6-A, 6-B, 6-C)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('node_id', type=int,
                        help='Node ID  (0-7, determines listen port: 8000 + node_id)')
    parser.add_argument('all_ips', nargs=8, metavar='IP',
                        help='IPs of ALL 8 nodes in order: ip_0 ip_1 ... ip_7')
    parser.add_argument('--dist',   default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution  (default: iid)')
    parser.add_argument('--alpha',  type=float, default=0.5,
                        help='Dirichlet alpha for non_iid  (default: 0.5)')
    parser.add_argument('--rounds', type=int, default=50,
                        help='Number of FL rounds  (default: 50)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 6-C: Node 3 exits at round 10 -- FC fault tolerance demo')

    args = parser.parse_args()

    if args.fault_demo:
        exp_label = 'decentralized_fc_fault'
    elif args.dist == 'non_iid':
        exp_label = 'decentralized_fc_noniid'
    else:
        exp_label = 'decentralized_fc_iid'
    log_path = setup_file_logging(exp_label, f'node_{args.node_id}')
    print(f"  Log file: {log_path}", flush=True)

    run_node(
        node_id          = args.node_id,
        all_ips          = args.all_ips,
        num_rounds       = args.rounds,
        local_epochs     = 5,
        batch_size       = 64,
        samples_per_node = 6250,
        distribution     = args.dist,
        alpha            = args.alpha,
        fault_demo       = args.fault_demo,
    )
