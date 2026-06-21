"""
server.py  --  Centralized FL Server  (Experiments 1, 2, 5-B)

Usage:
    python3 server.py <server_ip> [--clients N] [--rounds N] [--fault-demo]

Examples:
    python3 server.py 172.31.21.108                        # Exp 1 or 2
    python3 server.py 172.31.21.108 --fault-demo           # Exp 5-B (SPOF demo)
"""

import argparse
import pickle
import socket
import sys
import time

from shared.log   import log, log_thin, SEP, THIN, setup_file_logging
from shared.model import CNNCifar
from shared.net   import send_data, recv_data, make_server_socket
from shared.train import fedavg

FAIL_ROUND = 10

# IP → Node ID lookup for readable log lines
NODE_ID_MAP = {
    "172.31.21.108": 0,
    "172.31.31.28":  1,
    "172.31.24.251": 2,
    "172.31.26.122": 3,
    "172.31.24.136": 4,
    "172.31.22.247": 5,
    "172.31.20.96":  6,
    "172.31.18.64":  7,
}

def node_label(ip):
    nid = NODE_ID_MAP.get(ip, "?")
    return f"Node {nid} ({ip})"


# ============================================================
# RECEIVE PHASE  --  collect models from all clients
# ============================================================
def receive_models(server_ip, port, num_clients):
    log(f"Opening listener on {server_ip}:{port}")
    log(f"Waiting for {num_clients} clients to upload their trained models...")
    print()

    srv           = make_server_socket(server_ip, port, backlog=num_clients)
    srv.settimeout(240)
    client_states = []
    client_sizes  = []

    while len(client_states) < num_clients:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            log(f"ERROR: Timeout. Only received {len(client_states)}/{num_clients} models.")
            srv.close()
            return client_states, client_sizes, True   # timed_out=True

        try:
            conn.settimeout(120)
            raw = recv_data(conn)
            conn.close()
            if raw:
                client_states.append(pickle.loads(raw))
                client_sizes.append(len(raw))
                log(f"  Received from client {len(client_states)}/{num_clients}"
                    f"  |  {node_label(addr[0])}  |  {len(raw)/1024:.1f} KB")
        except Exception as e:
            log(f"  WARNING: {node_label(addr[0])} connection error ({e}). Retrying slot.")
            try:
                conn.close()
            except Exception:
                pass

    srv.close()
    return client_states, client_sizes, False   # timed_out=False


# ============================================================
# BROADCAST PHASE  --  send aggregated model to all clients
# ============================================================
def broadcast_model(server_ip, port, num_clients, agg_bytes):
    log(f"Opening listener on {server_ip}:{port} for client reconnections...")
    print()

    srv        = make_server_socket(server_ip, port, backlog=num_clients)
    srv.settimeout(120)
    sent_count = 0

    while sent_count < num_clients:
        try:
            conn, addr = srv.accept()
            send_data(conn, agg_bytes)
            conn.close()
            sent_count += 1
            log(f"  Sent to client {sent_count}/{num_clients}"
                f"  |  {node_label(addr[0])}  |  {len(agg_bytes)/1024:.1f} KB")
        except socket.timeout:
            log(f"ERROR: Timeout. Only sent to {sent_count}/{num_clients} clients.")
            break

    srv.close()
    return sent_count


# ============================================================
# MAIN SERVER LOOP
# ============================================================
def run_server(server_ip, num_clients, num_rounds, upload_port=9000, bcast_port=9001, fault_demo=False):

    print()
    print(SEP)
    print("  CENTRALIZED FL  --  SERVER")
    print(SEP)
    print()
    print(f"  Server IP    : {server_ip}  (upload:{upload_port}  broadcast:{bcast_port})")
    print(f"  Clients      : {num_clients}  (Nodes 1 through {num_clients})")
    print(f"  Rounds       : {num_rounds}")
    print(f"  Fault demo   : {'YES  --  Server exits at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
    print(f"  Started at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"  Note: If this server goes down, ALL {num_clients} clients halt.")
    print(f"        This is the Single Point of Failure of centralized FL.")
    print()

    round_results = []

    for round_num in range(1, num_rounds + 1):

        # ---- Experiment 5-B: Server deliberately exits ----
        if fault_demo and round_num == FAIL_ROUND:
            print()
            print(SEP)
            print("  EXPERIMENT 5-B  --  SERVER FAILURE  (SPOF DEMONSTRATION)")
            print(SEP)
            print()
            log(f"Server deliberately exiting at round {FAIL_ROUND}.")
            log(f"All {num_clients} clients are waiting to connect.")
            log(f"They will receive Connection Refused and cannot continue.")
            log(f"Entire FL system halts  --  Single Point of Failure confirmed.")
            print()
            sys.exit(0)

        # ---- Round header ----
        print()
        print(SEP)
        print(f"  ROUND {round_num:>2} / {num_rounds}"
              f"  --  SERVER"
              f"  --  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(SEP)
        print()

        # ---- [1/3] Receive ----
        log_thin("[1/3]  RECEIVING MODELS FROM ALL CLIENTS")
        t_recv                             = time.time()
        client_states, client_sizes, recv_timeout = receive_models(server_ip, upload_port, num_clients)
        recv_time                          = time.time() - t_recv
        clients_received                   = len(client_states) if client_states else 0

        if not client_states:
            log("Stopping experiment due to receive failure.")
            break

        total_recv = sum(client_sizes)
        print()
        log(f"All {num_clients} models received"
            f"  |  Total: {total_recv/1024:.1f} KB  |  Time: {recv_time:.3f}s")
        print()

        # ---- [2/3] FedAvg ----
        log_thin("[2/3]  FEDAVG AGGREGATION")
        t_agg     = time.time()
        avg_state = fedavg(client_states)
        agg_time  = time.time() - t_agg
        agg_bytes = pickle.dumps(avg_state)

        log(f"FedAvg complete"
            f"  |  {num_clients} models averaged"
            f"  |  Time: {agg_time:.3f}s"
            f"  |  Model size: {len(agg_bytes)/1024:.1f} KB")
        print()

        # ---- [3/3] Broadcast ----
        log_thin(f"[3/3]  BROADCASTING AGGREGATED MODEL")
        t_send     = time.time()
        sent_count = broadcast_model(server_ip, bcast_port, num_clients, agg_bytes)
        send_time  = time.time() - t_send

        total_sent = len(agg_bytes) * sent_count
        print()
        log(f"Broadcast complete"
            f"  |  Sent to {sent_count}/{num_clients} clients"
            f"  |  Total: {total_sent/1024:.1f} KB  |  Time: {send_time:.3f}s")

        round_results.append({
            'round':            round_num,
            'bytes_received':   total_recv,
            'bytes_sent':       total_sent,
            'recv_time':        recv_time,
            'send_time':        send_time,
            'clients_received': clients_received,
            'clients_sent':     sent_count,
            'recv_timeout':     recv_timeout,
        })

        print(f"[COMM_SUMMARY] round={round_num} arch=centralized node=server "
              f"train_time=0.00s comm_time={recv_time + send_time:.2f}s "
              f"round_time={recv_time + agg_time + send_time:.2f}s "
              f"bytes_sent={total_sent} bytes_recv={total_recv} "
              f"push_retries=0 timeout_hits={1 if recv_timeout else 0} "
              f"neighbors_received={clients_received} fanout={num_clients} "
              f"pre_blend_acc=N/A% post_blend_acc=N/A%")

        if round_num < num_rounds:
            print()
            log(f"Waiting 5s before next round...")
            time.sleep(5)

    # ---- Final summary ----
    if not round_results:
        return

    print()
    print(SEP)
    print("  SERVER  --  FINAL SUMMARY")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Recv (KB)':>12}  {'Sent (KB)':>12}"
          f"  {'Recv(s)':>10}  {'Send(s)':>10}  {'C.Recvd':>9}  {'C.Sent':>8}  {'T/O':>5}")
    print(f"  {THIN}")

    for r in round_results:
        to_flag = " *" if r.get('recv_timeout') else ""
        print(f"  {r['round']:<7} {r['bytes_received']/1024:>12.1f}  {r['bytes_sent']/1024:>12.1f}"
              f"  {r['recv_time']:>9.3f}s  {r['send_time']:>9.3f}s"
              f"  {r.get('clients_received', '?'):>9}  {r.get('clients_sent', '?'):>8}"
              f"  {r.get('timeout_hits', 0):>5}{to_flag}")

    total_rx   = sum(r['bytes_received'] for r in round_results)
    total_tx   = sum(r['bytes_sent']     for r in round_results)
    total_time = sum(r['recv_time'] + r['send_time'] for r in round_results)

    print()
    print(f"  Rounds completed       : {len(round_results)} / {num_rounds}")
    print(f"  Total received         : {total_rx/1024:.1f} KB")
    print(f"  Total sent             : {total_tx/1024:.1f} KB")
    print(f"  Total exchanged        : {(total_rx + total_tx)/1024:.1f} KB")
    print(f"  Total comm time        : {total_time:.3f}s")
    print(f"  Finished at            : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(SEP)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Centralized FL Server  (Experiments 1, 2, 5-B)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('server_ip',
                        help='Private IP of this instance (Node 0)')
    parser.add_argument('--clients', type=int, default=7,
                        help='Number of clients to wait for  (default: 7)')
    parser.add_argument('--rounds', type=int, default=50,
                        help='Number of FL rounds  (default: 50)')
    parser.add_argument('--dist', default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution of clients  (default: iid)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 5-B: server exits at round 10 to demonstrate SPOF')

    args = parser.parse_args()

    if args.fault_demo:
        exp_label = 'centralized_spof'
    elif args.dist == 'non_iid':
        exp_label = 'centralized_noniid'
    else:
        exp_label = 'centralized_iid'
    log_path  = setup_file_logging(exp_label, 'server')
    print(f"  Log file: {log_path}", flush=True)

    run_server(args.server_ip, args.clients, args.rounds, fault_demo=args.fault_demo)
