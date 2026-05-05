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

FAIL_ROUND = 16


# ============================================================
# RECEIVE PHASE  --  collect models from all clients
# ============================================================
def receive_models(server_ip, port, num_clients):
    log(f"Opening listener on {server_ip}:{port}")
    log(f"Waiting for {num_clients} clients to upload their trained models...")
    print()

    srv           = make_server_socket(server_ip, port, backlog=num_clients)
    srv.settimeout(180)
    client_states = []
    client_sizes  = []

    while len(client_states) < num_clients:
        try:
            conn, addr = srv.accept()
            conn.settimeout(120)
            raw = recv_data(conn)
            conn.close()
            if raw:
                client_states.append(pickle.loads(raw))
                client_sizes.append(len(raw))
                log(f"  Received from client {len(client_states)}/{num_clients}"
                    f"  |  {addr[0]}  |  {len(raw):,} bytes")
        except socket.timeout:
            log(f"ERROR: Timeout. Only received {len(client_states)}/{num_clients} models.")
            srv.close()
            return None, None

    srv.close()
    return client_states, client_sizes


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
                f"  |  {addr[0]}  |  {len(agg_bytes):,} bytes")
        except socket.timeout:
            log(f"ERROR: Timeout. Only sent to {sent_count}/{num_clients} clients.")
            break

    srv.close()
    return sent_count


# ============================================================
# MAIN SERVER LOOP
# ============================================================
def run_server(server_ip, num_clients, num_rounds, port=9000, fault_demo=False):

    print()
    print(SEP)
    print("  CENTRALIZED FL  --  SERVER")
    print(SEP)
    print()
    print(f"  Server IP    : {server_ip}:{port}")
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
        t_recv        = time.time()
        client_states, client_sizes = receive_models(server_ip, port, num_clients)
        recv_time     = time.time() - t_recv

        if client_states is None:
            log("Stopping experiment due to receive failure.")
            break

        total_recv = sum(client_sizes)
        print()
        log(f"All {num_clients} models received"
            f"  |  Total: {total_recv:,} bytes  |  Time: {recv_time:.1f}s")
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
            f"  |  Model size: {len(agg_bytes):,} bytes")
        print()

        # ---- [3/3] Broadcast ----
        log_thin(f"[3/3]  BROADCASTING AGGREGATED MODEL")
        t_send     = time.time()
        sent_count = broadcast_model(server_ip, port, num_clients, agg_bytes)
        send_time  = time.time() - t_send

        total_sent = len(agg_bytes) * sent_count
        print()
        log(f"Broadcast complete"
            f"  |  Sent to {sent_count}/{num_clients} clients"
            f"  |  Total: {total_sent:,} bytes  |  Time: {send_time:.1f}s")

        round_results.append({
            'round':          round_num,
            'bytes_received': total_recv,
            'bytes_sent':     total_sent,
            'recv_time':      recv_time,
            'send_time':      send_time,
        })

        if round_num < num_rounds:
            print()
            log(f"Waiting 15s before next round...")
            time.sleep(15)

    # ---- Final summary ----
    if not round_results:
        return

    print()
    print(SEP)
    print("  SERVER  --  FINAL SUMMARY")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Bytes Received':>16}  {'Bytes Sent':>14}"
          f"  {'Recv(s)':>8}  {'Send(s)':>8}")
    print(f"  {THIN}")

    for r in round_results:
        print(f"  {r['round']:<7} {r['bytes_received']:>16,}  {r['bytes_sent']:>14,}"
              f"  {r['recv_time']:>7.1f}s  {r['send_time']:>7.1f}s")

    total_rx   = sum(r['bytes_received'] for r in round_results)
    total_tx   = sum(r['bytes_sent']     for r in round_results)
    total_time = sum(r['recv_time'] + r['send_time'] for r in round_results)

    print()
    print(f"  Rounds completed       : {len(round_results)} / {num_rounds}")
    print(f"  Total bytes received   : {total_rx:,}")
    print(f"  Total bytes sent       : {total_tx:,}")
    print(f"  Total bytes exchanged  : {total_rx + total_tx:,}")
    print(f"  Total comm time        : {total_time:.1f}s")
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
    parser.add_argument('--rounds', type=int, default=30,
                        help='Number of FL rounds  (default: 30)')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Experiment 5-B: server exits at round 16 to demonstrate SPOF')

    args = parser.parse_args()

    exp_label = 'exp5b_centralized_spof' if args.fault_demo else 'centralized'
    log_path  = setup_file_logging(exp_label, 'server')
    print(f"  Log file: {log_path}", flush=True)

    run_server(args.server_ip, args.clients, args.rounds, fault_demo=args.fault_demo)
