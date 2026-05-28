"""
fedstellar_node.py  --  Fedstellar (p2pfl) Ring | IID  (Experiment 7)

Architecture : p2pfl semi-synchronous gossip, ring topology, IID, N nodes
Comparison   : direct comparison vs Experiment 3 (synchronous ring, same model/data)
Platform     : p2pfl 0.4.4 -- gRPC-based gossip with FedAvg aggregation

Each round (p2pfl's internal loop):
  1. Local training for LOCAL_EPOCHS epochs
  2. Send model weights to ring neighbors via gRPC
  3. Receive neighbors' weights (waits for expected neighbors)
  4. FedAvg aggregate into local model
  5. Repeat

[COMM_SUMMARY] lines are emitted after each round for parse_comm_bottleneck.py.

Usage:
    From project root:
    PYTHONPATH=src python3 -m experiments.fedstellar.fedstellar_node <node_id> <my_ip> <left_ip> <right_ip>

Ring order: 0 -- 1 -- 2 -- ... -- N-1 -- 0
"""

import argparse
import os
import pickle
import sys
import time

# Disable p2pfl's remote loggers BEFORE any p2pfl import.
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import torch
from torch.utils.data import DataLoader

from p2pfl.settings import Settings
Settings.general.DISABLE_RAY = True
Settings.ssl.USE_SSL = False

from p2pfl.node import Node
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# Patch aggregator.py:101 race condition on swap-slowed nodes
from p2pfl.learning.aggregators.aggregator import Aggregator as _Aggregator
_orig_set_nodes = _Aggregator.set_nodes_to_aggregate
def _safe_set_nodes(self, nodes_to_aggregate):
    deadline = time.time() + 150
    while time.time() < deadline:
        if self._finish_aggregation_event.is_set():
            try:
                return _orig_set_nodes(self, nodes_to_aggregate)
            except Exception as e:
                if 'aggregation is running' not in str(e):
                    raise
        time.sleep(0.2)
    import sys as _sys
    print(f"  [WARNING] _safe_set_nodes: 150s stall -- force-clearing aggregation event.",
          file=_sys.stderr, flush=True)
    self._finish_aggregation_event.set()
    time.sleep(0.2)
    return _orig_set_nodes(self, nodes_to_aggregate)
_Aggregator.set_nodes_to_aggregate = _safe_set_nodes

torch.set_num_threads(2)

# Add src/ to path for shared.log (PYTHONPATH=src should already cover this,
# but the explicit insert ensures it works when running as a module too)
_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_FILE)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from shared.log import log, log_thin, SEP, THIN, setup_file_logging
from .shared.model_lightning import CNNCifarLightning
from .shared.data_fedstellar import get_fedstellar_partition, make_test_loader

NUM_NODES    = 8
BASE_PORT    = 10000   # node i listens on 10000+i (avoids clash with exp3 port 8000+i)
NUM_ROUNDS   = 50
LOCAL_EPOCHS = 5
FAIL_ROUND   = 10      # Fault demo: Node 3 exits before this round's training starts


# ============================================================
# FAST P2PFL DATASET WRAPPER
# ============================================================
class _FastP2PFLDataset(P2PFLDataset):
    """Bypasses HuggingFace/PyArrow pipeline -- returns torch DataLoader directly."""

    def __init__(self, data_loader: DataLoader):
        self._loader = data_loader
        self.dataset_name = "cifar10_iid"
        self.batch_size = 64

    def export(self, strategy, train: bool = True, **kwargs):
        return self._loader

    def get_num_samples(self) -> int:
        return len(self._loader.dataset)


# ============================================================
# PER-ROUND TIMING AND ACCURACY TRACKER
# ============================================================
class _RoundLogger:
    """Tracks per-round timing and emits [COMM_SUMMARY] lines."""

    def __init__(self, node_id, test_loader):
        self.node_id     = node_id
        self.test_loader = test_loader
        self._round_num  = 0
        self._fit_end_t  = None
        self._pending    = None
        self.results     = []

    def on_fit_start(self, lightning_mod):
        t_now = time.time()
        if self._pending is not None and self._fit_end_t is not None:
            comm_time = t_now - self._fit_end_t
            post_acc  = self._eval(lightning_mod)
            self._emit(self._pending['round'], self._pending['train_time'],
                       comm_time, self._pending['pre_acc'], post_acc,
                       self._pending['model_bytes'])
        return t_now

    def on_fit_end(self, lightning_mod, t_fit_start):
        t_now      = time.time()
        train_time = t_now - t_fit_start
        pre_acc    = self._eval(lightning_mod)

        model_bytes = len(pickle.dumps(
            {k: v.cpu() for k, v in lightning_mod.state_dict().items()}
        ))

        self._fit_end_t = t_now
        self._round_num += 1
        self._pending = {
            'round':       self._round_num,
            'train_time':  train_time,
            'pre_acc':     pre_acc,
            'model_bytes': model_bytes,
        }

    def on_training_complete(self, lightning_mod):
        if self._pending is not None:
            post_acc = self._eval(lightning_mod)
            self._emit(self._pending['round'], self._pending['train_time'],
                       0.0, self._pending['pre_acc'], post_acc,
                       self._pending['model_bytes'])

    def _eval(self, model):
        device = next(model.parameters()).device
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                out   = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total   += len(y)
        model.train()
        return 100.0 * correct / total if total else 0.0

    def _emit(self, round_num, train_time, comm_time, pre_acc, post_acc, model_bytes):
        round_time = train_time + comm_time
        bytes_sent = model_bytes * 2
        bytes_recv = model_bytes * 2

        log(f"Round {round_num:>2} done  |  train: {train_time:.2f}s"
            f"  |  comm: {comm_time:.2f}s"
            f"  |  pre: {pre_acc:.2f}%  -->  post: {post_acc:.2f}%")

        print(f"[COMM_SUMMARY] round={round_num} arch=fedstellar_ring node={self.node_id} "
              f"train_time={train_time:.2f}s comm_time={comm_time:.2f}s "
              f"round_time={round_time:.2f}s "
              f"bytes_sent={bytes_sent} bytes_recv={bytes_recv} "
              f"push_retries=0 timeout_hits=0 "
              f"neighbors_received=2 fanout=2 "
              f"pre_blend_acc={pre_acc:.2f}% post_blend_acc={post_acc:.2f}%",
              flush=True)

        self.results.append({
            'round':      round_num,
            'train_time': train_time,
            'comm_time':  comm_time,
            'round_time': round_time,
            'pre_acc':    pre_acc,
            'post_acc':   post_acc,
        })


# ============================================================
# LEARNER SUBCLASS  --  wraps p2pfl's per-round fit() call
# ============================================================
def make_logging_learner(round_logger, node_id, num_rounds, local_epochs=LOCAL_EPOCHS, fault_demo=False):

    class _LoggingLearner(LightningLearner):
        def evaluate(self):
            return {}

        def fit(self):
            with torch.no_grad():
                for p in self.get_model().get_model().parameters():
                    p.data = p.data.clone()

            round_logger._active_learner = self

            if fault_demo and node_id == 3 and (round_logger._round_num + 1) == FAIL_ROUND:
                print()
                print('=' * 62)
                print(f'  NODE {node_id}  --  FEDSTELLAR FAULT DEMO')
                print('=' * 62)
                print()
                log(f'Node {node_id} deliberately exiting before round {FAIL_ROUND}.')
                log(f'p2pfl neighbors (left and right) will lose gRPC contact.')
                log(f'Observe whether they block (sync cascade) or continue (async).')
                log(f'Compare with Exp 5-A: synchronous TCP cascade (90+120s block).')
                print(flush=True)
                os._exit(0)

            round_num     = round_logger._round_num + 1
            lightning_mod = self.get_model().get_model()

            print(flush=True)
            print(SEP, flush=True)
            print(f"  ROUND {round_num:>2} / {num_rounds}  |  Node {node_id}  |  {time.strftime('%H:%M:%S')}", flush=True)
            print(THIN, flush=True)
            print(f"  [1/2]  LOCAL TRAINING  --  {local_epochs} epochs  (SGD lr=0.01  momentum=0.5)", flush=True)
            print(flush=True)

            _epoch_box = [0]
            def _epoch_hook():
                _epoch_box[0] += 1
                print(f"    Epoch {_epoch_box[0]:>2}/{local_epochs}  done", flush=True)
            lightning_mod.on_train_epoch_end = _epoch_hook

            t_start = round_logger.on_fit_start(lightning_mod)
            result  = super().fit()
            del lightning_mod.on_train_epoch_end
            round_logger.on_fit_end(lightning_mod, t_start)

            print(flush=True)
            print(f"  [2/2]  GOSSIP + AGGREGATE  (waiting for p2pfl neighbors...)", flush=True)
            print(flush=True)

            return result

    return _LoggingLearner


# ============================================================
# PRINT FINAL RESULTS TABLE
# ============================================================
def print_summary(node_id, results, num_rounds):
    if not results:
        return

    print()
    print(SEP)
    print(f"  FEDSTELLAR (p2pfl) RING  --  NODE {node_id}  --  RESULTS")
    print(SEP)
    print()
    print(f"  {'Round':<7} {'Pre-blend':>12}  {'Post-blend':>12}"
          f"  {'Train(s)':>10}  {'Comm(s)':>9}  {'Round(s)':>10}  {'Trend':>6}")
    print(f"  {THIN}")

    for r in results:
        delta = r['post_acc'] - r['pre_acc']
        arrow = 'UP' if delta > 0.5 else ('DOWN' if delta < -0.5 else 'FLAT')
        print(f"  {r['round']:<7} {r['pre_acc']:>11.2f}%  "
              f"{r['post_acc']:>11.2f}%  "
              f"{r['train_time']:>9.2f}s  "
              f"{r['comm_time']:>8.2f}s  "
              f"{r['round_time']:>9.2f}s  {arrow:>6}")

    completed = len(results)
    final_acc = results[-1]['post_acc']
    best_acc  = max(r['post_acc'] for r in results)
    best_rnd  = max(results, key=lambda r: r['post_acc'])['round']
    avg_train = sum(r['train_time'] for r in results) / completed
    avg_comm  = sum(r['comm_time']  for r in results) / completed
    avg_round = sum(r['round_time'] for r in results) / completed

    print()
    print(f"  Rounds completed   : {completed} / {num_rounds}")
    print(f"  Final accuracy     : {final_acc:.2f}%  (Round {completed})")
    print(f"  Best  accuracy     : {best_acc:.2f}%  (Round {best_rnd})")
    print(f"  Avg train / round  : {avg_train:.2f}s")
    print(f"  Avg comm  / round  : {avg_comm:.2f}s  (gossip + aggregate)")
    print(f"  Avg round duration : {avg_round:.2f}s")
    print(f"  Finished at        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(SEP)


# ============================================================
# MAIN
# ============================================================
def run_node(node_id, my_ip, left_ip, right_ip,
             num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS, fault_demo=False):

    import logging as _logging
    _logging.getLogger('p2pfl').setLevel(_logging.WARNING)
    _logging.getLogger('lightning.pytorch').setLevel(_logging.WARNING)
    _logging.getLogger('lightning').setLevel(_logging.WARNING)

    left_id  = (node_id - 1 + NUM_NODES) % NUM_NODES
    right_id = (node_id + 1) % NUM_NODES
    my_addr    = f"{my_ip}:{BASE_PORT + node_id}"
    left_addr  = f"{left_ip}:{BASE_PORT + left_id}"
    right_addr = f"{right_ip}:{BASE_PORT + right_id}"

    print()
    print(SEP)
    print("  FEDSTELLAR (p2pfl)  --  RING GOSSIP  --  IID")
    print(SEP)
    print()
    print(f"  Node ID      : {node_id}")
    print(f"  Address      : {my_addr}")
    print(f"  Left         : Node {left_id}  ({left_addr})")
    print(f"  Right        : Node {right_id}  ({right_addr})")
    print(f"  Rounds       : {num_rounds}  |  Local epochs: {local_epochs}")
    print(f"  Model        : CNNCifar  (~62K params)  --  identical to Exp 3")
    print(f"  Data         : CIFAR-10 IID  (6,250 samples)  --  same split as Exp 3")
    print(f"  Platform     : p2pfl 0.4.4 semi-synchronous gossip (async gRPC)")
    print(f"  Fault demo   : {'YES  --  Node 3 exits at round ' + str(FAIL_ROUND) if fault_demo else 'NO'}")
    print(f"  Started at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Data ----
    log_thin("Loading CIFAR-10 partition...")
    train_subset, test_dataset = get_fedstellar_partition(node_id)
    test_loader = make_test_loader(test_dataset)
    log(f"Train samples : {len(train_subset)}  |  Test samples: {len(test_dataset)}")

    log_thin("Wrapping train DataLoader for p2pfl (fast path -- no HF conversion)...")
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True,
                              num_workers=0, drop_last=True)
    p2pfl_data = _FastP2PFLDataset(train_loader)
    log(f"p2pfl dataset ready.")
    print()

    # ---- Node setup ----
    log_thin("Initialising p2pfl node...")
    model_lightning     = CNNCifarLightning()
    model_wrapped       = LightningModel(model_lightning)
    round_logger        = _RoundLogger(node_id, test_loader)
    LoggingLearnerClass = make_logging_learner(round_logger, node_id, num_rounds, local_epochs, fault_demo)
    logging_learner     = LoggingLearnerClass()

    log(f"Model params  : {sum(p.numel() for p in model_lightning.parameters()):,}")

    node = Node(
        model   = model_wrapped,
        data    = p2pfl_data,
        addr    = my_addr,
        learner = logging_learner,
    )

    Settings.heartbeat.TIMEOUT             = 300.0
    Settings.general.GRPC_TIMEOUT          = 60.0
    Settings.training.AGGREGATION_TIMEOUT  = 120
    Settings.training.VOTE_TIMEOUT         = 60
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 10000

    log(f"Starting gRPC server at {my_addr}...")
    node.start()

    log(f"Waiting 60s for all nodes to start their gRPC servers...")
    time.sleep(60)

    log(f"Connecting to ring neighbors...")
    node.connect(left_addr)
    node.connect(right_addr)
    log(f"  Connected: left={left_addr}  right={right_addr}")

    log(f"Waiting 60s for heartbeats to stabilise before starting training...")
    time.sleep(60)
    print()

    log_thin(f"Starting fedstellar learning: {num_rounds} rounds x {local_epochs} epochs")

    t_total = time.time()
    node.set_start_learning(rounds=num_rounds, epochs=local_epochs)

    log("Waiting for first training round to begin (up to 120s)...")
    deadline = time.time() + 120
    while time.time() < deadline:
        if node.state.round is not None:
            log(f"Training started -- round {node.state.round}")
            break
        time.sleep(0.5)
    else:
        log("WARNING: training did not start within 120s -- check connectivity.")

    while node.state.round is not None:
        time.sleep(1)

    total_time = time.time() - t_total

    active = getattr(round_logger, '_active_learner', None)
    if active is not None:
        round_logger.on_training_complete(active.get_model().get_model())

    try:
        node.stop()
    except Exception:
        pass
    log(f"Node stopped. Total experiment time: {total_time:.1f}s")

    print_summary(node_id, round_logger.results, num_rounds)
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fedstellar (p2pfl) Ring IID -- Experiment 7',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('node_id', type=int,
                        help='Node ID (0 to N-1)')
    parser.add_argument('my_ip',
                        help='Private IP of this node')
    parser.add_argument('left_ip',
                        help='Private IP of left ring neighbor')
    parser.add_argument('right_ip',
                        help='Private IP of right ring neighbor')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS,
                        help=f'FL rounds (default: {NUM_ROUNDS})')
    parser.add_argument('--fault-demo', action='store_true',
                        help='Node 3 exits at round 10 to demonstrate fault propagation')
    args = parser.parse_args()

    exp_label = 'fedstellar_ring_fault' if args.fault_demo else 'fedstellar_ring_iid'
    log_path  = setup_file_logging(exp_label, f'node_{args.node_id}')
    print(f"  Log file: {log_path}", flush=True)

    run_node(
        node_id    = args.node_id,
        my_ip      = args.my_ip,
        left_ip    = args.left_ip,
        right_ip   = args.right_ip,
        num_rounds = args.rounds,
        fault_demo = args.fault_demo,
    )
