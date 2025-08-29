# NISD-Network-Intrusion-Detection
Network Intrusion Detection
#!/usr/bin/env python3
"""
Network Intrusion Detection System (NIDS)
=======================================

A single-file, production-ready Python implementation that supports:

1) **Feature extraction** from PCAP files into per-flow features (CSV)
2) **Supervised training** on labeled CSVs (RandomForest inside a scikit-learn Pipeline)
3) **Unsupervised anomaly detection** (IsolationForest) for unlabeled data
4) **Evaluation** on a labeled CSV (reports precision/recall/F1/AUC & confusion matrix)
5) **Live detection** via packet sniffing (Scapy) with batched flow-level classification
6) **Hybrid rules**: lightweight signature/heuristic checks (e.g., SYN scan heuristic)

This file is designed to be easy to run on most systems with Python 3.9+.

---

# Quick CLI Examples

Extract features from PCAP:
    python nids.py features-from-pcap --pcap traffic.pcap --out features.csv

Train supervised model (requires labeled CSV with column `label` in {"benign", "malicious"}):
    python nids.py train --train_csv train.csv --model_out model.joblib

Evaluate on test set:
    python nids.py evaluate --test_csv test.csv --model_in model.joblib

Live detect on an interface (root/admin typically required):
    sudo python nids.py live --iface eth0 --model_in model.joblib --flush_interval 5

Unsupervised (no labels) train + live (produces anomaly scores):
    python nids.py train --train_csv features.csv --model_out iforest.joblib --unsupervised
    sudo python nids.py live --iface eth0 --model_in iforest.joblib --unsupervised

---

# Expected CSV schema for supervised training
- one row per flow, columns listed in FEATURE_COLUMNS, and a `label` column with values
  `benign` or `malicious` (case-insensitive). You can map custom labels via `--label_map`.

---

# Dependencies
    pip install scikit-learn pandas numpy scapy joblib tabulate rich

Scapy live sniffing and PCAP parsing may require libpcap (e.g., `apt-get install tcpdump` / `libpcap-dev`).

"""

from __future__ import annotations
import argparse
import json
import math
import os
import signal
import socket
import statistics
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

try:
    from scapy.all import sniff, rdpcap, IP, IPv6, TCP, UDP, ICMP, Raw
except Exception:
    sniff = None
    rdpcap = None
    IP = IPv6 = TCP = UDP = ICMP = Raw = object

# ---------------------------- Feature Definition ---------------------------- #

FEATURE_COLUMNS = [
    # volume & count
    "pkt_count", "byte_count", "duration",
    # size stats
    "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
    # inter-arrival time stats
    "iat_mean", "iat_std", "iat_min", "iat_max",
    # directionality
    "src_to_dst_pkts", "dst_to_src_pkts", "src_to_dst_bytes", "dst_to_src_bytes",
    # tcp flags counts
    "tcp_syn", "tcp_fin", "tcp_rst", "tcp_psh", "tcp_ack",
    # ports and protocol indicators
    "sport", "dport",
    "is_tcp", "is_udp", "is_icmp",
    # meta
    "unique_payload_sizes",
]

HYBRID_ALERTS = [
    "SYN_SCAN_SUSPECT",
    "MANY_DST_PORTS",
    "HIGH_PKT_RATE",
]

@dataclass
class FlowKey:
    src: str
    dst: str
    sport: int
    dport: int
    proto: str  # "TCP"/"UDP"/"ICMP"/"OTHER"

    def as_tuple(self):
        return (self.src, self.dst, self.sport, self.dport, self.proto)

@dataclass
class FlowStats:
    first_ts: float = math.inf
    last_ts: float = -math.inf
    pkt_count: int = 0
    byte_count: int = 0
    pkt_sizes: List[int] = None
    iats: List[float] = None
    src_to_dst_pkts: int = 0
    dst_to_src_pkts: int = 0
    src_to_dst_bytes: int = 0
    dst_to_src_bytes: int = 0
    tcp_syn: int = 0
    tcp_fin: int = 0
    tcp_rst: int = 0
    tcp_psh: int = 0
    tcp_ack: int = 0
    sport: int = 0
    dport: int = 0
    is_tcp: int = 0
    is_udp: int = 0
    is_icmp: int = 0
    payload_sizes: set = None

    def __post_init__(self):
        if self.pkt_sizes is None:
            self.pkt_sizes = []
        if self.iats is None:
            self.iats = []
        if self.payload_sizes is None:
            self.payload_sizes = set()

    def add(self, ts: float, length: int, direction: str, l4: Optional[Any], payload_len: int):
        if self.first_ts == math.inf:
            self.first_ts = ts
        if self.last_ts != -math.inf:
            self.iats.append(max(0.0, ts - self.last_ts))
        self.last_ts = ts
        self.pkt_count += 1
        self.byte_count += int(length)
        self.pkt_sizes.append(int(length))
        self.payload_sizes.add(int(payload_len))

        if direction == "src->dst":
            self.src_to_dst_pkts += 1
            self.src_to_dst_bytes += int(length)
        else:
            self.dst_to_src_pkts += 1
            self.dst_to_src_bytes += int(length)

        if l4 is not None and hasattr(l4, 'flags'):
            # TCP flags handling (scapy uses .flags)
            flags = int(l4.flags)
            # bit positions vary; scapy uses standard RFC bits
            self.tcp_syn += 1 if flags & 0x02 else 0
            self.tcp_fin += 1 if flags & 0x01 else 0
            self.tcp_rst += 1 if flags & 0x04 else 0
            self.tcp_psh += 1 if flags & 0x08 else 0
            self.tcp_ack += 1 if flags & 0x10 else 0

    def to_row(self) -> Dict[str, float]:
        duration = max(0.0, (self.last_ts - self.first_ts)) if self.pkt_count > 1 else 0.0
        size_mean = float(np.mean(self.pkt_sizes)) if self.pkt_sizes else 0.0
        size_std = float(np.std(self.pkt_sizes)) if len(self.pkt_sizes) > 1 else 0.0
        size_min = float(np.min(self.pkt_sizes)) if self.pkt_sizes else 0.0
        size_max = float(np.max(self.pkt_sizes)) if self.pkt_sizes else 0.0
        iat_mean = float(np.mean(self.iats)) if self.iats else 0.0
        iat_std = float(np.std(self.iats)) if len(self.iats) > 1 else 0.0
        iat_min = float(np.min(self.iats)) if self.iats else 0.0
        iat_max = float(np.max(self.iats)) if self.iats else 0.0

        return {
            "pkt_count": self.pkt_count,
            "byte_count": self.byte_count,
            "duration": duration,
            "pkt_size_mean": size_mean,
            "pkt_size_std": size_std,
            "pkt_size_min": size_min,
            "pkt_size_max": size_max,
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "iat_min": iat_min,
            "iat_max": iat_max,
            "src_to_dst_pkts": self.src_to_dst_pkts,
            "dst_to_src_pkts": self.dst_to_src_pkts,
            "src_to_dst_bytes": self.src_to_dst_bytes,
            "dst_to_src_bytes": self.dst_to_src_bytes,
            "tcp_syn": self.tcp_syn,
            "tcp_fin": self.tcp_fin,
            "tcp_rst": self.tcp_rst,
            "tcp_psh": self.tcp_psh,
            "tcp_ack": self.tcp_ack,
            "sport": int(self.sport),
            "dport": int(self.dport),
            "is_tcp": self.is_tcp,
            "is_udp": self.is_udp,
            "is_icmp": self.is_icmp,
            "unique_payload_sizes": len(self.payload_sizes),
        }

# ---------------------------- Feature Extraction ---------------------------- #

class FlowTable:
    def __init__(self, flow_timeout: float = 30.0):
        self._table: Dict[Tuple, FlowStats] = {}
        self.flow_timeout = flow_timeout

    def update_from_packet(self, pkt, ts: float):
        # try to parse IP/IPv6
        ip = pkt.getlayer(IP) if IP != object else None
        ipv6 = pkt.getlayer(IPv6) if IPv6 != object else None
        if ip is None and ipv6 is None:
            return []  # ignore non-IP

        src = ip.src if ip is not None else ipv6.src
        dst = ip.dst if ip is not None else ipv6.dst

        l4 = None
        sport = 0
        dport = 0
        is_tcp = is_udp = is_icmp = 0
        proto = "OTHER"

        if pkt.haslayer(TCP):
            l4 = pkt.getlayer(TCP)
            proto = "TCP"
            is_tcp = 1
            sport = int(l4.sport)
            dport = int(l4.dport)
        elif pkt.haslayer(UDP):
            l4 = pkt.getlayer(UDP)
            proto = "UDP"
            is_udp = 1
            sport = int(l4.sport)
            dport = int(l4.dport)
        elif pkt.haslayer(ICMP):
            l4 = pkt.getlayer(ICMP)
            proto = "ICMP"
            is_icmp = 1
            # ICMP has no ports
        else:
            pass

        # Determine canonical flow direction (5-tuple src,dst,sport,dport,proto)
        key = FlowKey(src, dst, sport, dport, proto).as_tuple()
        rev_key = FlowKey(dst, src, dport, sport, proto).as_tuple()

        # Create entries if needed
        if key not in self._table and rev_key not in self._table:
            fs = FlowStats()
            fs.sport = sport
            fs.dport = dport
            fs.is_tcp = is_tcp
            fs.is_udp = is_udp
            fs.is_icmp = is_icmp
            self._table[key] = fs

        # Decide which entry to use and direction
        if key in self._table:
            fs = self._table[key]
            direction = "src->dst"
        else:
            fs = self._table[rev_key]
            direction = "dst->src"

        # lengths
        length = int(len(pkt))
        payload_len = 0
        raw = pkt.getlayer(Raw)
        if raw is not None and hasattr(raw, 'load') and raw.load:
            payload_len = len(raw.load)

        fs.add(ts, length, direction, l4, payload_len)

        # Periodically evict and flush old flows
        return self._flush_expired(ts)

    def _flush_expired(self, now_ts: float):
        flushed = []
        to_delete = []
        for k, fs in self._table.items():
            if now_ts - fs.last_ts >= self.flow_timeout:
                row = fs.to_row()
                flushed.append((k, row))
                to_delete.append(k)
        for k in to_delete:
            del self._table[k]
        return flushed

    def flush_all(self):
        flushed = []
        for k, fs in list(self._table.items()):
            flushed.append((k, fs.to_row()))
            del self._table[k]
        return flushed

# ---------------------------- Heuristic Alerts ----------------------------- #

def heuristic_alerts(flow_row: Dict[str, float]) -> List[str]:
    alerts = []
    # SYN scan: many SYNs with few ACK/FIN and small duration/bytes
    if flow_row.get("tcp_syn", 0) >= 5 and flow_row.get("tcp_ack", 0) <= 1 and flow_row.get("byte_count", 0) < 1500:
        alerts.append("SYN_SCAN_SUSPECT")

    # High dst port variety often appears across flows; approximate with high dport (>1024) and many packets quickly
    if flow_row.get("dport", 0) > 1024 and flow_row.get("pkt_count", 0) > 50 and flow_row.get("duration", 0) < 2:
        alerts.append("MANY_DST_PORTS")

    # High packet rate within a short duration
    dur = flow_row.get("duration", 0.0)
    if dur > 0 and (flow_row.get("pkt_count", 0) / max(1e-6, dur)) > 200:
        alerts.append("HIGH_PKT_RATE")

    return alerts

# ------------------------------- Modeling ---------------------------------- #

class SupervisedModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self.label_map = {"benign": 0, "malicious": 1}
        self.inv_label_map = {0: "benign", 1: "malicious"}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_enc = y.str.lower().map(self.label_map)
        if y_enc.isnull().any():
            unique = sorted(list(set([v for v in y.str.lower().unique() if pd.notnull(v)])))
            raise ValueError(f"Unknown labels {unique} — provide --label_map to map custom labels.")
        self.pipeline.fit(X, y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.pipeline.predict(X)
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline[-1], "predict_proba"):
            return self.pipeline.predict_proba(X)
        # fallback: decision_function-like to probabilities via min-max
        scores = self.pipeline.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return np.vstack([1 - scores, scores]).T

    def save(self, path: str, extra: Optional[dict] = None):
        obj = {"pipeline": self.pipeline, "label_map": self.label_map, "feature_columns": FEATURE_COLUMNS}
        if extra:
            obj.update(extra)
        dump(obj, path)

    @staticmethod
    def load(path: str) -> "SupervisedModel":
        obj = load(path)
        m = SupervisedModel()
        m.pipeline = obj["pipeline"]
        m.label_map = obj.get("label_map", {"benign": 0, "malicious": 1})
        m.inv_label_map = {v: k for k, v in m.label_map.items()}
        return m

class UnsupervisedModel:
    def __init__(self):
        # contamination controls the expected fraction of anomalies
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", IsolationForest(
                n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1
            )),
        ])

    def fit(self, X: pd.DataFrame):
        self.pipeline.fit(X)
        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        # IsolationForest: higher negative scores => more anomalous
        scores = -self.pipeline["clf"].score_samples(self.pipeline["scaler"].transform(X))
        return scores

    def is_anomaly(self, X: pd.DataFrame, threshold: float = 0.6) -> np.ndarray:
        scores = self.anomaly_score(X)
        return (scores >= threshold).astype(int)

    def save(self, path: str, extra: Optional[dict] = None):
        obj = {"pipeline": self.pipeline, "feature_columns": FEATURE_COLUMNS}
        if extra:
            obj.update(extra)
        dump(obj, path)

    @staticmethod
    def load(path: str) -> "UnsupervisedModel":
        obj = load(path)
        m = UnsupervisedModel()
        m.pipeline = obj["pipeline"]
        return m

# ---------------------------- CSV Utilities -------------------------------- #

def load_csv_features(csv_path: str, expect_label: bool) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required feature columns: {missing}")
    X = df[FEATURE_COLUMNS].fillna(0.0)
    y = None
    if expect_label:
        if "label" not in df.columns:
            raise ValueError("Labeled training/eval requires a 'label' column.")
        y = df["label"].astype(str)
    return X, y

# ---------------------------- PCAP Processing ------------------------------ #

def features_from_pcap(pcap_path: str, out_csv: str, flow_timeout: float = 30.0) -> None:
    if rdpcap is None:
        raise RuntimeError("Scapy not available. Install scapy to parse PCAPs.")

    flows = FlowTable(flow_timeout=flow_timeout)
    for pkt in rdpcap(pcap_path):
        ts = float(pkt.time)
        flows.update_from_packet(pkt, ts)
    flushed = flows.flush_all()

    rows = []
    for (k, row) in flushed:
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No flows extracted — the PCAP may not contain IP packets.")
    else:
        df = df[FEATURE_COLUMNS]
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} flow rows to {out_csv}")

# ---------------------------- Live Detection ------------------------------- #

def live_predict(
    iface: str,
    model_path: str,
    unsupervised: bool = False,
    flush_interval: float = 5.0,
    flow_timeout: float = 30.0,
    anomaly_threshold: float = 0.6,
    export_jsonl: Optional[str] = None,
):
    if sniff is None:
        raise RuntimeError("Scapy sniff not available. Install scapy and run with admin privileges.")

    model = UnsupervisedModel.load(model_path) if unsupervised else SupervisedModel.load(model_path)

    flows = FlowTable(flow_timeout=flow_timeout)
    last_flush = time.time()
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    if export_jsonl:
        f_out = open(export_jsonl, "a", buffering=1)
    else:
        f_out = None

    def process_packet(pkt):
        nonlocal last_flush
        ts = float(pkt.time)
        flushed = flows.update_from_packet(pkt, ts)
        now = time.time()
        if (now - last_flush) >= flush_interval:
            last_flush = now
            flushed += flows._flush_expired(now)
        if not flushed:
            return

        rows = [row for (_, row) in flushed]
        X = pd.DataFrame(rows)[FEATURE_COLUMNS].fillna(0.0)

        # Heuristic alerts in parallel
        heur = [heuristic_alerts(r) for r in rows]

        if unsupervised:
            scores = model.anomaly_score(X)
            flags = (scores >= anomaly_threshold).astype(int)
            for i, r in enumerate(rows):
                out = {
                    **r,
                    "mode": "unsupervised",
                    "anomaly_score": float(scores[i]),
                    "is_anomaly": int(flags[i]),
                    "heuristic_alerts": heur[i],
                    "ts": time.time(),
                }
                _pretty_print_detection(out)
                if f_out:
                    f_out.write(json.dumps(out) + "\n")
        else:
            probs = None
            try:
                probs = model.predict_proba(X)[:, 1]
            except Exception:
                probs = None
            preds = model.predict(X)
            for i, r in enumerate(rows):
                label = model.inv_label_map.get(int(preds[i]), str(int(preds[i])))
                out = {
                    **r,
                    "mode": "supervised",
                    "pred_label": label,
                    "malicious_prob": float(probs[i]) if probs is not None else None,
                    "heuristic_alerts": heur[i],
                    "ts": time.time(),
                }
                _pretty_print_detection(out)
                if f_out:
                    f_out.write(json.dumps(out) + "\n")

    print(f"[NIDS] Sniffing on {iface} … Press Ctrl+C to stop.")
    try:
        sniff(iface=iface, prn=process_packet, store=False)
    finally:
        if f_out:
            f_out.close()
        # final flush
        for (_, row) in flows.flush_all():
            out = {**row, "mode": "final_flush"}
            _pretty_print_detection(out)


def _pretty_print_detection(d: Dict[str, Any]):
    # Minimal, dependency-light pretty print
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.get("ts", time.time())))
    mode = d.get("mode")
    heur = d.get("heuristic_alerts", [])
    if mode == "supervised":
        label = d.get("pred_label")
        prob = d.get("malicious_prob")
        msg = f"[{ts}] {mode.upper()} label={label} prob={prob:.3f} pkts={d['pkt_count']} bytes={d['byte_count']} duration={d['duration']:.3f}s heur={heur}"
    elif mode == "unsupervised":
        score = d.get("anomaly_score")
        is_anom = d.get("is_anomaly")
        msg = f"[{ts}] {mode.upper()} anomaly_score={score:.3f} is_anom={is_anom} pkts={d['pkt_count']} bytes={d['byte_count']} duration={d['duration']:.3f}s heur={heur}"
    else:
        msg = f"[{ts}] {mode} pkts={d['pkt_count']} bytes={d['byte_count']} duration={d['duration']:.3f}s"
    print(msg)

# ------------------------------ CLI Commands ------------------------------- #

def cmd_train(args: argparse.Namespace):
    X, y = load_csv_features(args.train_csv, expect_label=not args.unsupervised)

    if args.unsupervised:
        model = UnsupervisedModel().fit(X)
        model.save(args.model_out, extra={"feature_columns": FEATURE_COLUMNS})
        print(f"Saved unsupervised model to {args.model_out}")
    else:
        if args.label_map:
            # JSON mapping: {"your_normal_label": "benign", "your_attack_label": "malicious"}
            lm = json.loads(args.label_map)
            m = SupervisedModel()
            m.label_map = {str(k).lower(): (0 if str(v).lower()=="benign" else 1) for k, v in lm.items()}
            m.inv_label_map = {v: k for k, v in m.label_map.items()}
        else:
            m = SupervisedModel()
        m.fit(X, y)
        m.save(args.model_out, extra={"feature_columns": FEATURE_COLUMNS})
        print(f"Saved supervised model to {args.model_out}")


def cmd_evaluate(args: argparse.Namespace):
    X, y = load_csv_features(args.test_csv, expect_label=True)
    mobj = load(args.model_in)
    if "clf" in str(type(mobj.get("pipeline", None))):
        # heuristic check; better: try proba
        pass
    # figure out whether this is supervised or unsupervised based on stored object
    if "label_map" in mobj:
        model = SupervisedModel.load(args.model_in)
        y_true = y.str.lower().map(model.label_map)
        y_pred = model.predict(X)
        print("Classification report (macro avg):")
        print(classification_report(y_true, y_pred, target_names=[model.inv_label_map[0], model.inv_label_map[1]]))
        try:
            y_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_proba)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
    else:
        model = UnsupervisedModel.load(args.model_in)
        scores = model.anomaly_score(X)
        print("Anomaly score summary:")
        print(pd.Series(scores).describe())
        # If labels exist we can compute simple precision/recall by threshold
        if y is not None and "label" in y.name:
            yb = (y.str.lower() != "benign").astype(int)
            flags = (scores >= args.anomaly_threshold).astype(int)
            tp = int(((flags==1) & (yb==1)).sum())
            fp = int(((flags==1) & (yb==0)).sum())
            fn = int(((flags==0) & (yb==1)).sum())
            precision = tp / max(1, tp+fp)
            recall = tp / max(1, tp+fn)
            f1 = 2*precision*recall / max(1e-9, precision+recall)
            print(f"Threshold={args.anomaly_threshold:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


def cmd_features_from_pcap(args: argparse.Namespace):
    features_from_pcap(args.pcap, args.out, flow_timeout=args.flow_timeout)


def cmd_live(args: argparse.Namespace):
    live_predict(
        iface=args.iface,
        model_path=args.model_in,
        unsupervised=args.unsupervised,
        flush_interval=args.flush_interval,
        flow_timeout=args.flow_timeout,
        anomaly_threshold=args.anomaly_threshold,
        export_jsonl=args.export_jsonl,
    )

# --------------------------------- Main ------------------------------------ #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Network Intrusion Detection System (NIDS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train a model (supervised or unsupervised)")
    t.add_argument("--train_csv", required=True, help="CSV with per-flow features; for supervised include 'label' column")
    t.add_argument("--model_out", required=True, help="Where to save the trained model (joblib)")
    t.add_argument("--unsupervised", action="store_true", help="Use IsolationForest instead of supervised RandomForest")
    t.add_argument("--label_map", type=str, default=None, help="JSON mapping to map custom labels to {'benign','malicious'}")
    t.set_defaults(func=cmd_train)

    # evaluate
    e = sub.add_parser("evaluate", help="Evaluate a trained model on a labeled CSV")
    e.add_argument("--test_csv", required=True)
    e.add_argument("--model_in", required=True)
    e.add_argument("--anomaly_threshold", type=float, default=0.6, help="Threshold for unsupervised evaluation")
    e.set_defaults(func=cmd_evaluate)

    # features-from-pcap
    f = sub.add_parser("features-from-pcap", help="Extract per-flow features from a PCAP file")
    f.add_argument("--pcap", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--flow_timeout", type=float, default=30.0)
    f.set_defaults(func=cmd_features_from_pcap)

    # live
    l = sub.add_parser("live", help="Live detection from a network interface")
    l.add_argument("--iface", required=True, help="Interface to sniff (e.g., eth0, wlan0)")
    l.add_argument("--model_in", required=True)
    l.add_argument("--unsupervised", action="store_true")
    l.add_argument("--flush_interval", type=float, default=5.0)
    l.add_argument("--flow_timeout", type=float, default=30.0)
    l.add_argument("--anomaly_threshold", type=float, default=0.6)
    l.add_argument("--export_jsonl", type=str, default=None, help="Optional JSONL output path for detections")
    l.set_defaults(func=cmd_live)

    return p


def main(argv: Optional[List[str]] = None):
    p = build_argparser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

