"""Compact recursive ID3 and C4.5 classification trees."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def entropy(y):
    _, counts = np.unique(np.asarray(y, dtype=str), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _split_info(groups):
    sizes = np.asarray([len(g) for g in groups if len(g)], dtype=float)
    if not len(sizes):
        return 0.0
    p = sizes / sizes.sum()
    return float(-(p * np.log2(p)).sum())


@dataclass
class Node:
    prediction: str
    samples: int
    counts: dict
    entropy: float
    feature: str | None = None
    kind: str = "leaf"
    threshold: float | None = None
    children: dict | None = None
    gain: float = 0.0
    score: float = 0.0


class DecisionTree:
    def __init__(self, algorithm="ID3", feature_kinds=None, max_depth=6,
                 min_samples_split=4, min_impurity_decrease=0.0):
        if algorithm not in {"ID3", "C4.5"}:
            raise ValueError("algorithm must be ID3 or C4.5")
        self.algorithm = algorithm
        self.feature_kinds = feature_kinds or {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.features_ = list(X.columns)
        self.root_ = self._grow(X.reset_index(drop=True), pd.Series(y).reset_index(drop=True), 0, set())
        return self

    def _grow(self, X, y, depth, used):
        labels = y.astype(str)
        vals, counts = np.unique(labels, return_counts=True)
        prediction = str(vals[np.argmax(counts)])
        node = Node(prediction, len(y), dict(zip(map(str, vals), map(int, counts))), entropy(labels))
        if node.entropy == 0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return node
        candidates = []
        for feature in self.features_:
            if self.algorithm == "ID3" and feature in used:
                continue
            col = X[feature]
            is_numeric = self.algorithm == "C4.5" and self.feature_kinds.get(feature) == "numeric"
            if is_numeric:
                raw = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                unique = np.unique(raw[np.isfinite(raw)])
                if len(unique) < 2:
                    continue
                thresholds = (unique[:-1] + unique[1:]) / 2
                # Bound cost on high-cardinality columns while preserving ordered coverage.
                if len(thresholds) > 128:
                    thresholds = thresholds[np.linspace(0, len(thresholds)-1, 128).astype(int)]
                for threshold in thresholds:
                    mask = raw <= threshold
                    groups = [np.flatnonzero(mask), np.flatnonzero(~mask)]
                    if min(map(len, groups)) == 0:
                        continue
                    gain = node.entropy - sum(len(ix)/len(y)*entropy(labels.iloc[ix]) for ix in groups)
                    si = _split_info(groups)
                    score = gain / si if si > 0 else 0.0
                    candidates.append((score if self.algorithm == "C4.5" else gain, gain, feature, "threshold", float(threshold), groups))
            else:
                values = col.astype(str).to_numpy()
                unique = np.unique(values)
                if len(unique) < 2:
                    continue
                groups = [np.flatnonzero(values == value) for value in unique]
                gain = node.entropy - sum(len(ix)/len(y)*entropy(labels.iloc[ix]) for ix in groups)
                si = _split_info(groups)
                score = gain / si if si > 0 else 0.0
                candidates.append((score if self.algorithm == "C4.5" else gain, gain, feature, "categorical", None, groups))
        if not candidates:
            return node
        # C4.5 considers gain ratios among splits whose information gain is
        # at least the average candidate gain, reducing tiny-gain artifacts.
        if self.algorithm == "C4.5":
            average_gain = sum(c[1] for c in candidates) / len(candidates)
            eligible = [c for c in candidates if c[1] >= average_gain - 1e-12]
            if not eligible:
                eligible = candidates
            best = max(eligible, key=lambda c: c[0])
        else:
            best = max(candidates, key=lambda c: c[0])
        score, gain, feature, kind, threshold, groups = best
        if gain < self.min_impurity_decrease or gain <= 1e-12:
            return node
        node.feature, node.kind, node.threshold = feature, kind, threshold
        node.gain, node.score, node.children = gain, score, {}
        next_used = used | {feature}
        for i, indices in enumerate(groups):
            key = ("≤" if i == 0 else ">") + f" {threshold:.6g}" if kind == "threshold" else str(X.iloc[indices[0]][feature])
            node.children[key] = self._grow(X.iloc[indices].reset_index(drop=True), labels.iloc[indices].reset_index(drop=True), depth+1, next_used)
        return node

    def _predict_one(self, row, node):
        while node.children:
            if node.kind == "threshold":
                key = ("≤" if float(row[node.feature]) <= node.threshold else ">") + f" {node.threshold:.6g}"
            else:
                key = str(row[node.feature])
            next_node = node.children.get(key)
            if next_node is None:
                return node.prediction
            node = next_node
            if node.feature is None:
                return node.prediction
        return node.prediction

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=getattr(self, 'features_', [f"feature_{i}" for i in range(X.shape[1])]))
        return np.asarray([self._predict_one(row, self.root_) for _, row in X.iterrows()], dtype=object)

    def to_dot(self, max_nodes=500):
        lines = ["digraph Tree {", 'graph [rankdir=TB];', 'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#94a3b8", fillcolor="#eff6ff"];', 'edge [fontname="Helvetica", fontsize=9, color="#64748b"];']
        counter = [0]
        def esc(value):
            return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        def visit(node):
            idx = counter[0]; counter[0] += 1
            if counter[0] > max_nodes:
                return None
            majority = max(node.counts, key=node.counts.get)
            label = f"{node.feature} {'≤ ' + format(node.threshold, '.5g') if node.kind == 'threshold' else ''}" if node.feature else f"Predict: {node.prediction}"
            label += f"\\nEntropy: {node.entropy:.3f} | Gain: {node.gain:.3f}\\nSamples: {node.samples} | Class: {majority}"
            lines.append(f'node{idx} [label="{esc(label)}"];')
            for edge, child in (node.children or {}).items():
                child_id = visit(child)
                if child_id is not None:
                    lines.append(f'node{idx} -> node{child_id} [label="{esc(edge)}"];')
            return idx
        visit(self.root_)
        lines.append("}")
        return "\n".join(lines)
