#!/usr/bin/env python3
"""Decision Curve Analysis (Vickers & Elkin, 2006).

Net benefit at threshold probability p_t:
    NB(p_t) = (TP / n) - (FP / n) * (p_t / (1 - p_t))

Reference strategies:
    treat-all:  NB(p_t) = prevalence - (1 - prevalence) * (p_t / (1 - p_t))
    treat-none: NB(p_t) = 0 for all p_t

A model is clinically useful at p_t if its NB curve sits above both
reference strategies at that threshold.
"""
from __future__ import annotations

import numpy as np


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, p_t: float) -> float:
    if not 0.0 < p_t < 1.0:
        raise ValueError(f"p_t must be in (0, 1); got {p_t}")
    pred = (y_prob >= p_t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    n = len(y_true)
    return tp / n - fp / n * (p_t / (1 - p_t))


def treat_all_net_benefit(y_true: np.ndarray, p_t: float) -> float:
    prev = float(y_true.mean())
    return prev - (1 - prev) * (p_t / (1 - p_t))


def dca_curve(y_true: np.ndarray, y_prob: np.ndarray,
              p_t_range: np.ndarray) -> dict:
    nb_model = np.array([net_benefit(y_true, y_prob, pt) for pt in p_t_range])
    nb_all = np.array([treat_all_net_benefit(y_true, pt) for pt in p_t_range])
    nb_none = np.zeros_like(p_t_range)
    return {"p_t": p_t_range, "model": nb_model, "treat_all": nb_all, "treat_none": nb_none}
