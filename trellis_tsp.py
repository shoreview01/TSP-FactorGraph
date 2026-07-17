r"""
trellis_tsp.py
==============

Exact TSP as a min-sum (Viterbi) pass on a hypercube trellis -- the Held-Karp
dynamic program, vectorised with torch so it runs on GPU (CUDA / Apple MPS) and
falls back to CPU automatically.  No edge pruning, no beam, no heuristic
fallback: every edge is available and the whole subset table is explored, so the
returned tour is ALWAYS optimal.  This is what lets it mesh with CAP without
losing optimality -- each cluster CAP produces is routed exactly.

Trellis view of Held-Karp:
    state  : (visited set S, last city j)   -- a vertex of the hypercube {0,1}^n
                                               tagged with the current endpoint
    stage  : |S|                            -- processed layer by layer
    g[S,j] : min length of a path from the depot visiting exactly S, ending at j
    edge   : g[S,j] = min_{i in S\{j}} g[S\{j}, i] + D[i,j]     (min-sum / tropical)
    close  : tour = min_j g[full, j] + D[j, depot]

Cost: O(2^n * n^2) time, O(2^n * n) memory.  Exact is exponential, so the table
bounds n; on GPU this is comfortable for the capacity-bounded cluster sizes that
CAP produces (n up to ~24 fits in a few GB).  Beyond MEM_LIMIT_NODES it raises
rather than silently approximating.
"""

from __future__ import annotations

import numpy as np
import torch

MEM_LIMIT_NODES = 26          # refuse 2^n*n tables larger than this (OOM guard)


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _popcount(masks, n):
    pc = torch.zeros_like(masks)
    tmp = masks.clone()
    for _ in range(n):
        pc += tmp & 1
        tmp >>= 1
    return pc


def trellis_tsp(D, start=0, device=None):
    """Shortest closed tour over all nodes of D, exact (Held-Karp min-sum trellis).

    Returns (route, length) with route = [start, ..., start].  Runs on GPU if one
    is available.  Raises ValueError if n exceeds MEM_LIMIT_NODES.
    """
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    if n <= 1:
        return [start], 0.0
    if n == 2:
        o = 1 - start
        return [start, o, start], float(D[start, o] + D[o, start])
    if n > MEM_LIMIT_NODES:
        raise ValueError(
            f"n={n} exceeds MEM_LIMIT_NODES={MEM_LIMIT_NODES}; the exact 2^n table "
            f"would not fit. Raise MEM_LIMIT_NODES only if the device has the memory.")

    dev = _pick_device(device)
    dtype = torch.float32 if dev.type == "mps" else torch.float64
    Dt = torch.as_tensor(D, dtype=dtype, device=dev)

    size = 1 << n
    INF = torch.inf
    g = torch.full((size, n), INF, dtype=dtype, device=dev)
    back = torch.full((size, n), -1, dtype=torch.long, device=dev)

    start_bit = 1 << start
    g[start_bit, start] = 0.0

    masks = torch.arange(size, dtype=torch.long, device=dev)
    pc = _popcount(masks, n)
    ar = torch.arange(n, dtype=torch.long, device=dev)
    has_start = (masks & start_bit) != 0

    for t in range(1, n):                                    # expand size-t -> size-(t+1)
        layer = masks[(pc == t) & has_start]
        if layer.numel() == 0:
            continue
        for k in range(n):                                   # candidate next city
            src = layer[(layer & (1 << k)) == 0]             # subsets missing k
            if src.numel() == 0:
                continue
            g_src = g[src]                                   # (M, n)
            in_mask = (((src.unsqueeze(1)) >> ar) & 1).bool()
            cand = g_src + Dt[:, k].unsqueeze(0)             # min-sum edge j -> k
            cand = torch.where(in_mask, cand, torch.full_like(cand, INF))
            best, arg = cand.min(dim=1)
            tgt = src | (1 << k)
            better = best < g[tgt, k]
            if better.any():
                tb = tgt[better]
                g[tb, k] = best[better]
                back[tb, k] = arg[better]

    full = size - 1
    close = g[full] + Dt[:, start]                            # return to depot
    close[start] = INF
    last = int(close.argmin().item())
    cost = float(close[last].item())

    route = []                                               # Viterbi back-track
    S, j = full, last
    while j != -1:
        route.append(j)
        pj = int(back[S, j].item())
        S ^= (1 << j)
        j = pj
    route.reverse()
    route.append(start)
    return route, cost
