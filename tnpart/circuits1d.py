"""Random non-brickwork 1+1D circuits as tensor networks, with spacetime-tile blocks.

Layer r: random 1-qubit gates on every qubit, then a random subset of disjoint
adjacent pairs (random order, each kept with probability p_gate) with Haar 2-qubit
gates — so the gate pattern is irregular, not a brickwork. quimb tags every gate
tensor with 'I{q}' (qubits) and 'ROUND_{r}'; initial-state tensors carry 'PSI0'.
"""
from __future__ import annotations

import numpy as np
import quimb as qu
import quimb.tensor as qtn


def random_circuit(n, depth, seed=0, p_gate=0.6):
    rng = np.random.default_rng(seed)
    circ = qtn.Circuit(n)
    for r in range(depth):
        for q in range(n):
            circ.apply_gate_raw(qu.rand_uni(2, seed=int(rng.integers(1 << 30))), (q,), gate_round=r)
        used = set()
        for q in rng.permutation(n - 1):
            if q in used or q + 1 in used or rng.random() > p_gate:
                continue
            circ.apply_gate_raw(qu.rand_uni(4, seed=int(rng.integers(1 << 30))), (int(q), int(q) + 1), gate_round=r)
            used.update((q, q + 1))
    return circ


def amplitude_tn(circ, bits=None):
    """Closed network for <bits|U|0…0>; the bra tensors carry 'BRA' and 'ROUND_{depth+1}'."""
    n = circ.N
    bits = bits or "0" * n
    tn = circ.psi.copy()
    for q in range(n):
        v = np.array([1, 0], dtype=complex) if bits[q] == "0" else np.array([0, 1], dtype=complex)
        tn &= qtn.Tensor(v, inds=(f"k{q}",), tags={"BRA", f"I{q}"})
    return tn


def norm_tn(circ):
    """Closed double-layer network <ψ|ψ> (exactly 1 for a unitary circuit)."""
    ket = circ.psi.copy()
    bra = ket.conj()
    for t in bra:
        t.add_tag("BRA")
    return ket & bra


def _qubits_rounds(t):
    qs = [int(g[1:]) for g in t.tags if g.startswith("I") and g[1:].isdigit()]
    rs = [int(g.split("_")[1]) for g in t.tags if g.startswith("ROUND_")]
    return qs, rs


def circuit_blocks(bq, bt, depth):
    """block id = (q_block, t_block): 2-qubit gates go with their lower qubit;
    PSI0 → round −1 (first t-block); BRA → round `depth` (last t-block)."""
    def block_of(t):
        qs, rs = _qubits_rounds(t)
        if "PSI0" in t.tags:
            r = -1
        elif "BRA" in t.tags:
            r = depth
        else:
            r = rs[0]
        return (min(qs) // bq, (r + 1) // bt)
    return block_of


def circuit_key(tn):
    """Boundary-bond ordering key: (qubit, round) of the two attached tensors."""
    pos = {}
    for tid, t in tn.tensor_map.items():
        qs, rs = _qubits_rounds(t)
        pos[tid] = (min(qs), rs[0] if rs else (-1 if "PSI0" in t.tags else 10 ** 6))
    return lambda ind: tuple(sorted(pos[tid] for tid in tn.ind_map[ind]))


def block_grid(bq, bt, n, depth):
    """Number of (q, t) blocks for the amplitude network."""
    nq = -(-n // bq)
    nt = -(-(depth + 2) // bt)
    return nq, nt
