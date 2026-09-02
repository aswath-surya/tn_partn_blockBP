"""BlockBP on an arbitrary (closed, scalar) tensor network.

Sources (refs/): Guo, Poletti & Arad 2301.05844v3 — Eq. (blockbp-messages)
    m_{A→B}^{(ℓ+1)} = Tr( T_A ∏_{A'∈N_A∖{B}} m_{A'→A}^{(ℓ)} ),
  messages are MPS of bond dimension χ_m over the cut bonds of the block edge,
  stop when the average distance between normalised consecutive messages < 1e-5
  (App. A: convergence measured as ε_ℓ/ε_1, max 10 iterations).
Kaufmann & Arad 2402.04834 §3.3–3.4 — normalise each outgoing message by making it
  left-canonical with unit L2 norm; distance Δ = (1/M Σ ‖|m⟩⟨m| − |m'⟩⟨m'|‖²)^{1/2}
  (gauge- and phase-invariant); damping m ← (1−η) m_new + η m_old; two thresholds
  δ0 (stop) < δ1 (trust); Bethe estimate of Tr(T) = ∏_A Tr(T_A ∏ m̂) with pair
  normalisation Tr(m̂_{A→B} m̂_{B→A}) = 1, i.e. log Z ≈ Σ_A log Z_A − Σ_{AB} log Z_AB.
Hack 2411.04957 §Results — BlockBP is the tree-equivalent (TE) method: exact iff the
  block graph is a tree; scheduling (sequential vs flooded updates) changes the fixed
  point reached on loopy block graphs.

Generic-graph version: a block edge (A,B) is the set of tensor-network indices shared
between the two blocks; the message MPS lives on those indices in a chosen linear
order (default: Fiedler ordering of the A-side attachment tensors — the cutwidth
heuristic; or a user key). Block contractions are exact (cotengra path) followed by
SVD compression to an MPS — optimal for a given χ_m, cost exponential in the
boundary length; the lattice-specialised boundary-MPS/zip-up route of the papers
is the production replacement for long boundaries.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import quimb.tensor as qtn


class BlockBP:
    def __init__(self, tn, block_of, chi_m=8, *, order="fiedler", order_key=None, damping=0.0,
                 tol=1e-5, tol_trust=1e-3, max_iter=30, schedule="parallel", init="random",
                 seed=0, cutoff=1e-12, dense_limit=2 ** 24, optimize="greedy"):
        """tn: closed quimb TensorNetwork (contracts to a scalar).
        block_of: callable(tensor) -> hashable block id, or dict tag -> block id
        (each tensor must carry exactly one of those tags)."""
        self.tn = tn
        self.chi_m, self.damping, self.tol, self.tol_trust = chi_m, damping, tol, tol_trust
        self.max_iter, self.schedule, self.init, self.cutoff = max_iter, schedule, init, cutoff
        self.dense_limit, self.optimize = dense_limit, optimize
        self.rng = np.random.default_rng(seed)
        if isinstance(block_of, dict):
            tagmap = block_of
            def block_of(t):
                hits = [tagmap[g] for g in t.tags if g in tagmap]
                if len(hits) != 1:
                    raise ValueError(f"tensor {t.tags} matches {len(hits)} block tags")
                return hits[0]
        # ---- tensors → blocks
        self.tid_block = {tid: block_of(t) for tid, t in tn.tensor_map.items()}
        self.blocks = sorted(set(self.tid_block.values()), key=str)
        self.T = {A: qtn.TensorNetwork([tn.tensor_map[tid].copy() for tid, b in self.tid_block.items() if b == A])
                  for A in self.blocks}
        # ---- block edges: indices shared by tensors in different blocks
        edges = defaultdict(list)
        for ind, tids in tn.ind_map.items():
            if len(tids) != 2:
                if len(tids) == 1:
                    raise ValueError(f"open index {ind}: the network must be closed")
                raise ValueError(f"hyper-index {ind} (shared by {len(tids)} tensors) not supported")
            a, b = (self.tid_block[t] for t in tids)
            if a != b:
                edges[frozenset((a, b))].append(ind)
        self.edges = dict(edges)
        self.nbrs = defaultdict(set)
        for e in self.edges:
            a, b = tuple(e)
            self.nbrs[a].add(b)
            self.nbrs[b].add(a)
        self.is_tree = len(self.edges) == len(self.blocks) - 1
        # ---- boundary orderings
        self.order = {}
        for e, inds in self.edges.items():
            if order_key is not None:
                self.order[e] = sorted(inds, key=order_key)
            elif order == "fiedler" and len(inds) > 2:
                self.order[e] = self._fiedler_order(e, inds)
            else:
                self.order[e] = sorted(inds)
        # ---- messages
        self.m = {}
        for e in self.edges:
            a, b = tuple(e)
            for (src, dst) in ((a, b), (b, a)):
                self.m[(src, dst)] = self._init_message(e)
        self.history = []

    # ------------------------------------------------------------ helpers
    def _dims(self, inds):
        return [self.tn.ind_size(i) for i in inds]

    def _fiedler_order(self, e, inds):
        """Order boundary bonds by the Fiedler coordinate of the tensor they attach
        to inside the smaller-id block (a min-cutwidth heuristic)."""
        import networkx as nx
        A = min(tuple(e), key=str)
        TA = self.T[A]
        G = nx.Graph()
        tids = list(TA.tensor_map)
        G.add_nodes_from(tids)
        for ind, ts in TA.ind_map.items():
            if len(ts) == 2:
                u, v = tuple(ts)
                G.add_edge(u, v)
        if G.number_of_nodes() < 3 or not nx.is_connected(G):
            return sorted(inds)
        try:
            fv = nx.fiedler_vector(G, normalized=True, seed=0)
        except Exception:
            return sorted(inds)
        coord = dict(zip(G.nodes, fv))
        def key(ind):
            tid = next(iter(TA.ind_map[ind]))
            return (coord[tid], ind)
        return sorted(inds, key=key)

    def _to_mps(self, t, inds):
        """Dense tensor over `inds` → left-canonical unit-norm MPS with bond ≤ χ_m."""
        dims = self._dims(inds)
        psi = np.asarray(t.transpose(*inds).data).reshape(-1)
        mps = qtn.MatrixProductState.from_dense(psi, dims=dims, max_bond=self.chi_m, cutoff=self.cutoff)
        mps.reindex_({mps.site_ind(i): ind for i, ind in enumerate(inds)})
        return self._normalize(mps)

    def _normalize(self, mps):
        mps.left_canonize()
        nrm = np.sqrt(abs(complex((mps.H & mps).contract())))
        last = mps[mps.L - 1]
        last.modify(data=last.data / nrm)
        return mps

    def _init_message(self, e):
        inds = self.order[e]
        dims = self._dims(inds)
        n = int(np.prod(dims))
        if self.init == "random":
            psi = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        else:                                            # "uniform": product of ones
            psi = np.ones(n, dtype=complex)
        t = qtn.Tensor(psi.reshape(dims), inds=inds)
        return self._to_mps(t, inds)

    def _overlap(self, m1, m2):
        """Sesquilinear ⟨m1|m2⟩ (both live on the same indices)."""
        return complex((m1.H & m2).contract())

    def _pairing(self, m1, m2):
        """Bilinear Tr(m1 m2) — the Bethe pair normalisation of the lemma."""
        return complex((m1 & m2).contract())

    # ------------------------------------------------------------ core update
    def _block_with_messages(self, A, exclude=None):
        tnA = self.T[A].copy()
        for B in self.nbrs[A]:
            if B == exclude:
                continue
            tnA &= self.m[(B, A)]
        return tnA

    def compute_message(self, A, B):
        """m_{A→B} = Tr(T_A ∏_{A'≠B} m_{A'→A}) as an MPS over the (A,B) boundary."""
        e = frozenset((A, B))
        inds = self.order[e]
        if int(np.prod(self._dims(inds))) > self.dense_limit:
            raise RuntimeError(f"boundary {A}-{B} too large for the dense route ({len(inds)} bonds)")
        t = self._block_with_messages(A, exclude=B).contract(output_inds=inds, optimize=self.optimize)
        return self._to_mps(t, inds)

    def iterate(self):
        """One sweep over all directed block edges; returns Δ."""
        keys = sorted(self.m, key=str)
        old = dict(self.m)
        new = {}
        d2 = []
        self.last_edge_delta = {}
        for (A, B) in keys:
            m_new = self.compute_message(A, B)
            if self.damping > 0:
                mixed = (1 - self.damping) * m_new + self.damping * old[(A, B)]
                mixed.compress(max_bond=self.chi_m, cutoff=self.cutoff)
                m_new = self._normalize(mixed)
            ov = abs(self._overlap(old[(A, B)], m_new)) ** 2
            d2.append(2 * max(0.0, 1 - ov))              # ‖|m⟩⟨m| − |m'⟩⟨m'|‖_F² for unit vectors
            self.last_edge_delta[(A, B)] = float(np.sqrt(d2[-1]))
            if self.schedule == "sequential":
                self.m[(A, B)] = m_new                    # later updates in this sweep see it
            else:
                new[(A, B)] = m_new
        if self.schedule != "sequential":
            self.m.update(new)
        delta = float(np.sqrt(np.mean(d2)))
        self.history.append(delta)
        return delta

    def run(self, verbose=False):
        t0 = time.perf_counter()
        for it in range(1, self.max_iter + 1):
            delta = self.iterate()
            if verbose:
                print(f"  iter {it:3d}  Δ = {delta:.3e}", flush=True)
            if delta < self.tol:
                break
        return dict(iters=len(self.history), delta=self.history[-1], converged=self.history[-1] < self.tol,
                    trusted=self.history[-1] < self.tol_trust, is_tree=self.is_tree,
                    n_blocks=len(self.blocks), n_edges=len(self.edges),
                    max_boundary=max(len(v) for v in self.edges.values()), t=time.perf_counter() - t0)

    # ------------------------------------------------------------ inference
    def block_Z(self, A):
        return complex(self._block_with_messages(A).contract(optimize=self.optimize))

    def log_Z(self):
        """Bethe/TE estimate: log Z ≈ Σ_A log Tr(T_A ∏ m) − Σ_{(A,B)} log Tr(m_{A→B} m_{B→A}).
        Exact on a tree block graph (up to χ_m truncation)."""
        lz = 0j
        for A in self.blocks:
            lz += np.log(self.block_Z(A))
        for e in self.edges:
            a, b = tuple(e)
            lz -= np.log(self._pairing(self.m[(a, b)], self.m[(b, a)]))
        return lz

    def Z(self):
        return np.exp(self.log_Z())

    def environment(self, A):
        """TN of the converged incoming messages of block A (its approximate environment)."""
        env = qtn.TensorNetwork([])
        for B in self.nbrs[A]:
            env &= self.m[(B, A)]
        return env
