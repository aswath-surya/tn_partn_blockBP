"""2D nearest-neighbour circuit families on an Lx x Ly open square lattice.

A circuit is a list of *layers*; a layer is a list of *rounds*; a round is a
list of gates on pairwise-disjoint sites (an edge-colour class of the
interaction graph = the unit of parallel execution, brief §1.4(iv)).

Bond convention (matches quimb ``PEPS.gen_bond_coos``): ``((i,j),(i,j+1))``
horizontal, ``((i,j),(i+1,j))`` vertical, first coordinate lexicographically
smaller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Coord = tuple[int, int]
Bond = tuple[Coord, Coord]


def canon(a: Coord, b: Coord) -> Bond:
    return (a, b) if a < b else (b, a)


def grid_sites(Lx, Ly):
    return [(i, j) for i in range(Lx) for j in range(Ly)]


def grid_bonds(Lx, Ly):
    """All bonds in quimb order."""
    out = []
    for i in range(Lx):
        for j in range(Ly):
            if j + 1 < Ly:
                out.append(((i, j), (i, j + 1)))
            if i + 1 < Lx:
                out.append(((i, j), (i + 1, j)))
    return out


def square_rounds(Lx, Ly):
    """The 4 matchings of the open square lattice: H-even, H-odd, V-even, V-odd."""
    h_even = [((i, j), (i, j + 1)) for i in range(Lx) for j in range(0, Ly - 1, 2)]
    h_odd = [((i, j), (i, j + 1)) for i in range(Lx) for j in range(1, Ly - 1, 2)]
    v_even = [((i, j), (i + 1, j)) for i in range(0, Lx - 1, 2) for j in range(Ly)]
    v_odd = [((i, j), (i + 1, j)) for i in range(1, Lx - 1, 2) for j in range(Ly)]
    return [r for r in (h_even, h_odd, v_even, v_odd) if r]


@dataclass
class Gate:
    G: np.ndarray            # (2,2) or (4,4) unitary, quimb convention
    where: tuple             # (coord,) or (coord_a, coord_b)
    name: str = ""

    @property
    def arity(self):
        return len(self.where)

    @property
    def bond(self):
        return canon(*self.where) if len(self.where) == 2 else None


@dataclass
class Round:
    kind: str                # "1q" | "2q"
    gates: list


@dataclass
class Circuit:
    Lx: int
    Ly: int
    layers: list             # list[list[Round]]
    family: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_qubits(self):
        return self.Lx * self.Ly

    @property
    def depth(self):
        return len(self.layers)

    def rounds(self):
        """Yield (layer_index, round_index, Round)."""
        for li, layer in enumerate(self.layers):
            for ri, rnd in enumerate(layer):
                yield li, ri, rnd

    def n_2q(self):
        return sum(len(r.gates) for _, _, r in self.rounds() if r.kind == "2q")

    def bond_traffic(self, from_layer=0, from_round=0):
        """Remaining number of 2-qubit gates per bond from (layer, round) onwards."""
        t = {b: 0 for b in grid_bonds(self.Lx, self.Ly)}
        for li, ri, rnd in self.rounds():
            if (li, ri) < (from_layer, from_round) or rnd.kind != "2q":
                continue
            for g in rnd.gates:
                t[g.bond] += 1
        return t


# ---------------------------------------------------------------- gate factories

def haar_unitary(n, rng):
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def exp_x(theta):
    return np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * _X


def exp_zz(theta):
    zz = np.kron(_Z, _Z)
    return np.cos(theta) * np.eye(4) - 1j * np.sin(theta) * zz


# ---------------------------------------------------------------- families

def brickwork_2d(Lx, Ly, depth, seed=0, two_qubit="haar", single="haar"):
    """Random brickwork: per layer, random 1q on every site then the 4 matchings.

    two_qubit: 'haar' (fastest χ growth) | 'cz' | 'rzz' (random-angle exp(-iθZZ)).
    """
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(depth):
        rounds = []
        if single == "haar":
            rounds.append(Round("1q", [Gate(haar_unitary(2, rng), (s,), "u1") for s in grid_sites(Lx, Ly)]))
        for bonds in square_rounds(Lx, Ly):
            gates = []
            for a, b in bonds:
                if two_qubit == "haar":
                    G, nm = haar_unitary(4, rng), "u2"
                elif two_qubit == "cz":
                    G, nm = _CZ, "cz"
                elif two_qubit == "rzz":
                    G, nm = exp_zz(rng.uniform(0, np.pi / 2)), "rzz"
                else:
                    raise ValueError(two_qubit)
                gates.append(Gate(G, (a, b), nm))
            rounds.append(Round("2q", gates))
        layers.append(rounds)
    return Circuit(Lx, Ly, layers, "brickwork",
                   dict(depth=depth, seed=seed, two_qubit=two_qubit, single=single))


def trotter_tfim_2d(Lx, Ly, n_steps, dt=0.1, J=1.0, h=1.0, init="x", disorder=0.0, seed=0):
    """First-order Trotter of H = -Σ_e J_e ZZ - h Σ X from |+>^n (init='x') or |0>^n.

    ``disorder`` > 0 draws J_e = J·(1 + disorder·u_e), u_e ~ U(-1, 1) per bond:
    weak bonds build entanglement slowly, so with a finite truncation cutoff the
    bond dimensions χ_e become heterogeneous across the lattice — the regime in
    which placement is non-trivial (brief §2.4)."""
    rng = np.random.default_rng(seed)
    Je = {b: J * (1.0 + disorder * rng.uniform(-1, 1)) for b in grid_bonds(Lx, Ly)}
    layers = []
    if init == "x":
        layers.append([Round("1q", [Gate(_H, (s,), "h") for s in grid_sites(Lx, Ly)])])
    Ux = exp_x(h * dt)
    for _ in range(n_steps):
        rounds = [Round("1q", [Gate(Ux, (s,), "rx") for s in grid_sites(Lx, Ly)])]
        for bonds in square_rounds(Lx, Ly):
            rounds.append(Round("2q", [Gate(exp_zz(Je[canon(a, b)] * dt), (a, b), "rzz") for a, b in bonds]))
        layers.append(rounds)
    return Circuit(Lx, Ly, layers, "tfim",
                   dict(n_steps=n_steps, dt=dt, J=J, h=h, init=init, disorder=disorder, seed=seed))


def ghz_2d(Lx, Ly):
    """GHZ via H on (0,0), CNOT chain along row 0, then CNOTs down every column.

    χ = 2 everywhere on the spanning tree: a pure-overhead control (brief §7.5)."""
    layers = [[Round("1q", [Gate(_H, ((0, 0),), "h")])]]
    for j in range(Ly - 1):
        layers.append([Round("2q", [Gate(_CNOT, ((0, j), (0, j + 1)), "cx")])])
    for i in range(Lx - 1):
        layers.append([Round("2q", [Gate(_CNOT, ((i, j), (i + 1, j)), "cx") for j in range(Ly)])])
    return Circuit(Lx, Ly, layers, "ghz")


FAMILIES = {"brickwork": brickwork_2d, "tfim": trotter_tfim_2d, "ghz": ghz_2d}
