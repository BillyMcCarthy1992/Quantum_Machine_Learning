"""
The junk drawer. Everything the experiment scripts need lives in here so I
only had to get it right once.

Stuff you should know before poking at this:

- Parameters are shaped (M, 2, L, n) and axes are (M, L, n). A "block" is
  L layers of rotations+CZ, then the same thing mirrored and inverted. At
  the start we set the second half to minus the first half, which makes the
  whole circuit the identity. That's the entire trick from the Grant paper.
- The B layer out front (random rotations + CZ, never trained) is there
  because an identity circuit obviously can't entangle anything, and the
  Heisenberg ground state is very entangled, so without B you're stuck.
- Every bit of randomness comes from numpy SeedSequence children. One child
  per trial, then split again into a "training" stream and a "null" stream.
  That split matters: it means drawing more null directions doesn't shift
  the trajectory by a single bit. Learned that one the annoying way.
- Costs are (offset, scale, observable) triples so I can write
  cost = offset + scale * <obs> and not think about signs again:
    heisenberg -> (0, +1, H)                      just the energy
    global     -> (1, -1, |0..0><0..0|)           1 - fidelity
    local      -> (1/2, -1, mean of Z_i / 2)      1 - average <P0_i>
  The last two bottom out at the same state, which is the whole point of
  the locality experiment.
"""

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

_ROT = (qml.RX, qml.RY, qml.RZ)


# ----------------------------------------------------- building the circuit
def entangler_B(b_axes, b_params, n):
    for l in range(b_axes.shape[0]):
        for q in range(n):
            _ROT[b_axes[l, q]](b_params[l, q], wires=q)
        for q in range(n - 1):
            qml.CZ(wires=[q, q + 1])


def identity_blocks(params, axes, n, M, L):
    for m in range(M):
        for l in range(L):
            for q in range(n):
                _ROT[axes[m, l, q]](params[m, 0, l, q], wires=q)
            for q in range(n - 1):
                qml.CZ(wires=[q, q + 1])
        for l in reversed(range(L)):
            for q in range(n - 1):
                qml.CZ(wires=[q, q + 1])
            for q in range(n):
                _ROT[axes[m, l, q]](params[m, 1, l, q], wires=q)


def device(n):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return qml.device("default.qubit", wires=n)


# ----------------------------------------- the things we actually measure
def heisenberg(n, J=1.0, h=1.0):
    coeffs, ops = [], []
    for i in range(n):
        j = (i + 1) % n
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(J)
            ops.append(P(i) @ P(j))
    for i in range(n):
        coeffs.append(h)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(n, J=1.0, h=1.0):
    return float(np.min(np.linalg.eigvalsh(qml.matrix(heisenberg(n, J, h)))))


def cost_def(kind, n):
    """Returns (offset, scale, observable_factory(mode)). mode in
    {'native', 'hermitian'}; hermitian wraps the dense matrix (avoid n>10
    for 'global'/'heisenberg' hermitian mode: 2^n x 2^n dense)."""
    if kind == "heisenberg":
        base = heisenberg(n)
        off, sc = 0.0, 1.0
    elif kind == "global":
        base = qml.Projector(np.zeros(n, dtype=int), wires=list(range(n)))
        off, sc = 1.0, -1.0
    elif kind == "local":
        base = qml.Hamiltonian(
            [1.0 / (2 * n)] * n, [qml.PauliZ(i) for i in range(n)]
        )
        off, sc = 0.5, -1.0
    else:
        raise ValueError(kind)

    def factory(mode):
        if mode == "native":
            return base
        return qml.Hermitian(qml.matrix(base), wires=list(range(n)))

    return off, sc, factory


def make_cost(dev, factory, off, sc, b_axes, b_params, axes, n, M, L, dm, mode):
    obs = factory(mode)

    def _tape(params):
        entangler_B(b_axes, b_params, n)
        identity_blocks(params, axes, n, M, L)
        return qml.expval(obs)

    qn = qml.QNode(_tape, dev) if dm is None else qml.QNode(
        _tape, dev, diff_method=dm
    )

    def cost(p):
        return off + sc * qn(p)

    return cost


def choose_diff_path(kind, n, M, L, b_layers, seed=12345):
    """Try (diff_method, obs mode) at production size; return first working
    with its measured per-step (cost+grad) time."""
    rng = np.random.default_rng(seed)
    b_axes, b_params, axes, p = draw_instance(rng, n, M, L, b_layers)
    off, sc, factory = cost_def(kind, n)
    for dm, mode in (("adjoint", "native"), ("adjoint", "hermitian"),
                     (None, "native")):
        if mode == "hermitian" and n > 10 and kind != "local":
            continue  # dense 2^n matrix too large
        try:
            cost = make_cost(device(n), factory, off, sc, b_axes, b_params,
                             axes, n, M, L, dm, mode)
            gf = qml.grad(cost)
            t0 = time.time()
            c = float(cost(pnp.array(p, requires_grad=True)))
            g = np.asarray(gf(pnp.array(p, requires_grad=True)))
            dt = time.time() - t0
            assert np.isfinite(g).all() and np.isfinite(c)
            return dm, mode, dt
        except Exception:
            continue
    raise RuntimeError(f"no working differentiation path for cost '{kind}'")


# --------- how far a block has drifted from being the identity (numpy, fast)
def _np_rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _np_ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _np_rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


_NP_ROT = (_np_rx, _np_ry, _np_rz)


def _apply(T, U, q):
    return np.moveaxis(np.tensordot(U, T, axes=([1], [q])), 0, q)


def _cz(T, n):
    for q in range(n - 1):
        idx = [slice(None)] * n
        idx[q] = 1
        idx[q + 1] = 1
        T[tuple(idx)] *= -1.0
    return T


def block_probe(p_m, axes_m, n, L):
    """1 - |Tr U_block| / 2^n, computed in pure numpy (validated vs the
    column-based construction; 0 at identity init by construction)."""
    dim = 2**n
    T = np.eye(dim, dtype=complex).reshape((2,) * n + (dim,))
    for l in range(L):
        for q in range(n):
            T = _apply(T, _NP_ROT[axes_m[l, q]](p_m[0, l, q]), q)
        T = _cz(T, n)
    for l in reversed(range(L)):
        T = _cz(T, n)
        for q in range(n):
            T = _apply(T, _NP_ROT[axes_m[l, q]](p_m[1, l, q]), q)
    return 1.0 - abs(np.trace(T.reshape(dim, dim))) / dim


def all_probes(p, axes, n, M, L):
    return np.array([block_probe(p[m], axes[m], n, L) for m in range(M)],
                    dtype=np.float32)


# ------------- hand-rolled optimizers, because I wanted the state visible
class Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
        self.lr = lr

    def step(self, p, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9**self.t)
        vh = self.v / (1 - 0.999**self.t)
        return p - self.lr * mh / (np.sqrt(vh) + 1e-8)


class GD:
    def __init__(self, shape, lr):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g


def make_optimizer(name, shape, lr):
    return {"adam": Adam, "gd": GD}[name](shape, lr)


# ------------------------------------------------------ odds and ends
def draw_instance(rng, n, M, L, b_layers):
    b_axes = rng.integers(0, 3, size=(b_layers, n))
    b_params = rng.uniform(0, 2 * np.pi, size=(b_layers, n))
    axes = rng.integers(0, 3, size=(M, L, n))
    p = np.zeros((M, 2, L, n))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, n))
    p[:, 1] = -p[:, 0]
    return b_axes, b_params, axes, p


def null_point(rng, p0, r):
    """theta_0 + r * u_hat, u_hat uniform on the unit sphere in R^P."""
    u = rng.standard_normal(p0.shape)
    u /= np.linalg.norm(u)
    return p0 + r * u


def checkpoint_steps(steps, k=24):
    return np.unique(
        np.concatenate(
            [[0], np.round(np.logspace(0, np.log10(max(steps - 1, 2)), k)),
             [steps - 1]]
        ).astype(int)
    )


def default_watch(n, M, L):
    idx = []
    shape = (M, 2, L, n)
    for m in range(M):
        for half in range(2):
            for l in (0, L // 2, L - 1):
                for q in range(n):
                    idx.append(np.ravel_multi_index((m, half, l, q), shape))
    return np.array(sorted(set(idx)))