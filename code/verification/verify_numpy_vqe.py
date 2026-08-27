"""
Plain-numpy checks for the Heisenberg VQE setup, before trusting the
PennyLane version of it.

Does five things:
  1. gets the exact ground energy by brute-force diagonalisation, which is
     the green line every training plot is measured against
  2. confirms the blocks leave B|0> untouched at init, entangler and all
  3. parameter-shift vs finite difference, again
  4. actually trains a tiny version and checks the energy goes DOWN, plus
     watches the block drift away from identity as it does (0 -> 0.98,
     which is a nice preview of the real thing)
  5. prints what the starting energy looks like for a few random B draws,
     so we know roughly how far there is to fall
"""

import time

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


ROT = (rx, ry, rz)


def kron_term(paulis, n):
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, paulis.get(q, I2))
    return out


def heisenberg_matrix(n, J=1.0, h=1.0):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        j = (i + 1) % n
        for P in (X, Y, Z):
            H += J * kron_term({i: P, j: P}, n)
    for i in range(n):
        H += h * kron_term({i: Z}, n)
    return H


def apply_1q(state, U, q):
    state = np.tensordot(U, state, axes=([1], [q]))
    return np.moveaxis(state, 0, q)


def apply_cz_chain(state, n):
    for q in range(n - 1):
        idx = [slice(None)] * n
        idx[q] = 1
        idx[q + 1] = 1
        state[tuple(idx)] *= -1.0
    return state


def entangler_B(state, b_axes, b_params, n):
    for l in range(b_axes.shape[0]):
        for q in range(n):
            state = apply_1q(state, ROT[b_axes[l, q]](b_params[l, q]), q)
        state = apply_cz_chain(state, n)
    return state


def identity_blocks(state, p, axes, n, M, L):
    for m in range(M):
        for l in range(L):
            for q in range(n):
                state = apply_1q(state, ROT[axes[m, l, q]](p[m, 0, l, q]), q)
            state = apply_cz_chain(state, n)
        for l in reversed(range(L)):
            state = apply_cz_chain(state, n)
            for q in range(n):
                state = apply_1q(state, ROT[axes[m, l, q]](p[m, 1, l, q]), q)
    return state


def run_state(p, axes, b_axes, b_params, n, M, L):
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    state = entangler_B(state, b_axes, b_params, n)
    return identity_blocks(state, p, axes, n, M, L)


def energy(p, axes, b_axes, b_params, n, M, L, Hmat):
    psi = run_state(p, axes, b_axes, b_params, n, M, L).ravel()
    return float(np.real(psi.conj() @ Hmat @ psi))


def full_grad_shift(p, axes, b_axes, b_params, n, M, L, Hmat):
    g = np.zeros_like(p)
    flat = p.ravel()
    for k in range(flat.size):
        pp, pm = flat.copy(), flat.copy()
        pp[k] += np.pi / 2
        pm[k] -= np.pi / 2
        g.ravel()[k] = 0.5 * (
            energy(pp.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
            - energy(pm.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
        )
    return g


def block_unitary(p_m, axes_m, n, L):
    dim = 2**n
    U = np.empty((dim, dim), dtype=complex)
    for col in range(dim):
        state = np.zeros(dim, dtype=complex)
        state[col] = 1.0
        state = state.reshape((2,) * n)
        state = identity_blocks(state, p_m[None], axes_m[None], n, 1, L)
        U[:, col] = state.ravel()
    return U


def main():
    # 1. brute-force the ground energy -- this is the target line on the plots
    t0 = time.time()
    H7 = heisenberg_matrix(7)
    evals = np.linalg.eigvalsh(H7)
    print(f"n=7 periodic Heisenberg (J=h=1): E_min = {evals[0]:.6f}  "
          f"(gap to 1st excited: {evals[1] - evals[0]:.4f})  "
          f"[{time.time() - t0:.1f}s]")

    # tiny circuit for the structural checks; big enough to be meaningful
    n, M, L, b_layers = 4, 1, 3, 2
    Hmat = heisenberg_matrix(n)
    e_min4 = float(np.linalg.eigvalsh(Hmat)[0])
    rng = np.random.default_rng(42)
    b_axes = rng.integers(0, 3, size=(b_layers, n))
    b_params = rng.uniform(0, 2 * np.pi, size=(b_layers, n))
    axes = rng.integers(0, 3, size=(M, L, n))
    p = np.zeros((M, 2, L, n))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, n))
    p[:, 1] = -p[:, 0]

    # 2. blocks do nothing at init, even with the entangler in front
    state0 = np.zeros((2,) * n, dtype=complex)
    state0[(0,) * n] = 1.0
    psi_b = entangler_B(state0.copy(), b_axes, b_params, n)
    psi_full = run_state(p, axes, b_axes, b_params, n, M, L)
    d = np.max(np.abs(psi_full - psi_b))
    assert d < 1e-10, f"blocks not identity at init ({d:.2e})"
    probe = 1 - abs(np.trace(block_unitary(p[0], axes[0], n, L))) / 2**n
    assert probe < 1e-10, f"block probe nonzero at init ({probe:.2e})"
    print(f"identity at init OK (diff {d:.1e}); block probe at init = "
          f"{probe:.1e}")

    # 3. gradients agree with a dumb finite difference
    for k in (0, 5, p.size - 1):
        pp, pm = p.copy().ravel(), p.copy().ravel()
        h_ = 1e-5
        pp[k] += h_
        pm[k] -= h_
        fd = (
            energy(pp.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
            - energy(pm.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
        ) / (2 * h_)
        ppp, pmm = p.copy().ravel(), p.copy().ravel()
        ppp[k] += np.pi / 2
        pmm[k] -= np.pi / 2
        gs = 0.5 * (
            energy(ppp.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
            - energy(pmm.reshape(p.shape), axes, b_axes, b_params, n, M, L, Hmat)
        )
        assert abs(fd - gs) < 1e-8, f"shift vs FD mismatch at {k}"
    print("parameter-shift == finite difference OK")

    # 4. can it actually train? energy should drop, blocks should scramble
    e_init = energy(p, axes, b_axes, b_params, n, M, L, Hmat)
    m_t, v_t = np.zeros_like(p), np.zeros_like(p)
    b1, b2, eps, lr, steps = 0.9, 0.999, 1e-8, 0.05, 40
    pc = p.copy()
    for t in range(steps):
        g = full_grad_shift(pc, axes, b_axes, b_params, n, M, L, Hmat)
        m_t = b1 * m_t + (1 - b1) * g
        v_t = b2 * v_t + (1 - b2) * g * g
        pc = pc - lr * (m_t / (1 - b1 ** (t + 1))) / (
            np.sqrt(v_t / (1 - b2 ** (t + 1))) + eps
        )
    e_final = energy(pc, axes, b_axes, b_params, n, M, L, Hmat)
    dist = float(np.linalg.norm(pc - p))
    probe_after = 1 - abs(np.trace(block_unitary(pc[0], axes[0], n, L))) / 2**n
    assert e_final < e_init - 0.1, "training smoke test failed to descend"
    print(f"training smoke test OK (n=4, M=1, L=3, Adam lr=0.05, {steps} "
          f"steps):\n  E {e_init:+.4f} -> {e_final:+.4f}  "
          f"(E_min {e_min4:+.4f});  ||dtheta|| = {dist:.3f};  "
          f"block probe 0 -> {probe_after:.3f}")

    # 5. where does training start from, roughly?
    es = []
    for s in range(5):
        r = np.random.default_rng(100 + s)
        ba = r.integers(0, 3, size=(7, 7))
        bp = r.uniform(0, 2 * np.pi, size=(7, 7))
        s0 = np.zeros((2,) * 7, dtype=complex)
        s0[(0,) * 7] = 1.0
        psi = entangler_B(s0, ba, bp, 7).ravel()
        es.append(float(np.real(psi.conj() @ H7 @ psi)))
    print(f"init energies <B0|H|B0> at n=7 over 5 B draws: "
          + ", ".join(f"{e:+.2f}" for e in es))


if __name__ == "__main__":
    main()