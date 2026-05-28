# soap_truncated.py
# ─────────────────────────────────────────────────────────────────────────────
# SOAP with two memory-efficient extensions, building on Vyas et al. 2024
# (https://github.com/nikhilvyas/SOAP).
#
# 1. STREAMING LOW-RANK Kronecker factors
#    Maintain L (and/or R) as a low-rank factor F such that L ≈ F F^T,
#    WITHOUT ever materializing the full d × d preconditioner.
#    F has shape (d, k) where k << d.
#
#    Memory savings vs. vanilla SOAP, per side:
#      vanilla:      d² floats  for GG + d² for Q  = 2 d² floats
#      streaming:    d × k for F + d × k for Q     = 2 d k floats
#    Ratio: d / k. For MLP up-proj (d=4096, k=32): 128× smaller.
#
# 2. SOAP-MINI second moment
#    In SOAP's projected (eigenbasis) space, replace the full per-element
#    second moment V with a reduced form:
#      'scalar'  : single scalar v_t per layer
#      'per_row' : one v_t per row of projected gradient
#      'per_col' : one v_t per col of projected gradient
#      'none'    : standard SOAP (default)
#
#    SOAP's projection approximately equalizes curvature across directions,
#    so a coarser V is principled in the projected space.
# ─────────────────────────────────────────────────────────────────────────────

import math
import re
import time
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim


# Parameter-name patterns that get streaming low-rank approximation.
# Each entry: (regex, {side_index: rank}).
# side_index: 0 = first dim (output / L-side), 1 = second dim (input / R-side)
DEFAULT_LOW_RANK_PATTERNS = [
    # nanoGPT layout
    (re.compile(r'attn\.wq\.weight'),    {0: 256, 1: 128}),
    (re.compile(r'attn\.wk\.weight'),    {0: 256, 1: 128}),
    (re.compile(r'attn\.wv\.weight'),    {0: 256, 1: 128}),
    
    (re.compile(r'mlp\.c_fc\.weight'),   {0: 128, 1: 128}),
    (re.compile(r'mlp\.c_proj\.weight'), {0: 128, 1: 128}),
]

# Block-diagonal structure (existing feature, kept for compatibility)
HEAD_BLOCK_PATTERNS = [re.compile(r'attn\.wk')]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_low_rank_config(param_name, patterns):
    """Returns {side: rank} dict if param matches, else None."""
    if param_name is None:
        return None
    for pat, rank_map in patterns:
        if pat.search(param_name):
            return rank_map
    return None


def streaming_lowrank_update(F_old, G_side, beta2, k, eps=1e-30):
    """
    Streaming low-rank update of L = F F^T such that
        L_new ≈ beta2 * L_old + (1 - beta2) * G_side @ G_side.T,
    truncated to rank k.

    The full d × d L is never materialized; only the d × k factor F is stored.

    Args:
        F_old: (d, k) tensor — current low-rank factor, OR None for first step.
        G_side: (d, q) tensor — appropriately-shaped gradient
                (G itself for L-side; G.T for R-side).
        beta2: float — EMA decay.
        k: int — target rank.
        eps: float — numerical stabilizer.

    Returns:
        F_new: (d, k) tensor — updated low-rank factor (columns = sqrt(eig)*u_i)
        Q:     (d, k) tensor — orthonormal eigenvectors (top-k of L_new), descending
        eigs:  (k,) tensor — corresponding eigenvalues, descending, clamped >= 0
    """
    sqrt_b2 = math.sqrt(beta2)
    sqrt_1mb2 = math.sqrt(1.0 - beta2)

    # Build A so that A @ A.T = beta2 * F_old F_old^T + (1-beta2) G_side G_side^T
    if F_old is None or F_old.numel() == 0:
        # Bootstrap step: L = (1-beta2) * G_side G_side^T
        A = sqrt_1mb2 * G_side
    else:
        A = torch.cat([sqrt_b2 * F_old, sqrt_1mb2 * G_side], dim=1)

    d, n_cols = A.shape
    device, dtype = A.device, A.dtype

    # If A is already <= k columns, no truncation needed; just orthogonalize.
    if n_cols <= k:
        # Compute eigenvectors via small Gram on A^T A (n_cols × n_cols)
        AtA = A.T @ A
        AtA = 0.5 * (AtA + AtA.T)
        trace = AtA.diagonal().sum().clamp(min=1e-12)
        reg = (1e-6 * trace) / AtA.shape[0]
        eye_q = torch.eye(n_cols, device=device, dtype=dtype)
        AtA = AtA + reg * eye_q
        
        eigs_small, V_small = torch.linalg.eigh(AtA)
        # Descending order; pad to k with zero eigenvalues / zero columns
        eigs_small = torch.flip(eigs_small, [0]).clamp(min=0)
        V_small = torch.flip(V_small, [1])
        F_packed = A @ V_small  # (d, n_cols), columns have norm sqrt(eigs_small)

        F_new = torch.zeros(d, k, device=device, dtype=dtype)
        F_new[:, :n_cols] = F_packed
        eigs = torch.zeros(k, device=device, dtype=dtype)
        eigs[:n_cols] = eigs_small

        sqrt_eigs = torch.sqrt(eigs.clamp(min=0)).clamp(min=eps)
        Q = F_new / sqrt_eigs.unsqueeze(0)
        return F_new, Q, eigs

    # Truncate to rank k via the cheaper Gram side
    if d <= n_cols:
        # eigendecompose A A^T (d × d) — gets U directly, columns of U are eigenvectors
        AAt = A @ A.T
        AAt = 0.5 * (AAt + AAt.T)
        trace = AAt.diagonal().sum().clamp(min=1e-12)
        reg = (1e-6 * trace) / AAt.shape[0]
        eye_d = torch.eye(d, device=device, dtype=dtype)
        AAt = AAt + reg * eye_d
        
        eigs_full, U_full = torch.linalg.eigh(AAt)
        eigs = torch.flip(eigs_full, [0])[:k].clamp(min=0)
        U = torch.flip(U_full, [1])[:, :k]
        Q = U
        F_new = U * torch.sqrt(eigs).clamp(min=eps).unsqueeze(0)
    else:
        # eigendecompose A^T A ((k+q) × (k+q)) — get V then F_new = A V
        AtA = A.T @ A
        AtA = 0.5 * (AtA + AtA.T)
        trace = AtA.diagonal().sum().clamp(min=1e-12)
        reg = (1e-6 * trace) / AtA.shape[0]
        eye_q = torch.eye(n_cols, device=device, dtype=dtype)
        AtA = AtA + reg * eye_q
        
        eigs_full, V_full = torch.linalg.eigh(AtA)
        eigs = torch.flip(eigs_full, [0])[:k].clamp(min=0)
        V = torch.flip(V_full, [1])[:, :k]
        F_new = A @ V  # (d, k); column i has norm sqrt(eig_i)
        sqrt_eigs = torch.sqrt(eigs).clamp(min=eps)
        Q = F_new / sqrt_eigs.unsqueeze(0)

    return F_new, Q, eigs


def reduce_for_soap_mini(grad_projected_sq, mode):
    """
    Reduce a per-element squared-gradient tensor to the SOAP-mini grouping.

    Args:
        grad_projected_sq: tensor in projected space, any shape.
        mode: 'none' | 'scalar' | 'per_row' | 'per_col'

    Returns:
        Reduced tensor with shape that broadcasts to grad_projected_sq's shape.
    """
    if mode == 'none':
        return grad_projected_sq
    if mode == 'scalar':
        # Mean over all dims; shape with all 1s for broadcasting.
        out = grad_projected_sq.mean()
        return out.view(*([1] * grad_projected_sq.dim()))
    if mode == 'per_row':
        # Average over all dims except 0 → shape (d0, 1, 1, ...)
        reduce_dims = tuple(range(1, grad_projected_sq.dim()))
        if len(reduce_dims) == 0:
            return grad_projected_sq
        return grad_projected_sq.mean(dim=reduce_dims, keepdim=True)
    if mode == 'per_col':
        # Average over all dims except the last → shape (1, ..., 1, d_last)
        reduce_dims = tuple(range(grad_projected_sq.dim() - 1))
        if len(reduce_dims) == 0:
            return grad_projected_sq
        return grad_projected_sq.mean(dim=reduce_dims, keepdim=True)
    raise ValueError(f"Unknown soap_mini_mode: {mode!r}")


def soap_mini_shape(proj_shape, mode):
    """Return the target shape for exp_avg_sq under SOAP-mini."""
    if mode == 'none':
        return list(proj_shape)
    if mode == 'scalar':
        return [1] * len(proj_shape)
    if mode == 'per_row':
        return [proj_shape[0]] + [1] * (len(proj_shape) - 1)
    if mode == 'per_col':
        return [1] * (len(proj_shape) - 1) + [proj_shape[-1]]
    raise ValueError(f"Unknown soap_mini_mode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# SOAP with streaming low-rank + SOAP-mini
# ─────────────────────────────────────────────────────────────────────────────

class SOAPTruncated(optim.Optimizer):
    """
    SOAP optimizer with:
      - Streaming low-rank Kronecker factors (no full L materialization).
      - SOAP-mini reduced second moment in the projected space.
      - Backward-compatible with vanilla SOAP when no low-rank patterns set.

    Args:
        params: iterable of nn.Parameter or param dicts.
        lr, betas, shampoo_beta, eps, weight_decay, precondition_frequency,
        max_precond_dim, merge_dims, precondition_1d, normalize_grads,
        data_format, correct_bias: same as vanilla SOAP.
        param_to_name: dict mapping id(param) -> name string. Required for
            matching against low-rank patterns / head-block patterns.
        n_heads: int, only used for head-blocked params.

        use_streaming_lowrank: if True, parameters matching low_rank_patterns
            use streaming low-rank factor F; if False, behaves like vanilla
            SOAP regardless of patterns.
        low_rank_patterns: list of (compiled_regex, {side: rank}). Defaults to
            DEFAULT_LOW_RANK_PATTERNS. Only used when use_streaming_lowrank=True.

        soap_mini_mode: 'none' | 'scalar' | 'per_row' | 'per_col'.
            Reduces the projected-space second moment V.
        soap_mini_apply: which params to apply SOAP-mini to.
            'all'        — every 2D param
            'low_rank'   — only params using streaming low-rank
            'mlp'        — params whose name matches MLP patterns
            'none'       — disable (overrides soap_mini_mode='none')
        soap_mini_mlp_patterns: regexes considered "MLP" when soap_mini_apply='mlp'.

        use_k_block_diag: keep existing block-diagonal-attn-keys feature.
    """

    DEFAULT_MLP_PATTERNS = [
        re.compile(r'mlp\.c_fc'),
        re.compile(r'mlp\.c_proj'),
        re.compile(r'mlp\.dense_h_to_4h'),
        re.compile(r'mlp\.dense_4h_to_h'),
    ]

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        param_to_name=None,
        n_heads: int = 16,
        # ---- New options ----
        use_streaming_lowrank: bool = False,
        streaming_q_freq: int = 1,
        low_rank_patterns=None,
        soap_mini_mode: str = 'none',
        soap_mini_apply: str = 'all',
        soap_mini_mlp_patterns=None,
        use_k_block_diag: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, merge_dims=merge_dims,
            precondition_1d=precondition_1d, normalize_grads=normalize_grads,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)
        self._data_format = data_format
        self.param_to_name = param_to_name or {}
        self.n_heads = n_heads

        self.use_streaming_lowrank = use_streaming_lowrank
        self.streaming_q_freq = streaming_q_freq
        self.low_rank_patterns = (
            low_rank_patterns if low_rank_patterns is not None
            else DEFAULT_LOW_RANK_PATTERNS
        )

        assert soap_mini_mode in ('none', 'scalar', 'per_row', 'per_col')
        assert soap_mini_apply in ('all', 'low_rank', 'mlp', 'none')
        self.soap_mini_mode = soap_mini_mode
        self.soap_mini_apply = soap_mini_apply
        self.soap_mini_mlp_patterns = (
            soap_mini_mlp_patterns if soap_mini_mlp_patterns is not None
            else self.DEFAULT_MLP_PATTERNS
        )

        self.use_k_block_diag = use_k_block_diag
        self._last_eig_time = 0.0

    # ─── helpers for per-param routing ─────────────────────────────────────

    def _is_head_blocked(self, param_name):
        if not self.use_k_block_diag or param_name is None:
            return False
        return any(p.search(param_name) for p in HEAD_BLOCK_PATTERNS)

    def _streaming_config(self, param_name):
        """Returns {side: rank} or None for streaming low-rank."""
        if not self.use_streaming_lowrank:
            return None
        return get_low_rank_config(param_name, self.low_rank_patterns)

    def _soap_mini_active(self, param_name, has_streaming):
        """Should this param use SOAP-mini second moment?"""
        if self.soap_mini_mode == 'none' or self.soap_mini_apply == 'none':
            return False
        if self.soap_mini_apply == 'all':
            return True
        if self.soap_mini_apply == 'low_rank':
            return has_streaming
        if self.soap_mini_apply == 'mlp':
            if param_name is None:
                return False
            return any(p.search(param_name) for p in self.soap_mini_mlp_patterns)
        return False

    # ─── core ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # ---- first-time init for this parameter ----
                if "step" not in state:
                    state["step"] = 0

                if "Q" not in state:
                    param_name = self.param_to_name.get(id(p))
                    self._init_state(
                        grad, state, group, param_name,
                    )
                    # First step: no parameter update; just initialize preconditioner.
                    self._update_preconditioner(grad, state, group)
                    continue

                # ---- regular SOAP step ----
                grad_projected = self._project(grad, state)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                soap_mini = state.get("soap_mini_mode", "none")

                state["step"] += 1

                # First moment: same as standard Adam (in projected space)
                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))

                # Second moment: standard SOAP, or SOAP-mini reduction
                gp_sq = grad_projected.square()
                if soap_mini != 'none':
                    gp_sq = reduce_for_soap_mini(gp_sq, soap_mini)
                exp_avg_sq.mul_(beta2).add_(gp_sq, alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bc2 ** 0.5) / bc1

                # exp_avg / denom (broadcasting works for SOAP-mini's reduced shapes)
                norm_grad = self._project_back(exp_avg / denom, state)

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + (norm_grad.square().mean()) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Update preconditioner (after parameter update — prevents leakage)
                self._update_preconditioner(grad, state, group)

        return loss

    # ─── state initialization ──────────────────────────────────────────────

    def _init_state(self, grad, state, group, param_name):
        state["param_name"] = param_name
        state["streaming_ranks"] = self._streaming_config(param_name)  # dict or None
        state["is_head_blocked"] = self._is_head_blocked(param_name)
        state["n_heads"] = self.n_heads if state["is_head_blocked"] else None
        
        # --- for debugging ---
        if state["streaming_ranks"] is not None:
            print(f"[DEBUG] SOAP-Streaming active for: {param_name} | Ranks: {state['streaming_ranks']}")
        # ---
        
        has_streaming = state["streaming_ranks"] is not None
        soap_mini = (
            self.soap_mini_mode
            if self._soap_mini_active(param_name, has_streaming)
            else 'none'
        )
        state["soap_mini_mode"] = soap_mini

        max_precond_dim = group["max_precond_dim"]

        if has_streaming:
            # Initialize F-list (one per side); Q-list and eigs-list mirror it.
            state["F"] = [None] * grad.dim()
            state["Q"] = [None] * grad.dim()
            state["eigs"] = [None] * grad.dim()
            # We don't maintain GG for streaming sides, but for non-streaming
            # sides we may need it (mixed mode).
            state["GG"] = []
            for idx, sh in enumerate(grad.shape):
                if idx in state["streaming_ranks"]:
                    state["GG"].append(None)  # placeholder
                elif sh <= max_precond_dim:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))
                else:
                    state["GG"].append([])
        else:
            # Standard SOAP path
            state["F"] = None
            state["Q"] = None
            state["GG"] = []
            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))
                else:
                    state["GG"].append([])

        # Compute the projected gradient shape, taking streaming ranks into account.
        proj_shape = list(grad.shape)
        if has_streaming:
            for side, rank in state["streaming_ranks"].items():
                proj_shape[side] = rank
        state["proj_shape"] = proj_shape

        # First moment is always full proj shape; second moment may be reduced.
        state["exp_avg"] = torch.zeros(proj_shape, dtype=grad.dtype, device=grad.device)
        v_shape = soap_mini_shape(proj_shape, soap_mini)
        state["exp_avg_sq"] = torch.zeros(v_shape, dtype=grad.dtype, device=grad.device)

        state["shampoo_beta"] = (
            group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
        )
        state["precondition_frequency"] = group["precondition_frequency"]

    # ─── preconditioner update ─────────────────────────────────────────────

    def _update_preconditioner(self, grad, state, group):
        """
        Updates state['F'] / state['GG'] and state['Q'] in-place.

        Streaming sides get refreshed every step (Q comes free from F).
        Non-streaming sides follow vanilla SOAP: accumulate GG every step,
        refresh Q every `precondition_frequency` steps.
        """
        
        beta_p = state["shampoo_beta"]
        max_pd = group["max_precond_dim"]
        streaming_ranks = state.get("streaming_ranks")
        has_streaming = streaming_ranks is not None

        # Save old Q for re-projecting exp_avg later (when Q changes)
        old_Q = state["Q"] if state["Q"] is not None and any(
            (q is not None and (not isinstance(q, list) or len(q) > 0))
            for q in (state["Q"] if isinstance(state["Q"], list) else [state["Q"]])
        ) else None

        # Re-project exp_avg back to the un-rotated space (so we can re-rotate after Q change).
        # Only do this if Q is fully populated (not on bootstrap step).
        Q_ready = (
            state["Q"] is not None
            and isinstance(state["Q"], list)
            and all(q is not None for q in state["Q"])
        )
        if Q_ready and state["step"] > 0:
            state["exp_avg"] = self._project_back(state["exp_avg"], state)

        if grad.dim() == 1:
            # 1D params: skip preconditioning unless precondition_1d is set.
            # Vanilla SOAP defers to AdamW for 1D in practice; we follow.
            return

        # ---- update each side ----
        for idx, sh in enumerate(grad.shape):
            if has_streaming and idx in streaming_ranks:
                k = streaming_ranks[idx]
                # Build the appropriate gradient slice for this side
                if grad.dim() == 2:
                    G_side = grad if idx == 0 else grad.T
                else:
                    # Generalize: bring side `idx` to the front, flatten the rest
                    perm = [idx] + [j for j in range(grad.dim()) if j != idx]
                    G_side = grad.permute(perm).reshape(sh, -1)

                F_old = state["F"][idx]
                F_new, Q_new, eigs_new = streaming_lowrank_update(
                    F_old, G_side, beta_p, k
                )
                state["F"][idx] = F_new
                # state["Q"][idx] = Q_new   not needed since used with streaming_q_freq below
                state["eigs"][idx] = eigs_new
                
                is_bootstrap = (state["Q"][idx] is None)
                if is_bootstrap or (state["step"] % self.streaming_q_freq == 0):
                    state["Q"][idx] = Q_new

            elif sh <= max_pd:
                # Vanilla SOAP path for this side
                gg = state["GG"][idx]
                # Outer product accumulation
                if grad.dim() == 2:
                    if idx == 0:
                        outer = grad @ grad.T
                    else:
                        outer = grad.T @ grad
                else:
                    contract = [j for j in range(grad.dim()) if j != idx]
                    outer = torch.tensordot(grad, grad, dims=[contract, contract])
                gg.lerp_(outer, 1.0 - beta_p)

                # Refresh Q on first step or every precondition_frequency
                need_refresh = (
                    (state["Q"] is None)
                    or (state["Q"][idx] is None)
                    or (state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0)
                )
                if need_refresh:
                    if state["Q"] is None:
                        state["Q"] = [None] * grad.dim()
                    Q_new = self._eigh_top(gg, idx, state)
                    state["Q"][idx] = Q_new

        # Re-project exp_avg into the new basis (skip on bootstrap when exp_avg is freshly zero).
        new_Q_ready = (
            state["Q"] is not None
            and isinstance(state["Q"], list)
            and all(q is not None for q in state["Q"])
        )
        if new_Q_ready and state["step"] > 0:
            # exp_avg is currently in the OLD basis (we project_back'd above)
            state["exp_avg"] = self._project(state["exp_avg"], state)
        elif new_Q_ready and state["step"] == 0:
            # Bootstrap: exp_avg is zeros; just leave it but ensure shape matches proj_shape
            # (it was initialized with proj_shape so this is automatic).
            pass

    def _eigh_top(self, gg, idx, state):
        """Full eigendecomposition of GG matrix for non-streaming sides."""
        gg_sym = 0.5 * (gg + gg.T)
        eye = torch.eye(gg.shape[0], device=gg.device, dtype=gg.dtype)
        try:
            eigs, Q = torch.linalg.eigh(gg_sym + 1e-30 * eye)
        except Exception:
            eigs, Q = torch.linalg.eigh(
                gg_sym.to(torch.float64) + 1e-30 * eye.to(torch.float64)
            )
            Q = Q.to(gg.dtype)
        return torch.flip(Q, [1])

    # ─── projection ────────────────────────────────────────────────────────

    def _project(self, grad, state):
        """Project gradient (or exp_avg) to the eigenbasis."""
        if state["Q"] is None:
            return grad
        out = grad
        for idx, mat in enumerate(state["Q"]):
            if mat is None or (isinstance(mat, list) and len(mat) == 0):
                # No projection on this side; rotate it to the back of the dim list
                # so that the next iteration's tensordot acts on the next side.
                perm = list(range(1, out.dim())) + [0]
                out = out.permute(perm)
            else:
                # Q has shape (d, k_or_d); contract grad's dim 0 with Q's dim 0.
                out = torch.tensordot(out, mat, dims=[[0], [0]])
        return out

    def _project_back(self, projected, state):
        """Project from eigenbasis back to the original space."""
        if state["Q"] is None:
            return projected
        out = projected
        for idx, mat in enumerate(state["Q"]):
            if mat is None or (isinstance(mat, list) and len(mat) == 0):
                perm = list(range(1, out.dim())) + [0]
                out = out.permute(perm)
            else:
                # Contract dim 0 with Q's column dim (dim 1)
                out = torch.tensordot(out, mat, dims=[[0], [1]])
        return out