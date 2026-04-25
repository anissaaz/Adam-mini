# SOAP optimizer from https://github.com/nikhilvyas/SOAP
# Original paper: https://arxiv.org/abs/2409.11321

import torch
import torch.nn as nn
import torch.optim as optim

import re
import time

from itertools import chain

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py

# Parameter-name patterns that get block-diagonal L preconditioner.
# Add 'attn.wq' later to extend, 'mlp.c_fc|mlp.c_proj' for MLPs (needs different block size).
HEAD_BLOCK_PATTERNS = [re.compile(r'attn\.wk')]
LOW_RANK_PATTERNS = [(re.compile(r'mlp\.c_fc\.weight'), 1, 32)]  # (pattern, side_idx, rank)


def blockdiag_eigh(L, n_heads):
    """Eigendecompose a block-diagonal matrix by extracting per-head blocks and processing them as a batched eigh.
    
    L: (d, d). Only the n_heads diagonal blocks of size (head_dim, head_dim) are used;
    off-block entries are ignored (assumed already zero).
    
    Returns (Q, eigs):
      Q:    (d, d) block-diagonal orthogonal matrix
      eigs: (d,) concatenated eigenvalues, ordered ascending within each block
    """
    d = L.shape[0]
    assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"
    head_dim = d // n_heads
    
    # Extract diagonal blocks into (n_heads, head_dim, head_dim) tensor
    blocks = torch.zeros((n_heads, head_dim, head_dim), 
                         dtype=L.dtype, device=L.device)
    for i in range(n_heads):
        s = slice(i * head_dim, (i + 1) * head_dim)
        blocks[i] = L[s, s]
    
    # Symmetrize and ridge in a batched way
    blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
    eye_b = torch.eye(head_dim, dtype=blocks.dtype, device=blocks.device).expand_as(blocks)
    blocks = blocks + 1e-30 * eye_b
    
    # Batched eigh — single kernel launch
    try:
        eigs_batched, vecs_batched = torch.linalg.eigh(blocks)
    except Exception:
        eigs_batched, vecs_batched = torch.linalg.eigh(blocks.to(torch.float64))
        eigs_batched = eigs_batched.to(L.dtype)
        vecs_batched = vecs_batched.to(L.dtype)
        
    # Assemble back into (d, d) block-diagonal Q
    Q = torch.zeros((d, d), dtype=L.dtype, device=L.device)
    eigs = torch.zeros(d, dtype=L.dtype, device=L.device)
    for i in range(n_heads):
        s = slice(i * head_dim, (i + 1) * head_dim)
        Q[s, s] = vecs_batched[i]
        eigs[s] = eigs_batched[i]
    
    return Q, eigs

def get_low_rank_config(param_name, patterns=LOW_RANK_PATTERNS):
    """Returns (side, rank) if param matches a low-rank pattern, else None."""
    if param_name is None:
        return None
    for pat, side, rank in patterns:
        if pat.search(param_name):
            return (side, rank)
    return None

def truncated_eigh(L, k):
    """Top-k eigendecomposition of symmetric PSD L via full eigh + truncation.
    Returns (Q_k, eigs_k) where Q_k is (d, k), eigs_k is (k,) descending.
    
    lobpcg is intentionally avoided: on matrices where k is not tiny relative
    to d (here k=64, d=4096, ratio=1.6%), lobpcg frequently fails to converge
    within its iteration budget and returns finite-but-wrong eigenvectors.
    Full eigh is faster in practice and always returns correct eigenvectors.
    """
    L = 0.5 * (L + L.T)
    d = L.shape[0]
    # if not torch.isfinite(L).all():
    #     L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    
    ridge = L.diag().abs().mean().clamp(min=1e-2) * 1e-6 + 1e-8  # stays on GPU
    eye = torch.eye(d, device=L.device, dtype=L.dtype)
    eigs_full, Q_full = torch.linalg.eigh(L + ridge * eye)
    # try:
    #     eigs_full, Q_full = torch.linalg.eigh(L + ridge * eye)
    #     # if not torch.isfinite(Q_full).all() or not torch.isfinite(eigs_full).all():
    #     #     raise RuntimeError("eigh fp32 silent NaN")
    # except Exception:
    #     L64 = L.to(torch.float64)
    #     mean64 = L64.diag().abs().mean().item()
    #     ridge64 = max(1e-4 * mean64, 1e-12)
    #     eye64 = torch.eye(d, device=L.device, dtype=torch.float64)
    #     eigs_full, Q_full = torch.linalg.eigh(L64 + ridge64 * eye64)
    #     if not torch.isfinite(Q_full).all():
    #         Q_full = torch.eye(d, dtype=torch.float64, device=L.device)
    #         eigs_full = torch.ones(d, dtype=torch.float64, device=L.device)
    #     eigs_full = eigs_full.to(L.dtype)
    #     Q_full = Q_full.to(L.dtype)

    eigs = torch.flip(eigs_full, [0])[:k]
    Q = torch.flip(Q_full, [1])[:, :k]
    return Q, eigs

class SOAP(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        max_precond_dim: int=10000, # 
        merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        param_to_name: int = None, 
        n_heads: int=16,
        use_k_block_diag: bool = False,
        use_low_rank: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        self.param_to_name = param_to_name or {}
        self.n_heads = n_heads
        self._last_eig_time = 0.0
        self.use_k_block_diag = use_k_block_diag
        self.use_low_rank = use_low_rank
        self._eigh_fp32_failures = 0
        
    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        
        if curr_shape > 1 or len(new_shape)==0:
            new_shape.append(curr_shape)
        
        new_grad = grad.reshape(new_shape)
        return new_grad               

    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0 
                
                if 'Q' not in state:
                    param_name = self.param_to_name.get(id(p))
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        precondition_1d=group['precondition_1d'],
                        shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]),
                        max_precond_dim=group['max_precond_dim'],
                        merge_dims=group["merge_dims"],
                        param_name=param_name,
                        n_heads=self.n_heads,
                    )
                    
                    proj_shape = list(grad.shape)
                    if state.get('low_rank_side') is not None:
                        proj_shape[state['low_rank_side']] = state['low_rank_k']
                    state["exp_avg"] = torch.zeros(proj_shape, dtype=grad.dtype, device=grad.device)
                    state["exp_avg_sq"] = torch.zeros(proj_shape, dtype=grad.dtype, device=grad.device)
                    
                    
                    self.update_preconditioner(grad, state,
                                               max_precond_dim=group['max_precond_dim'],
                                               merge_dims=group["merge_dims"],
                                               precondition_1d=group["precondition_1d"])
                    continue # first step is skipped so that we never use the current gradients in the projection.  
                  
                # State initialization
                # if "exp_avg" not in state:
                #     # Exponential moving average of gradient values
                #     state["exp_avg"] = torch.zeros_like(grad)
                #     # Exponential moving average of squared gradient values
                #     state["exp_avg_sq"] = torch.zeros_like(grad)
                

                # Projecting gradients to the eigenbases of Shampoo's preconditioner 
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(grad, state, merge_dims=group["merge_dims"], 
                                              max_precond_dim=group['max_precond_dim'])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                
                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner 
                # i.e. projecting to the eigenbases of matrices in state['GG']
                # exp_avg_projected = self.project(exp_avg, state, merge_dims=group["merge_dims"],
                #                                  max_precond_dim=group['max_precond_dim'])
                exp_avg_projected = exp_avg
                
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2 ** .5) / bias_correction1

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(exp_avg_projected / denom, state, merge_dims=group["merge_dims"],
                                                 max_precond_dim=group['max_precond_dim'])

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)
                
                p.add_(norm_grad, alpha=-step_size)
                

                # From AdamW code: Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    
                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(grad, state, 
                                               max_precond_dim=group['max_precond_dim'],
                                               merge_dims=group["merge_dims"],
                                               precondition_1d=group["precondition_1d"])
        
        return loss
    
    def _is_head_blocked(self, param_name):
        if not self.use_k_block_diag:
            return False
        if param_name is None:
            return False
        return any(p.search(param_name) for p in HEAD_BLOCK_PATTERNS)

    def init_preconditioner(self, grad, state, precondition_frequency=10, 
                            shampoo_beta=0.95, max_precond_dim=10000, precondition_1d=False,
                            merge_dims=False, param_name=None, n_heads=None):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        
        state['param_name'] = param_name
        state['is_head_blocked'] = self._is_head_blocked(param_name)
        state['n_heads'] = n_heads if state['is_head_blocked'] else None  
        
        # low-rank config
        lr_config = get_low_rank_config(param_name) if self.use_low_rank else None
        state['low_rank_side'] = lr_config[0] if lr_config else None
        state['low_rank_k'] = lr_config[1] if lr_config else None
         
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))
                    
        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta
   
        
    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [0]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        
    def update_preconditioner(self, grad, state, 
                              max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        
        if state.get('is_head_blocked'):
            assert not merge_dims, "Head-blocked params not supported with merge_dims=True"
        if state.get('low_rank_side') is not None:
            assert not merge_dims, "Low-rank params not supported with merge_dims=True"
    
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1-state['shampoo_beta'])
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                                new_grad,
                                new_grad,
                                dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                            )
                        state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        if state.get('is_head_blocked') and idx == 0:
                            n_heads = state['n_heads']
                            head_dim = sh // n_heads
                            for h in range(n_heads):
                                g_h = grad[h * head_dim:(h + 1) * head_dim]  # (head_dim, d_in)
                                # Outer product over the head_dim slice; contracts the d_in axis
                                outer = torch.tensordot(g_h, g_h, dims=[list(range(1, g_h.dim()))] * 2)
                                state['GG'][idx][
                                    h * head_dim:(h + 1) * head_dim,
                                    h * head_dim:(h + 1) * head_dim
                                ].lerp_(outer, 1 - state['shampoo_beta'])
                        else:
                            outer_product = torch.tensordot(
                                    grad,
                                    grad,
                                    # Contracts across all dimensions except for k.
                                    dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                                )
                            state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
                     
        if state['Q'] is None:
            t0 = time.time()
            state['Q'] = self.get_orthogonal_matrix(state['GG'], state)
            self._last_eig_time += time.time() - t0
        
        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            t0 = time.time()
            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
            self._last_eig_time += time.time() - t0      

        if state["step"] > 0:
            state["exp_avg"] = self.project(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim) 
        
        if state.get('low_rank_side') is not None and state['step'] < 2:
            side = state['low_rank_side']
            print(f"[DEBUG] {state['param_name']} low-rank: "
                f"GG[{side}].shape={state['GG'][side].shape}, "
                f"Q[{side}].shape={state['Q'][side].shape}")
        

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [1]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
                
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        

    def get_orthogonal_matrix(self, mat, state=None):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for i, m in enumerate(matrix):
            if len(m) == 0:
                final.append([])
                continue

            if state is not None and state.get('is_head_blocked') and i == 0:
                Q, _ = blockdiag_eigh(m, state['n_heads'])
                Q = torch.flip(Q, [1])

            elif state is not None and state.get('low_rank_side') == i:
                Q, _ = truncated_eigh(m, state['low_rank_k'])

            else:
                m_sym = 0.5 * (m + m.T)

                # for debugging:
                # has_nan = torch.isnan(m_sym).any().item()
                # has_inf = torch.isinf(m_sym).any().item()
                # diag_min = m_sym.diag().min().item()
                # diag_max = m_sym.diag().max().item()
                # fro_norm = m_sym.norm().item()

                # if has_nan or has_inf:
                #     param_name = state.get('param_name', '?') if state else '?'
                #     step = state.get('step', '?') if state else '?'
                #     print(f"[CRITICAL] Non-finite GG for {param_name} step {step}: "
                #         f"nan={has_nan} inf={has_inf}")
                #     torch.save(m, f'/tmp/failed_GG_{param_name.replace(".", "_")}_step{step}.pt')

                # if not torch.isfinite(m_sym).all():
                #     m_sym = torch.nan_to_num(m_sym, nan=0.0, posinf=0.0, neginf=0.0)


                # ridge = m_sym.diag().abs().mean().clamp(min=1e-2) * 1e-6 + 1e-8  # stays on GPU
                # eye = torch.eye(m.shape[0], device=m.device)
                
                try:
                    _, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device))
                    # _, Q = torch.linalg.eigh(m_sym + ridge * eye)
                    # cuSOLVER can silently return NaN without raising — check explicitly
                    # if not torch.isfinite(Q).all():
                    #     raise RuntimeError("eigh fp32 silent NaN")
                except (torch._C._LinAlgError, RuntimeError) as e:
                    param_name = state.get('param_name', '?') if state else '?'
                    step = state.get('step', '?') if state else '?'
                    print(f"[WARN] eigh fallback to fp64 for {param_name} step {step}: {e}")
                    # print(f"  matrix stats: norm={fro_norm:.3e}, diag_min={diag_min:.3e}, "
                    #     f"diag_max={diag_max:.3e}, mean_eig={mean_eig:.3e}, ridge={ridge:.3e}")
                    # torch.save(m, f'/tmp/failed_eigh_{param_name.replace(".", "_")}_step{step}.pt')
                    # m64 = m_sym.to(torch.float64)
                    # ridge64 = 1e-4 * (m64.diag().abs().mean().item() + 1e-30)
                    # eye64 = torch.eye(m.shape[0], device=m.device, dtype=torch.float64)
                    # _, Q = torch.linalg.eigh(m64 + ridge64 * eye64)
                    # if not torch.isfinite(Q).all():
                    #     print(f"[CRITICAL] fp64 eigh also failed for {param_name}, using identity")
                    #     Q = torch.eye(m.shape[0], device=m.device, dtype=torch.float64)
                    _, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device))
                    Q = Q.to(m.dtype)
                    self._eigh_fp32_failures += 1

                Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final
        

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration 
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state['GG']
        orth_list = state['Q']
        is_head_blocked = state.get('is_head_blocked')
        low_rank_side = state.get('low_rank_side')

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
        
        orig_shape = state['exp_avg_sq'].shape
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']
            
        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            if len(m)==0:
                final.append([])
                continue
            
            # Low-rank side: re-run truncated eigh (can't do QR on a (d,k) basis cleanly)
            # but only this one dimension; the other side still gets cheap QR
            if low_rank_side is not None and ind == low_rank_side:
                est_eig = torch.diag(o.T @ m @ o)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
                # run expensive truncated_eigh every 4 normal updates
                if state['step'] % (state['precondition_frequency'] * 1) == 0:      # effective frequency for truncated params
                    Q, _ = truncated_eigh(m, state['low_rank_k'])                   # naturally sorted descending
                else:
                    # run cheap QR on existing k-column basis
                    o = o[:, sort_idx]
                    power_iter = m @ o  # o is (d, k)
                    Q, _ = torch.linalg.qr(power_iter)
                if not float_data:
                    Q = Q.to(original_device).type(original_type)
                final.append(Q)
                continue
            
            # Head-blocked side: must use blockdiag_eigh
            if is_head_blocked and ind == 0:
                Q, _ = blockdiag_eigh(m, state['n_heads'])
                Q = torch.flip(Q, [1])
                if not float_data:
                    Q = Q.to(original_device).type(original_type)
                final.append(Q)
                continue
            
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:,sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        
        if merge_dims:
            if self._data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
                
        state['exp_avg_sq'] = exp_avg_sq
        return final
    
    