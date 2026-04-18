import math
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adam_mini(torch.optim.Optimizer):
    def __init__(
            self,
            named_parameters: Iterable[Tuple[str, nn.Parameter]],
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            *,
            model_sharding: bool = None,
            dim: int = 2048,
            n_heads: int = 32,
            n_kv_heads: Optional[int] = None,
            verbose=True,
            scalar_mlp_proj_after_step: int = 0,    # MLP down-proj: per-neuron → scalar
            per_head_qk_after_step: int = 20,   # Q/K: scalar → per-head
            per_head_v_after_step: int = int(1e18),  # V: scalar → per-head (default: never)
            scalar_qkv: bool = False,  # single shared v_t for Q+K+V combined
    ):

        '''
        This is the official implementation of Adam-mini (version 1.1.1),
        modified to support switching MLP contraction (down-projection) layers
        from per-neuron v_t to a single scalar v_t at a configurable step,
        and to support a shared scalar v_t across Q, K, V for both fused
        (e.g. NeoX query_key_value) and separate (e.g. nanoGPT wq/wk/wv or
        LLaMA q_proj/k_proj/v_proj) projection layouts.

        Paper: [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793).

        Github repo: https://github.com/zyushun/Adam-mini

        Arguments:
            named_parameters ('Iterable[Tuple[str, nn.Parameter]]'): Iterable of named parameters to optimize or dictionaries defining parameter groups. Usually set to model.named_parameters()

            lr (`float`, *optional*, defaults to 0.001): The learning rate to use.

            betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`): Same as Adam's betas parameters (b1, b2).

            eps (`float`, *optional*, defaults to 1e-06): Same as Adam's epsilon for numerical stability.

            weight_decay (`float`, *optional*, defaults to 0.0): Decoupled weight decay to apply.

            model_sharding (`bool`, *optional*, defaults to None): Set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise. Due to the historical reason, this argument is deprecated since version 1.0.2. We will assume that model parallelism is always used. We will remove this argument in the future version.

            dim (`int`, *optional*, defaults to 2048): Dimension for hidden features. Can be left unspecified if training non-transformer models.

            n_heads (`int`, *optional*, defaults to 32): Number of attention heads. Can be left unspecified if training non-transformer models.

            n_kv_heads (`int`, *optional*, defaults to None): Number of heads for Key and Value. Or equivalently, number of query groups in Group Query Attention. Also known as "n_query_groups". If not specified, it will be equal to n_head. Can be left unspecified if training non-transformer models.

            verbose (`bool`, *optional*, defaults to True): Print all the logs if true.

            scalar_mlp_proj_after_step (`int`, *optional*, defaults to 0): Training step at which MLP
                contraction (down-projection) layers switch from per-neuron v_t to a single scalar v_t.
                The per-neuron vmean is collapsed to a scalar via averaging to preserve EMA history.

            per_head_qk_after_step (`int`, *optional*, defaults to 20): Training step at which Q and K
                layers switch from a single shared v_t across all heads to per-head v_t. Ignored when
                scalar_qkv=True.

            per_head_v_after_step (`int`, *optional*, defaults to ~never): Training step at which V
                layers switch from a single shared v_t across all kv-heads to per-kv-head v_t. Ignored
                when scalar_qkv=True.

            scalar_qkv (`bool`, *optional*, defaults to False): If True, use a single shared scalar v_t
                across Q, K, and V for each attention layer, for the full run. Supports both fused QKV
                layouts (e.g. NeoX "query_key_value") via an inline vmean_qkv, and separate projection
                layouts (e.g. nanoGPT wq/wk/wv, LLaMA q_proj/k_proj/v_proj) via a post-step average
                across the matched (Q, K, V) triple. Per-head transitions for Q/K and V are suppressed
                when this is True.
        Example:

        ```python
        optimizer = Adam_mini(
                    named_parameters = model.named_parameters(),
                    lr = lr,
                    betas = (beta1,beta2),
                    eps = eps,
                    weight_decay = weight_decay,
                    dim = model_config.dim,
                    n_heads = model_config.n_heads,
                    n_kv_heads = model_config.n_kv_heads,
                    )
        ```

        '''
        self.named_parameters = named_parameters
        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        self.world_size = torch.cuda.device_count()
        self.verbose = verbose
        self.check_block_name = True
        self.head_numel = self.dim * self.dim // self.n_heads
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        if model_sharding is not None and verbose:
            print(
                "Warning by Adam-mini: model_sharding is deprecated since version 1.0.2. This argument is always set True. We will remove this argument in the future version.")


        # Embedding layer. Use one lr per token
        self.embd_names = {"embed", "embd", "wte"}
        # Output layers. Use one lr per token
        self.output_names = {"lm_head", "output", "final_layer"}
        # Query and Keys. User one lr per head
        self.wqk_names = {"k_proj", "q_proj", "wq", "wk", "query", "key"}
        # Values. Use one lr per head (same granularity as Q/K).
        # Shared scalar v_t across all heads before per_head_after_step, then per-head.
        self.wv_names = {"v_proj", "wv", "value"}
        # attn_proj. Use one lr per neuron
        self.attn_proj_names = {"o_proj", "wo", "attn.proj", "attention.dense"}
        # MLPs. Use one lr per neuron
        self.mlp_names = {"feed_forward", "linear", "mlp"}
        # MLP cont: per-neuron → scalar
        self.mlp_contraction_names = {"dense_4h_to_h"}
        self.scalar_mlp_proj_after_step = scalar_mlp_proj_after_step
        # Q/K: scalar → per-head
        self.per_head_qk_after_step = per_head_qk_after_step
        # V: scalar → per-head
        self.per_head_v_after_step = per_head_v_after_step
        # single shared v_t for Q+K+V combined
        self.scalar_qkv = scalar_qkv
        # Blocks that use Adam: bias terms
        self.adam_block_names = {"bias"}

        optim_groups = []

        for param_name, param in named_parameters:
            param_name = param_name.lower()
            if not param.requires_grad:
                continue
            if verbose:
                print('Adam-mini found the param block with name:', param_name, param.size())
            state = {}
            state["name"] = param_name
            state["params"] = param
            if "norm" in param_name or "ln" in param_name or "bias" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay

            optim_groups.append(state)

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

        # Build Q/K pairs for post-step vmean synchronization (separate Q/K projection case,
        # e.g. LLaMA-style q_proj/k_proj). For fused QKV (NeoX "query_key_value"), the shared
        # vmean is computed inline in the fused QKV block and this loop is a no-op.
        layer_to_qk = {}
        for group in self.param_groups:
            name = group["name"]
            if not any(wqk_name in name for wqk_name in self.wqk_names):
                continue
            key, role = self._get_qk_layer_key(name)
            if key is None:
                continue
            layer_to_qk.setdefault(key, {})[role] = group["params"][0]
        self._qk_pairs = [
            (v["q"], v["k"])
            for v in layer_to_qk.values()
            if "q" in v and "k" in v
        ]
        if verbose:
            print(f"Adam-mini found {len(self._qk_pairs)} separate Q/K pairs for shared per-head vmean.")

        # Build Q/K/V triples for post-step vmean synchronization when scalar_qkv=True on
        # separate projection layouts (nanoGPT wq/wk/wv, LLaMA q_proj/k_proj/v_proj).
        # Fused QKV handles scalar_qkv inline via vmean_qkv, so this loop is irrelevant there.
        layer_to_qkv = {}
        for group in self.param_groups:
            name = group["name"]
            is_qk_name = any(wqk_name in name for wqk_name in self.wqk_names)
            is_v_name = any(wv_name in name for wv_name in self.wv_names)
            if not (is_qk_name or is_v_name):
                continue
            key, role = self._get_qkv_layer_key(name)
            if key is None:
                continue
            layer_to_qkv.setdefault(key, {})[role] = group["params"][0]
        self._qkv_triples = [
            (d["q"], d["k"], d["v"])
            for d in layer_to_qkv.values()
            if {"q", "k", "v"}.issubset(d.keys())
        ]
        if verbose:
            print(f"Adam-mini found {len(self._qkv_triples)} separate Q/K/V triples for shared scalar_qkv vmean.")

    def _get_qk_layer_key(self, name):
        """Return (layer_key, 'q'|'k') for a separate Q or K param, or (None, None) for fused/other.

        Handles q_proj/k_proj, wq/wk, and lone 'query'/'key' keywords.
        Fused names containing both 'query' and 'key' (e.g. 'query_key_value') are skipped.
        """
        for q_kw, k_kw in [("q_proj", "k_proj"), ("wq", "wk")]:
            if q_kw in name:
                return name.replace(q_kw, "\x00"), "q"
            if k_kw in name:
                return name.replace(k_kw, "\x00"), "k"
        # Generic 'query'/'key': skip if both appear (fused QKV weight).
        q_in = "query" in name
        k_in = "key" in name
        if q_in and not k_in:
            return name.replace("query", "\x00"), "q"
        if k_in and not q_in:
            return name.replace("key", "\x00"), "k"
        return None, None

    def _get_qkv_layer_key(self, name):
        """Return (layer_key, 'q'|'k'|'v') for a separate Q, K, or V param, or (None, None).

        Handles q_proj/k_proj/v_proj, wq/wk/wv, and lone 'query'/'key'/'value' keywords.
        Fused names containing more than one of these keywords (e.g. 'query_key_value') are skipped
        so they don't collide with the fused-QKV branch that handles scalar_qkv inline.
        """
        for q_kw, k_kw, v_kw in [("q_proj", "k_proj", "v_proj"), ("wq", "wk", "wv")]:
            if q_kw in name:
                return name.replace(q_kw, "\x00"), "q"
            if k_kw in name:
                return name.replace(k_kw, "\x00"), "k"
            if v_kw in name:
                return name.replace(v_kw, "\x00"), "v"
        q_in = "query" in name
        k_in = "key" in name
        v_in = "value" in name
        num_present = int(q_in) + int(k_in) + int(v_in)
        if num_present != 1:
            # Skip fused weights (two or more of the three keywords) or unrelated names.
            return None, None
        if q_in:
            return name.replace("query", "\x00"), "q"
        if k_in:
            return name.replace("key", "\x00"), "k"
        if v_in:
            return name.replace("value", "\x00"), "v"
        return None, None

    def count_block(self):
        count_embd = 0
        count_output = 0
        count_wqk = 0
        count_wv = 0
        count_attn_proj = 0
        count_mlp = 0
        count_mlp_contraction = 0
        for group in self.param_groups:
            name = group["name"]
            if "bias" in name:
                continue
            if any(embd_name in name for embd_name in self.embd_names):
                count_embd += 1
            if any(output_name in name for output_name in self.output_names):
                count_output += 1
            if any(wqk_name in name for wqk_name in self.wqk_names):
                count_wqk += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
            if any(wv_name in name for wv_name in self.wv_names):
                count_wv += 1
            if any(attn_proj_name in name for attn_proj_name in self.attn_proj_names):
                count_attn_proj += 1
            if any(contraction_name in name for contraction_name in self.mlp_contraction_names):
                count_mlp_contraction += 1
            elif any(mlp_name in name for mlp_name in self.mlp_names):
                count_mlp += 1
        if self.verbose:
            print(
                f'Adam-mini found {count_embd} embedding layers, {count_output} output layers; {count_wqk} Querys and Keys;  {count_wv} Values;  {count_attn_proj} attn_proj;  {count_mlp} MLPs;  {count_mlp_contraction} MLP contraction layers (switching to scalar v_t at step {self.scalar_mlp_proj_after_step}); Q/K/V use shared v_t before step {self.per_head_qk_after_step}, then per-head; V uses shared v_t before step {self.per_head_v_after_step}, then per-head; scalar_qkv={self.scalar_qkv};')

        if count_embd == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the keywords in the name of your embedding layer'). ")
        if count_output == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.output_names.add('the keywords in the  name of your output layer').  Please ignore this warning if you are using weight-tying.")
        if count_wqk == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding two additional lines of code: optimizer.wqk_names.add('the keywords in the  name of your Query' ); optimizer.wqk_names.add('the keywords in the  name of your Key'). ")

        if count_wv == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Value found. If you are training Transformers, please check the name of your Value in attention blocks and manually add them to 'self.wv_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.wv_names.add('the keywords in the  name of your Value' ). ")

        if count_attn_proj == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No attn_proj found. If you are training Transformers, please check the name of your attn_proj in attention blocks and manually add them to 'self.attn_proj_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your attn_proj' ). ")

        if count_mlp == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No MLP found. If you are training Transformers, please check the name of your MLP in attention blocks and manually add them to 'self.mlp_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your MLP' ). ")

        if (count_output + count_embd + count_wqk + count_wv + count_attn_proj + count_mlp == 0) and self.verbose:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")

    @torch.no_grad()
    def step(self, closure=None):
        if self.check_block_name:
            self.count_block()
            self.check_block_name = False

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:
                state = self.state[p]
                if any(adam_block_name in name for adam_block_name in
                       self.adam_block_names):  # for bias terms
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    grad = p.grad
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)
                elif "query_key_value" in name or "qkv" in name:  # fused QKV (NeoX/Pythia layout)
                    if p.grad is None:
                        continue

                    head_dim = self.dim // self.n_heads

                    # NeoX/Pythia layout is interleaved by head: [q_0, k_0, v_0, q_1, k_1, v_1, ...]
                    # Reshape to (n_heads, 3, head_dim, hidden_dim) to safely slice them.
                    grad = p.grad
                    grad_reshaped = grad.view(self.n_heads, 3, head_dim, -1)

                    # Flatten the inner dimensions back to (n_heads, head_numel) for math
                    grad_q = grad_reshaped[:, 0, :, :].reshape(self.n_heads, -1)
                    grad_k = grad_reshaped[:, 1, :, :].reshape(self.n_heads, -1)
                    grad_v = grad_reshaped[:, 2, :, :].reshape(self.n_heads, -1)

                    if len(state) == 0:
                        state["m"]              = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"]           = 0
                        state["using_per_head_qk"] = False
                        state["using_per_head_v"]  = False

                        state["vmean_qk"] = torch.zeros_like(grad_q[0:1, 0:1], memory_format=torch.preserve_format)
                        state["vmean_v"]  = torch.zeros_like(grad_v[0:1, 0:1], memory_format=torch.preserve_format)
                        state["vmean_qkv"] = torch.zeros_like(grad_q[0:1, 0:1], memory_format=torch.preserve_format)

                    # ---- checkpoint migration ----
                    if "using_per_head_qk" not in state:
                        already = state["vmean_qk"].size(0) > 1
                        state["using_per_head_qk"] = already
                        if not already and state["step"] >= self.per_head_qk_after_step:
                            state["vmean_qk"] = state["vmean_qk"].expand(self.n_heads, 1).clone()
                            state["using_per_head_qk"] = True

                    if "using_per_head_v" not in state:
                        already = state["vmean_v"].size(0) > 1
                        state["using_per_head_v"] = already
                        if not already and state["step"] >= self.per_head_v_after_step:
                            state["vmean_v"] = state["vmean_v"].expand(self.n_heads, 1).clone()
                            state["using_per_head_v"] = True

                    # ---- scalar_qkv fallback for old checkpoints ----
                    if self.scalar_qkv and "vmean_qkv" not in state:
                        if "vmean_qk" in state and "vmean_v" in state:
                            # Weight by number of matrices: Q and K both feed vmean_qk, V feeds vmean_v
                            state["vmean_qkv"] = (
                                (state["vmean_qk"].mean() * 2 + state["vmean_v"].mean()) / 3.0
                            ).view(1, 1)
                        else:
                            state["vmean_qkv"] = torch.zeros_like(
                                grad_q[0:1, 0:1], memory_format=torch.preserve_format)

                    if state["step"] == 0 and self.verbose:
                        print(f"[DEBUG] Adam-mini QKV init: name={name}, "
                              f"vmean_qk shape={state['vmean_qk'].shape}, "
                              f"vmean_v shape={state['vmean_v'].shape}, "
                              f"QK threshold={self.per_head_qk_after_step}, "
                              f"V threshold={self.per_head_v_after_step}")

                    state["step"] += 1

                    # ---- per-head transitions at threshold step ----
                    if not state["using_per_head_qk"] and state["step"] >= self.per_head_qk_after_step:
                        state["vmean_qk"] = state["vmean_qk"].expand(self.n_heads, 1).clone()
                        state["using_per_head_qk"] = True
                        if self.verbose:
                            print(f"[DEBUG] Adam-mini: vmean_qk -> per-head at step {state['step']}, name={name}")

                    if not state["using_per_head_v"] and state["step"] >= self.per_head_v_after_step:
                        state["vmean_v"] = state["vmean_v"].expand(self.n_heads, 1).clone()
                        state["using_per_head_v"] = True
                        if self.verbose:
                            print(f"[DEBUG] Adam-mini: vmean_v -> per-head at step {state['step']}, name={name}")

                    # ---- weight decay and first moment (always needed) ----
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                    # ---- vmean update and stepsize ----
                    if self.scalar_qkv:
                        if state["step"] == 1 and self.verbose:
                            print(f"[DEBUG] Adam-mini: Triggered single scalar_qkv branch for {name}")
                        tmp_lr = torch.mean((grad_q ** 2 + grad_k ** 2 + grad_v ** 2) / 3.0)
                        state["vmean_qkv"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        h = (state["vmean_qkv"].sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = (1 / bias_correction_1) / h
                        step_qk_b = stepsize.view(1, 1, 1, 1)
                        step_v_b  = stepsize.view(1, 1, 1, 1)

                    else:
                        if state["using_per_head_qk"]:
                            tmp_lr_qk = torch.mean((grad_q ** 2 + grad_k ** 2) * 0.5, dim=1, keepdim=True)
                        else:
                            tmp_lr_qk = torch.mean((grad_q ** 2 + grad_k ** 2) * 0.5)

                        if state["using_per_head_v"]:
                            tmp_lr_v = torch.mean(grad_v ** 2, dim=1, keepdim=True)
                        else:
                            tmp_lr_v = torch.mean(grad_v ** 2)

                        state["vmean_qk"].mul_(beta2).add_(tmp_lr_qk, alpha=1 - beta2)
                        state["vmean_v"].mul_(beta2).add_(tmp_lr_v,   alpha=1 - beta2)

                        h_qk = (state["vmean_qk"].sqrt() / bias_correction_2_sqrt).add_(eps)
                        h_v  = (state["vmean_v"].sqrt()  / bias_correction_2_sqrt).add_(eps)
                        step_qk = (1 / bias_correction_1) / h_qk
                        step_v  = (1 / bias_correction_1) / h_v
                        # Reshape step sizes so they broadcast across (n_heads, 1, head_dim, hidden_dim)
                        step_qk_b = step_qk.view(-1, 1, 1, 1) if state["using_per_head_qk"] else step_qk.view(1, 1, 1, 1)
                        step_v_b  = step_v.view(-1, 1, 1, 1)  if state["using_per_head_v"]  else step_v.view(1, 1, 1, 1)

                    m_reshaped = state["m"].view(self.n_heads, 3, head_dim, -1)
                    update_q = m_reshaped[:, 0:1, :, :] * step_qk_b
                    update_k = m_reshaped[:, 1:2, :, :] * step_qk_b
                    update_v = m_reshaped[:, 2:3, :, :] * step_v_b
                    # Concatenate back to the (n_heads, 3, head_dim, hidden) structure
                    update     = torch.cat([update_q, update_k, update_v], dim=1).view_as(p)

                    p.add_(update, alpha=-lr)
                elif any(wqk_name in name for wqk_name in self.wqk_names):  # separate Q or K projection
                    if p.grad is None:
                        continue
                    head_numel = self.head_numel
                    if len(state) == 0:
                        m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(0)  # heads per gpu
                        state["step"] = 0
                        state["using_per_head_qk"] = False
                        # Scalar vmean shared across all heads until per_head_after_step.
                        # NOTE: slice of state["m"] keeps DTensor type for DTensor parameters.
                        state["vmean"] = torch.zeros_like(state["m"][0:1, 0:1],
                                                          memory_format=torch.preserve_format)

                    # Migration: infer flag for checkpoints saved before this feature.
                    if "using_per_head_qk" not in state:
                        if state["vmean"].size(0) > 1:
                            # Old checkpoint already had per-head vmean.
                            state["using_per_head_qk"] = True
                        elif state["step"] >= self.per_head_qk_after_step:
                            state["vmean"] = state["vmean"].expand(state["head_per_gpu"], 1).clone()
                            state["using_per_head_qk"] = True
                        else:
                            state["using_per_head_qk"] = False

                    grad = p.grad
                    head_per_gpu = state["head_per_gpu"]
                    grad = grad.view(head_per_gpu, head_numel)

                    state["step"] += 1

                    # At the threshold step, broadcast scalar vmean to per-head,
                    # preserving accumulated EMA history rather than reinitializing to zero.
                    # When scalar_qkv is True we never transition — Q, K, V stay on a single
                    # shared scalar v_t synchronized in the post-step loop below.
                    if (not self.scalar_qkv
                            and not state["using_per_head_qk"]
                            and state["step"] >= self.per_head_qk_after_step):
                        state["vmean"] = state["vmean"].expand(head_per_gpu, 1).clone()
                        state["using_per_head_qk"] = True

                    if state["using_per_head_qk"]:
                        tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                    else:
                        # One shared v_t for all heads: mean over every element.
                        tmp_lr = torch.mean(grad * grad)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    # h is (1,1) or (head_per_gpu,1) — both broadcast correctly to m's shape.
                    stepsize = (1 / bias_correction_1) / h
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                elif any(contraction_name in name for contraction_name in self.mlp_contraction_names):
                    # MLP contraction (down-projection): per-neuron v_t before
                    # scalar_mlp_proj_after_step, then a single scalar v_t afterwards.
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(0)
                        state["vmean"] = torch.zeros_like(
                            state["m"][0:state["neuron_per_gpu"], 0:1],
                            memory_format=torch.preserve_format,
                        )
                        state["using_scalar"] = False

                    # Migration: infer flag for checkpoints saved before this feature.
                    # Collapse per-neuron vmean to a scalar by averaging, preserving EMA history.
                    if "using_scalar" not in state:
                        if state["step"] >= self.scalar_mlp_proj_after_step:
                            state["vmean"] = state["vmean"].mean()
                            state["using_scalar"] = True
                        else:
                            state["using_scalar"] = False

                    state["step"] += 1

                    # At the threshold step, collapse per-neuron vmean -> scalar by averaging,
                    # preserving accumulated EMA history rather than reinitializing to zero.
                    if not state["using_scalar"] and state["step"] >= self.scalar_mlp_proj_after_step:
                        state["vmean"] = state["vmean"].mean()
                        state["using_scalar"] = True

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    neuron_per_gpu = state["neuron_per_gpu"]
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                    if state["using_scalar"]:
                        tmp_lr = torch.mean(grad * grad)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * stepsize
                        update.mul_(lr)
                        p.add_(-update)
                    else:
                        tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                        stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                        update = (state["m"] * stepsize).view(p.size())
                        update.mul_(lr)
                        p.add_(-update)
                elif any(wv_name in name for wv_name in self.wv_names):  # Values: per-head
                    if p.grad is None:
                        continue
                    kv_head_numel = self.head_numel  # head_dim * dim = dim^2/n_heads
                    if len(state) == 0:
                        m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, kv_head_numel)
                        state["kv_head_per_gpu"] = state["m"].size(0)  # kv heads per gpu
                        state["step"] = 0
                        state["using_per_head_v"] = False
                        # Scalar vmean shared across all kv heads until per_head_after_step.
                        state["vmean"] = torch.zeros_like(state["m"][0:1, 0:1],
                                                          memory_format=torch.preserve_format)

                    # Migration: infer flag for checkpoints saved before this feature.
                    if "using_per_head_v" not in state:
                        if state["vmean"].size(0) > 1:
                            # Old checkpoint had per-neuron vmean; collapse to per-kv-head by averaging.
                            kv_head_per_gpu_mig = state["kv_head_per_gpu"]
                            n_rows = state["vmean"].size(0)
                            rows_per_head = n_rows // kv_head_per_gpu_mig
                            state["vmean"] = state["vmean"].view(kv_head_per_gpu_mig, rows_per_head, 1).mean(dim=1)
                            state["using_per_head_v"] = True
                        elif state["step"] >= self.per_head_v_after_step:
                            state["vmean"] = state["vmean"].expand(state["kv_head_per_gpu"], 1).clone()
                            state["using_per_head_v"] = True
                        else:
                            state["using_per_head_v"] = False

                    grad = p.grad
                    kv_head_per_gpu = state["kv_head_per_gpu"]
                    grad = grad.view(kv_head_per_gpu, kv_head_numel)

                    state["step"] += 1

                    # At the threshold step, broadcast scalar vmean to per-kv-head.
                    # When scalar_qkv is True, suppress this transition — V stays on the
                    # shared scalar v_t, which is synced with Q and K in the post-step loop.
                    if (not self.scalar_qkv
                            and not state["using_per_head_v"]
                            and state["step"] >= self.per_head_v_after_step):
                        state["vmean"] = state["vmean"].expand(kv_head_per_gpu, 1).clone()
                        state["using_per_head_v"] = True

                    if state["using_per_head_v"]:
                        tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                    else:
                        # One shared v_t for all kv heads: mean over every element.
                        tmp_lr = torch.mean(grad * grad)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    # h is (1,1) or (kv_head_per_gpu,1) — both broadcast correctly to m's shape.
                    stepsize = (1 / bias_correction_1) / h
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                elif any(embd_name in name for embd_name in self.embd_names) or any(
                        output_name in name for output_name in self.output_names) or any(
                        mlp_name in name for mlp_name in self.mlp_names) or any(
                        attn_proj_name in name for attn_proj_name in self.attn_proj_names):
                    if p.grad is None:
                        continue
                    # neuron_numel = group["neuron_numel"] # assume grad is a matrix by default, so do not need this
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p.grad,
                                                      memory_format=torch.preserve_format)  # assume grad is a matrix by default, no need to view
                        # state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format).view(-1, neuron_numel)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(0)  # this is neuron per gpu
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # for standard tensor: state["vmean"] = torch.zeros(1, device=p.device)
                        # for DTensor: state["vmean"] = p.new_zeros(1)
                        # the following implementation unifies the above two lines
                        state["vmean"] = torch.zeros_like(state["m"][0:state["neuron_per_gpu"], 0:1],
                                                          memory_format=torch.preserve_format)

                    grad = p.grad  # .to(torch.float32)
                    neuron_per_gpu = state["neuron_per_gpu"]
                    # grad = grad.view(neuron_per_gpu, neuron_numel) # assume grad is a matrix by default, so no need to reshape
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                else:  # other blocks. By default, this is for LayerNorms. Sometimes it is also fine to put Value here
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel()).to(torch.float32).to(device)
                        reduced = False
                        if (self.world_size > 1):
                            tensor_list = [torch.zeros_like(block_numel) for _ in range(self.world_size)]

                            dist.all_gather(tensor_list, block_numel)
                            s = 0
                            block_numel = 0
                            for d in tensor_list:
                                if (d > 0):
                                    s = s + 1
                                block_numel = block_numel + d
                            if (s >= 2):
                                reduced = True

                        state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["reduced"] = reduced
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # For standard tensor: state["vmean"] = torch.zeros(1, device=p.device)
                        # For DTensor: state["vmean"] = p.new_zeros(1)
                        # the following implementation unifies the above two lines
                        state["vmean"] = torch.zeros_like(torch.sum(p * p), memory_format=torch.preserve_format)
                        state["block_numel"] = block_numel.item()
                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        grad = p.grad  # .to(torch.float32)
                        tmp_lr = torch.sum(grad * grad)

                    if (state["reduced"]):
                        # Force communication over GPUs when GPUs are available
                        if tmp_lr.device.type == 'cpu':
                            # Move the tensor to the current GPU device
                            tmp_lr_gpu = tmp_lr.to(torch.cuda.current_device())

                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                            # Move the result back to the CPU tensor
                            tmp_lr.copy_(tmp_lr_gpu.cpu())
                        else:
                            # Tensor is already on GPU, use NCCL backend
                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if (p.grad is None):
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)

        # Post-step sync across attention projections.
        # - scalar_qkv=True: average vmean across matched (Q, K, V) triples so all three share
        #   a single scalar v_t. This covers the separate-projection case (nanoGPT wq/wk/wv,
        #   LLaMA q_proj/k_proj/v_proj). Fused QKV handles scalar_qkv inline via vmean_qkv.
        # - scalar_qkv=False: original behavior — average vmean across matched Q/K pairs only.
        #   In the scalar phase this averages two scalars; in the per-head phase it averages
        #   per-head vmean across the Q and K parameters.
        if self.scalar_qkv:
            for q_p, k_p, v_p in self._qkv_triples:
                q_state = self.state[q_p]
                k_state = self.state[k_p]
                v_state = self.state[v_p]
                if ("vmean" not in q_state
                        or "vmean" not in k_state
                        or "vmean" not in v_state):
                    continue
                # With scalar_qkv suppressing per-head transitions, all three should be (1,1).
                # Guard against shape mismatches anyway (e.g. GQA with n_heads != n_kv_heads,
                # or a checkpoint loaded with per-head vmean already in place).
                if (q_state["vmean"].shape != k_state["vmean"].shape
                        or k_state["vmean"].shape != v_state["vmean"].shape):
                    continue
                avg = (q_state["vmean"] + k_state["vmean"] + v_state["vmean"]).mul_(1.0 / 3.0)
                q_state["vmean"].copy_(avg)
                k_state["vmean"].copy_(avg)
                v_state["vmean"].copy_(avg)
        else:
            for q_p, k_p in self._qk_pairs:
                q_state = self.state[q_p]
                k_state = self.state[k_p]
                if "vmean" not in q_state or "vmean" not in k_state:
                    continue
                if q_state["vmean"].shape != k_state["vmean"].shape:
                    continue  # GQA with n_heads != n_kv_heads: skip
                avg = (q_state["vmean"] + k_state["vmean"]).mul_(0.5)
                q_state["vmean"].copy_(avg)
                k_state["vmean"].copy_(avg)

        return loss
