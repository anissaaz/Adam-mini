### vanilla SOAP logging

# import os
# import torch

# def save_preconditioners(optimizer, param_to_name, step, out_dir, tracked_layers=None):
#     import os, torch, re
#     os.makedirs(out_dir, exist_ok=True)
#     snapshot = {}
#     layer_pat = re.compile(r'transformer\.h\.(\d+)\.')
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             name = param_to_name.get(id(p))
#             if name is None:
#                 continue
#             m = layer_pat.search(name)
#             if m is None:
#                 continue  # skip embedding, ln_f, lm_head
#             layer_idx = int(m.group(1))
#             if tracked_layers is not None and layer_idx not in tracked_layers:
#                 continue
#             # Only track attention and MLP weights
#             if not any(k in name for k in ['attn.wq', 'attn.wk', 'attn.wv', 'attn.wo',
#                                             'mlp.c_fc', 'mlp.c_proj']):
#                 continue
#             state = optimizer.state.get(p, {})
#             if 'GG' not in state or 'Q' not in state:
#                 continue
#             snapshot[name] = {
#                 'shape': tuple(p.shape),
#                 'layer_idx': layer_idx,
#                 'step': int(state.get('step', 0)),
#                 'GG': [g.detach().float().cpu() if torch.is_tensor(g) and len(g) > 0 else None
#                     for g in state['GG']],
#                 'Q':  [q.detach().half().cpu() if torch.is_tensor(q) and len(q) > 0 else None
#                     for q in state['Q']],
#             }
#     path = os.path.join(out_dir, f'step_{step:07d}.pt')
#     torch.save(snapshot, path)
#     print(f"[precond] step {step}: saved {len(snapshot)} tensors -> {path}")


def save_preconditioners_truncated(optimizer, param_to_name, step, out_dir,
                                    tracked_layers=None):
    import os, torch, re
    os.makedirs(out_dir, exist_ok=True)
    snapshot = {}
    layer_pat = re.compile(r'transformer\.h\.(\d+)\.')
    target_kinds = ('attn.wq', 'attn.wk', 'attn.wv', 'attn.wo',
                    'mlp.c_fc', 'mlp.c_proj')

    def tensor_or_none(t, dtype):
        if torch.is_tensor(t) and t.numel() > 0:
            return t.detach().to(dtype).cpu()
        return None

    def list_to_cpu(lst, dtype):
        if lst is None:
            return None
        return [tensor_or_none(t, dtype) for t in lst]

    for group in optimizer.param_groups:
        for p in group['params']:
            name = param_to_name.get(id(p))
            if name is None:
                continue
            m = layer_pat.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            if tracked_layers is not None and layer_idx not in tracked_layers:
                continue
            if not any(k in name for k in target_kinds):
                continue
            state = optimizer.state.get(p, {})
            if 'Q' not in state:
                continue
            snapshot[name] = {
                'shape': tuple(p.shape),
                'layer_idx': layer_idx,
                'step': int(state.get('step', 0)),
                'streaming_ranks': state.get('streaming_ranks'),  # dict or None
                'F':    list_to_cpu(state.get('F'),    torch.float32),
                'Q':    list_to_cpu(state.get('Q'),    torch.float16),
                'eigs': list_to_cpu(state.get('eigs'), torch.float32),
                'GG':   list_to_cpu(state.get('GG'),   torch.float32),
            }
    path = os.path.join(out_dir, f'step_{step:07d}.pt')
    torch.save(snapshot, path)
    print(f"[precond-trunc] step {step}: saved {len(snapshot)} tensors -> {path}")