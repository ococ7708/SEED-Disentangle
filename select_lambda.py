"""Source-held-out lambda line search (20261110 theory doc §4.1).

From a common P0' checkpoint (frozen teacher source), split the source subjects into
A (fit) and V (held-out), compute the identity-CE gradient d w.r.t. the
representation parameters (encoder + S) on a batch from A, then evaluate the
held-out source task loss

    J(lambda) = L_V(theta_b + eta*lambda*d)

for a few lambda candidates (including 0). This is the document's minimal
"short-term update gain" selection: it is a single-step line search, not a full
training sweep. The best non-negative lambda minimizes J(lambda).

Read-only: loads a checkpoint, never writes model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

WORK = r"D:/大学/脑机接口/SEED_CGB_DG"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seed_family_data import load_records, make_fold  # noqa: E402


def _prototypes(cfg, device):
    import hashlib
    from pathlib import Path
    from transformers import CLIPModel, CLIPTokenizer
    names = cfg['data']['class_names']
    cache = Path(WORK) / 'artifacts' / 'seed_family_prototypes'
    cache.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(json.dumps([cfg['clip_root'], names]).encode()).hexdigest()[:16]
    path = cache / f'{token}.npy'
    if path.exists():
        p = torch.tensor(np.load(path, allow_pickle=False), device=device)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(cfg['clip_root'], local_files_only=True)
        text_model = CLIPModel.from_pretrained(cfg['clip_root'], local_files_only=True).to(device).eval()
        batch = tokenizer(names, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            p = text_model.get_text_features(**batch)
            if not isinstance(p, torch.Tensor):
                p = p.pooler_output
        np.save(path, p.detach().cpu().numpy())
        del text_model
    return F.normalize(p.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fold', required=True, help='P0 run dir containing source_best.pt/run.json')
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.005, 0.01, 0.02])
    ap.add_argument('--eta', type=float, default=None, help='step size (default cfg train lr)')
    ap.add_argument('--av-split', type=int, default=2, help='number of held-out source subjects (V)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from train_seed_family_disentangle import read_config, encode, build_models, fused_logits
    cfg = read_config(args.config)
    device = torch.device(args.device)
    fold = os.path.abspath(args.fold)
    manifest = json.loads(open(os.path.join(fold, 'run.json'), encoding='utf-8').read())
    sessions = manifest['sessions']; target = manifest['target']
    records, subjects = load_records(cfg, sessions)
    parts, all_sources, _ = make_fold(cfg, records, target)

    # A/V split by source subject (V = last few).
    v_ids = all_sources[-args.av_split:]
    a_ids = all_sources[:-args.av_split]
    prototypes = _prototypes(cfg, device)
    model, metric, disent = build_models(cfg, len(all_sources), prototypes, device)
    ckpt = None
    for name in ('source_best.pt', 'target_best.pt', 'last.pt'):
        cand = os.path.join(fold, name)
        if os.path.exists(cand):
            ckpt = cand
            break
    if ckpt is None:
        raise FileNotFoundError(f'{fold}: no source_best/target_best/last checkpoint')
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(st['model']); metric.load_state_dict(st['metric']); disent.load_state_dict(st['disentangler'])
    model.eval(); metric.eval(); disent.eval()
    centroid = st['centroids'].to(device)
    repr_params = list(model.cast_encoder.parameters()) + list(disent.shared.parameters())
    eta = args.eta or cfg['train']['lr']
    k = len(cfg['data']['class_names'])

    # one batch from A
    xs, ys, rs = [], [], []
    for j, sid in enumerate(a_ids):
        x = torch.tensor(parts['train'][sid][0], device=device)
        y = torch.tensor(parts['train'][sid][1], device=device)
        xs.append(x); ys.append(y); rs.append(torch.full_like(y, all_sources.index(sid)))
    x = torch.cat(xs); y = torch.cat(ys); r = torch.cat(rs)

    # identity CE gradient d on A (a single sub-batch)
    aux = [disent.identity, disent.specific, disent.specific_id, disent.t_identity]
    for p in [q for m in aux for q in m.parameters()]:
        p.requires_grad_(True)
    s_a, q_a = encode(model, disent, x)
    # Raw identity-CE gradient (no GRL sign flip): d = grad_theta L_id, as in Eq. 5.
    lid_terms = [F.cross_entropy(disent.identity[c](s_a[y == c]), r[y == c])
                 for c in range(k) if (y == c).any()]
    lid = torch.stack(lid_terms).mean()
    d = torch.autograd.grad(lid, repr_params, allow_unused=True)
    d = [torch.zeros_like(p) if g is None else g.clone() for p, g in zip(repr_params, d)]
    model.zero_grad(set_to_none=True)

    # held-out task loss L_V on V (fused logits)
    def lv():
        tot, n = 0.0, 0
        with torch.no_grad():
            for sid in v_ids:
                xv = torch.tensor(parts['train'][sid][0], device=device)
                yv = torch.tensor(parts['train'][sid][1], device=device)
                sv, _ = encode(model, disent, xv)
                logits = fused_logits(model, metric, disent, sv, centroid, cfg)
                tot += float(F.cross_entropy(logits, yv).item()) * len(xv)
                n += len(xv)
        return tot / max(n, 1)

    base = [p.detach().clone() for p in repr_params]
    results = []
    for lam in args.lambdas:
        with torch.no_grad():
            for p, b, g in zip(repr_params, base, d):
                p.copy_(b + eta * lam * g)
        results.append({'lambda': lam, 'L_V': lv()})
        with torch.no_grad():
            for p, b in zip(repr_params, base):
                p.copy_(b)
    best = min(results, key=lambda z: z['L_V'])
    report = {'fold': fold, 'eta': eta, 'A': a_ids, 'V': v_ids,
              'base_L_V': lv(), 'results': results, 'best_lambda': best['lambda']}
    out = os.path.join(WORK, 'experiments_dtrm', 'select_lambda_result.json')
    open(out, 'w', encoding='utf-8').write(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
