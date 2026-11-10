"""Config-driven CAST-e3 P0/P1/P2 LOSO runner; source-trial checkpoint selection."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from seed_family_data import load_records, make_fold
from conditional_disentangler import ConditionalDisentangler

_HERE = Path(__file__).resolve().parent
# Source layout puts this file under experiments_dtrm/; the standalone repo puts it at
# the repo root. PROJECT must be the directory that config-relative paths are based on.
PROJECT = _HERE.parent if _HERE.name == 'experiments_dtrm' else _HERE


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('cache_root',):
        cfg['data'][key] = str(resolve(cfg['data'][key]))
    cfg['sgda_root'] = str(resolve(cfg['sgda_root']))
    cfg['clip_root'] = str(resolve(cfg['clip_root']))
    if len(set(cfg['data']['class_names'])) != len(cfg['data']['class_names']):
        raise ValueError('class_names must be unique and ordered by integer label')
    if cfg['variant'] not in ('P0', 'P1', 'P2', 'P3'):
        raise ValueError('variant must be P0/P1/P2/P3')
    if cfg['data'].get('stride', 1) < 1 or cfg['model']['sample_length'] < 1:
        raise ValueError('stride/sample_length must be positive')
    return cfg


def prototype_vectors(cfg, device):
    from transformers import CLIPModel, CLIPTokenizer
    names = cfg['data']['class_names']
    cache = PROJECT / 'artifacts' / 'seed_family_prototypes'
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
    if p.shape != (len(names), 512) or not torch.isfinite(p).all():
        raise ValueError(f'Expected finite CLIP prototypes [K,512], got {p.shape}')
    return F.normalize(p.float(), dim=-1)


def build_models(cfg, sources, prototypes, device):
    local_models = PROJECT / 'models'
    if (local_models / 'geosem_stda.py').exists():
        # Standalone repo: model implementation is vendored under ./models (version-pinned).
        sys.path.insert(0, str(PROJECT))
    else:
        # Source layout: reuse the external SGDA project.
        sys.path.insert(0, cfg['sgda_root'])
    from models.geosem_stda import GeoSemSTDA
    from models.context_affective_metric import MultiSourceAffectiveMetric
    m = cfg['model']; k = len(cfg['data']['class_names'])
    model = GeoSemSTDA(sources, num_electrodes=62, num_freq_bands=5, st_dim=128,
                      text_dim=512, sample_length=m['sample_length'], representation_mode='cast_level1',
                      classifier_type='clip', num_classes=k, cast_variant='e3_strong_de_channel_graph',
                      dropout=m['dropout']).to(device)
    rank = min(m['rank'], k - 1)
    if 'basis_va' in m:
        # Preserve the existing SEED-IV C1 basis when verified coordinates exist.
        va = torch.tensor(m['basis_va'], dtype=prototypes.dtype, device=device)
        if va.shape != (k, 2) or torch.linalg.matrix_rank(va - va.mean(0)) < rank:
            raise ValueError('Invalid or rank-deficient configured V-A coordinates')
        directions = (torch.linalg.pinv(va - va.mean(0)) @ (prototypes - prototypes.mean(0))).T
        if torch.linalg.matrix_rank(directions) < rank:
            raise ValueError('Rank-deficient V-A mapped basis')
        basis = torch.linalg.qr(directions, mode='reduced').Q[:, :rank]
    else:
        # No invented V-A coordinates for new datasets.
        centered = prototypes - prototypes.mean(0)
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        if singular[rank - 1] < 1e-6:
            raise ValueError('Semantic basis is rank deficient')
        basis = vh[:rank].T.contiguous()
    metric = MultiSourceAffectiveMetric(sources, head_sharing='independent', variant=m['head'],
        context_dim=128, prototypes=prototypes, basis=basis, rank=rank, hidden_dim=64,
        gamma=.5, tau=.07, lambda_regularization=1e-4).to(device)
    disent = ConditionalDisentangler(128, k, sources).to(device)
    return model, metric, disent


def encode(model, disent, x):
    h, _ = model.encode(x, x.new_zeros((len(x), 1, 1)))
    return disent(h)


def contrast_matrix(k, device):
    """K x (K-1) matrix A with A^T A = I and A^T 1 = 0 (all class-pair margins)."""
    a = torch.zeros((k, k - 1), dtype=torch.float32, device=device)
    for j in range(k - 1):
        a[: j + 1, j] = 1.0 / np.sqrt((j + 1) * (j + 2))
        a[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return a


def fused_logits(model, metric, disent, s, centroid, cfg):
    """Full inference-path fused logits ell [B,K] from shared s (all adapters + C1)."""
    zs = [model.prototype_head(a(s)) for a in model.adapters]
    distance = torch.stack([(F.normalize(z, dim=-1) - centroid[j]).norm(dim=-1)
                            for j, z in enumerate(zs)])
    weights = F.softmax(-distance / cfg['model']['fusion_tau'], dim=0)
    return metric.fused_output(s, zs, weights).logits



@torch.no_grad()
def centers(model, disent, parts, sources, device):
    model.eval(); disent.eval()
    result = []
    for j, r in enumerate(sources):
        total, count = None, 0
        for x in torch.tensor(parts[r][0]).split(128):
            s, _ = encode(model, disent, x.to(device))
            z = F.normalize(model.prototype_head(model.adapters[j](s)), dim=-1)
            total = z.sum(0) if total is None else total + z.sum(0)
            count += len(z)
        result.append(F.normalize(total / count, dim=-1))
    return torch.stack(result)


@torch.no_grad()
def evaluate(model, metric, disent, parts, centroid, cfg, device):
    model.eval(); metric.eval(); disent.eval()
    truth, prediction = [], []
    per_subject = []
    k = len(cfg['data']['class_names'])
    for subject, (x, y) in sorted(parts.items()):
        rows = []
        for xb in torch.tensor(x).split(128):
            s, _ = encode(model, disent, xb.to(device))
            zs = [model.prototype_head(a(s)) for a in model.adapters]
            distance = torch.stack([(F.normalize(z, dim=-1) - centroid[j]).norm(dim=-1)
                                    for j, z in enumerate(zs)])
            weights = F.softmax(-distance / cfg['model']['fusion_tau'], dim=0)
            rows.extend(metric.fused_output(s, zs, weights).logits.argmax(-1).cpu().tolist())
        truth.extend(y.tolist()); prediction.extend(rows)
        per_subject.append(float(np.mean(np.array(rows) == y)))
    return dict(accuracy=float(np.mean(np.array(truth) == prediction)),
                subject_mean_accuracy=float(np.mean(per_subject)),
                macro_f1=float(f1_score(truth, prediction, labels=list(range(k)), average='macro', zero_division=0)),
                per_class_recall=recall_score(truth, prediction, labels=list(range(k)), average=None, zero_division=0).tolist(),
                confusion_matrix=confusion_matrix(truth, prediction, labels=list(range(k))).tolist(), n_samples=len(truth))


def _teacher_manifest(cfg, sessions, target, seed, args):
    """Directory of the per-fold, source-dev-selected P0' run (teacher source)."""
    p0 = {**cfg, 'variant': 'P0'}
    effective = dict(config=p0, sessions=sessions, target=target, seed=seed,
                     epochs=args.epochs or cfg['train']['epochs'],
                     steps=args.steps or cfg['train']['steps_per_epoch'])
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()[:16]
    return resolve(cfg['output_root']) / cfg['data']['dataset'] / 'P0' / digest / f'target{target:02d}'


def _load_teacher(cfg, sessions, target, seed, args, n_sources, prototypes, device, sources):
    """Per-fold frozen P0' teacher, selected by SOURCE-DEV (never target-best)."""
    base = _teacher_manifest(cfg, sessions, target, seed, args)
    ckpt = base / 'source_best.pt'
    if not ckpt.exists():
        raise FileNotFoundError(
            f'{ckpt}: train P0 for this exact fold first '
            f'(source_best.pt is the only valid teacher; target_best.pt is not).')
    manifest = json.loads((base / 'run.json').read_text(encoding='utf-8'))
    if manifest.get('target') != target:
        raise ValueError(f'teacher target mismatch: {manifest.get("target")} vs {target}')
    if manifest.get('source_ids') != list(sources):
        raise ValueError('teacher source-subject order mismatch')
    if manifest.get('sessions') != list(sessions):
        raise ValueError(f'teacher session mismatch: {manifest.get("sessions")} vs {list(sessions)}')
    if (manifest.get('audit') or {}).get('normalization') != cfg['data']['normalization']:
        raise ValueError('teacher normalization mismatch')
    t_model, t_metric, t_disent = build_models(cfg, n_sources, prototypes, device)
    st = torch.load(ckpt, map_location=device, weights_only=False)
    t_model.load_state_dict(st['model']); t_metric.load_state_dict(st['metric'])
    t_disent.load_state_dict(st['disentangler'])
    t_model.eval(); t_metric.eval(); t_disent.eval()
    for p in list(t_model.parameters()) + list(t_metric.parameters()) + list(t_disent.parameters()):
        p.requires_grad_(False)
    return t_model, t_metric, t_disent, st['centroids'].to(device)


def run_fold(cfg, sessions, target, seed, device, prototypes, records, args):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parts, sources, audit = make_fold(cfg, records, target)
    effective = dict(config=cfg, sessions=sessions, target=target, seed=seed,
                     epochs=args.epochs or cfg['train']['epochs'], steps=args.steps or cfg['train']['steps_per_epoch'])
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()[:16]
    out = resolve(cfg['output_root']) / cfg['data']['dataset'] / cfg['variant'] / digest / f'target{target:02d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        raise FileExistsError(f'{out}: existing run; choose another output_root rather than overwrite')
    model_root = (PROJECT / 'models') if (PROJECT / 'models' / 'geosem_stda.py').exists() \
        else (Path(cfg['sgda_root']) / 'models')
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).with_name('conditional_disentangler.py'),
                     Path(__file__).with_name('seed_family_data.py'),
                     model_root / 'geosem_stda.py',
                     model_root / 'context_affective_metric.py']}
    (out / 'run.json').write_text(json.dumps(dict(**effective, source_ids=sources, audit=audit,
        code_sha256=fingerprints, data_files=[d['path'] for d in records.values()],
        checkpoint_selection='target-best accuracy (transductive)', target_used_for_training=False,
        target_used_for_checkpoint_selection=True), ensure_ascii=False, indent=2), encoding='utf-8')
    model, metric, disent = build_models(cfg, len(sources), prototypes, device)
    teacher = (_load_teacher(cfg, sessions, target, seed, args, len(sources), prototypes, device, sources)
               if cfg['variant'] == 'P3' else None)
    A = contrast_matrix(len(cfg['data']['class_names']), device)
    t = cfg['train']
    aux_modules = [disent.identity, disent.specific, disent.specific_id, disent.t_identity]
    auxiliary = [p for module in aux_modules for p in module.parameters()]
    task_opt = torch.optim.Adam([
        {'params': model.cast_encoder.parameters(), 'lr': t['lr'] * t['backbone_lr_multiplier']},
        {'params': model.adapters.parameters()}, {'params': model.prototype_head.parameters()},
        {'params': metric.parameters()}, {'params': disent.shared.parameters()}], lr=t['lr'], weight_decay=t['weight_decay'])
    # Eq. 21 representation params: the adversary is routed through encoder + S only.
    repr_params = list(model.cast_encoder.parameters()) + list(disent.shared.parameters())
    aux_opt = torch.optim.Adam(auxiliary, lr=t['lr'], weight_decay=t['weight_decay'])
    rng = np.random.default_rng(seed)
    cells = {(r, c): np.flatnonzero(parts['train'][r][1] == c) for r in sources
             for c in range(len(cfg['data']['class_names']))}
    best, best_dev, history = -1., -1., []
    warm = t['warmup_epochs']; variant = cfg['variant']
    for epoch in range(1, effective['epochs'] + 1):
        model.train(); metric.train(); disent.train()
        active = variant != 'P0' and epoch > warm
        strength = min(1., max(0., (epoch - warm) / max(1, t['ramp_epochs'])))
        with torch.no_grad():
            centroid = centers(model, disent, parts['train'], sources, device)
        model.train(); metric.train(); disent.train()
        sums = []
        for _ in range(effective['steps']):
            ss, qq, yy, rr, tasks, fusions, teacher_fused = [], [], [], [], [], [], []
            for j, r in enumerate(sources):
                ids = np.concatenate([rng.choice(cells[r, c], size=t['samples_per_class'], replace=True)
                                      for c in range(len(cfg['data']['class_names']))])
                x = torch.tensor(parts['train'][r][0][ids], device=device)
                y = torch.tensor(parts['train'][r][1][ids], device=device)
                s, q = encode(model, disent, x)
                z = model.prototype_head(model.adapters[j](s))
                branch = metric._branch_heads()[j](s, z)
                tasks.append(F.cross_entropy(branch.logits, y) + branch.regularization)
                fusions.append(F.cross_entropy(fused_logits(model, metric, disent, s, centroid, cfg), y))
                if teacher is not None:
                    t_model, t_metric, t_disent, t_centroid = teacher
                    with torch.no_grad():
                        s_t, _ = encode(t_model, t_disent, x)
                        teacher_fused.append(fused_logits(t_model, t_metric, t_disent, s_t, t_centroid, cfg))
                ss.append(s); qq.append(q); yy.append(y); rr.append(torch.full_like(y, j))
            s, q, y, r = map(torch.cat, (ss, qq, yy, rr))
            # The full fused decision is needed by both the discriminator fitting and the adversary.
            fused_s = fused_logits(model, metric, disent, s, centroid, cfg) if variant == 'P3' else None
            t_rel = (fused_s @ A) if variant == 'P3' else None
            # Separate discriminator fitting: s, q AND t heads are fitted by normal CE. The t head
            # must actually learn, otherwise the adversary faces a random classifier.
            if active:
                aux_opt.zero_grad(set_to_none=True)
                aux_loss = disent.identity_loss(s.detach(), y, r) + F.cross_entropy(disent.specific_id(q), r)
                if t_rel is not None:
                    aux_loss = aux_loss + disent.t_identity_loss(t_rel.detach(), y, r)
                aux_loss.backward(); aux_opt.step()
            for p in auxiliary:
                p.requires_grad_(False)
            task = torch.stack(tasks).mean() + torch.stack(fusions).mean()
            if variant == 'P3':
                adv = disent.t_identity_loss(t_rel, y, r, strength) if active else task.new_zeros(())
                if teacher is not None and active:
                    p0 = F.softmax(torch.cat(teacher_fused), dim=-1).detach()
                    keep = F.kl_div(F.log_softmax(fused_s, dim=-1), p0, reduction='batchmean')
                else:
                    keep = task.new_zeros(())
            else:
                adv = disent.identity_loss(s, y, r, strength) if active else task.new_zeros(())
                keep = task.new_zeros(())
            dec = disent.decorrelation(s, q.detach(), y) if active and variant == 'P2' else task.new_zeros(())
            loss = task + t['lambda_dec'] * dec + t.get('lambda_keep', 1.0) * keep
            task_opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            if active:
                # Eq. 21: the adversary updates representation params (encoder + S) only; it never
                # touches adapters / projection / C1 / fusion, so a gain cannot come from just
                # re-fitting the downstream head.
                adv_grads = torch.autograd.grad(t['lambda_adv'] * adv, repr_params,
                                                retain_graph=False, allow_unused=True)
                for p, g in zip(repr_params, adv_grads):
                    if g is not None:
                        p.grad = (p.grad + g) if p.grad is not None else g.clone()
            task_opt.step()
            for p in auxiliary:
                p.requires_grad_(True)
            sums.append([task.item(), adv.item(), dec.item(), keep.item()])
        dev = evaluate(model, metric, disent, parts['dev'], centroid, cfg, device)
        tgt = evaluate(model, metric, disent, parts['target'], centroid, cfg, device)
        record = dict(epoch=epoch, train_losses=np.mean(sums, axis=0).tolist(), source_dev=dev, target=tgt)
        history.append(record)
        state = dict(model=model.state_dict(), metric=metric.state_dict(), disentangler=disent.state_dict(),
                     centroids=centroid, prototypes=prototypes, epoch=epoch, source_ids=sources,
                     task_optimizer=task_opt.state_dict(), auxiliary_optimizer=aux_opt.state_dict(), effective=effective)
        if tgt['accuracy'] > best:
            best = tgt['accuracy']; torch.save(state, out / 'target_best.pt')
        if dev['subject_mean_accuracy'] > best_dev:
            best_dev = dev['subject_mean_accuracy']; torch.save(state, out / 'source_best.pt')
        torch.save(state, out / 'last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'{cfg["data"]["dataset"]} {variant} target={target} epoch={epoch} '
              f'target_best={best:.4f} target_acc={tgt["accuracy"]:.4f}', flush=True)
    saved = torch.load(out / 'target_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model']); metric.load_state_dict(saved['metric']); disent.load_state_dict(saved['disentangler'])
    result = evaluate(model, metric, disent, parts['target'], saved['centroids'], cfg, device)
    result.update(selected_epoch=saved['epoch'], target_best=best, class_names=cfg['data']['class_names'])
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'COMPLETE').write_text('ok\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', choices=['P0', 'P1', 'P2', 'P3'])
    parser.add_argument('--targets', type=int, nargs='+')
    parser.add_argument('--sessions', type=int, nargs='+', help='Override with one pooled session group')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--normalization', choices=['source_global_zscore', 'subject_zscore', 'record_minmax', 'none'])
    parser.add_argument('--cache-root')
    parser.add_argument('--clip-root', help='override clip_root (local CLIP text model dir)')
    parser.add_argument('--sgda-root', help='override sgda_root (external model code fallback)')
    parser.add_argument('--output-root')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.variant:
        cfg['variant'] = args.variant
    if args.sessions:
        cfg['data']['session_groups'] = [args.sessions]
    if args.seeds:
        cfg['train']['seeds'] = args.seeds
    if args.normalization:
        cfg['data']['normalization'] = args.normalization
    if args.cache_root:
        cfg['data']['cache_root'] = str(resolve(args.cache_root))
    if args.clip_root:
        cfg['clip_root'] = str(resolve(args.clip_root))
    if args.sgda_root:
        cfg['sgda_root'] = str(resolve(args.sgda_root))
    if args.output_root:
        cfg['output_root'] = str(resolve(args.output_root))
    if (args.epochs is not None and args.epochs < 1) or (args.steps is not None and args.steps < 1):
        parser.error('--epochs/--steps must be positive')
    if args.check_data:
        for sessions in cfg['data']['session_groups']:
            records, subjects = load_records(cfg, sessions)
            target = (args.targets or subjects)[0]
            if target not in subjects:
                raise ValueError(f'Unknown target {target}')
            _, sources, audit = make_fold(cfg, records, target)
            print(json.dumps(dict(dataset=cfg['data']['dataset'], sessions=sessions, subjects=subjects,
                n_sources=len(sources), classes=cfg['data']['class_names'], normalization=audit['normalization'])))
        return
    device = torch.device(args.device)
    prototypes = prototype_vectors(cfg, device)
    for sessions in cfg['data']['session_groups']:
        records, subjects = load_records(cfg, sessions)
        for seed in cfg['train']['seeds']:
            for target in args.targets or subjects:
                if target not in subjects:
                    raise ValueError(f'Unknown target {target}')
                run_fold(cfg, sessions, target, seed, device, prototypes, records, args)


if __name__ == '__main__':
    main()
