"""CPU integration checks with synthetic DE, never performance evidence."""
import copy
import json
import sys
import tempfile
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / 'experiments_dtrm' if (ROOT / 'experiments_dtrm').is_dir() else ROOT
sys.path.insert(0, str(CODE_ROOT))
from train_seed_family_disentangle import read_config, run_fold
from seed_family_data import load_records, make_fold
from conditional_disentangler import ConditionalDisentangler


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(91)
    checks = []
    d = ConditionalDisentangler(128, 4, 2)
    h = torch.randn(8, 128, requires_grad=True)
    s, q = d(h)
    torch.nn.functional.cross_entropy(d.specific_id(q), torch.arange(8) % 2).backward()
    assert h.grad is None, 'specific identity must not update CAST input'
    assert d.specific[0].weight.grad is not None
    checks.append({'specific_gradient_isolation': True})
    # Whole training/evaluation/checkpoint chain, all classes and P0/P1/P2.
    with tempfile.TemporaryDirectory(prefix='seed_disentangle_check_', ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        mat = tmp / 'features.mat'
        savemat(mat, {'de_test': rng.normal(size=(62, 6, 5)).astype('float32')})
        index = tmp / 'index.csv'
        index.write_text('path,key,layout,session,subject,trial,label\nfeatures.mat,de_test,CTF,1,1,1,happy\n', encoding='utf-8')
        subprocess.run([sys.executable, str(ROOT / 'scripts/import_seed_family_de.py'),
                        '--config', str(CODE_ROOT / 'configs/seed_family/seedvii.json'),
                        '--index', str(index), '--output', str(tmp / 'imported'),
                        '--feature-description', 'synthetic DE interface verification'], check=True)
        with np.load(tmp / 'imported/session1_subject01.npz') as f:
            assert f['x'].shape == (6, 62, 5) and (f['y'] == 0).all()
        checks.append({'mat_import_roundtrip': True})
        for k, dataset in [(3, 'seed'), (4, 'seediv'), (5, 'seedv'), (7, 'seedvii')]:
            cfg = read_config(CODE_ROOT / f'configs/seed_family/{dataset}.json')
            cache = tmp / dataset; cache.mkdir()
            for subject in (1, 2, 3):
                n = k * 2 * 5
                np.savez(cache / f'session1_subject{subject:02d}.npz',
                         x=rng.normal(size=(n, 62, 5)).astype('float32'),
                         y=np.repeat(np.arange(k), 10),
                         trial=np.repeat(np.arange(k * 2), 5), window=np.tile(np.arange(5), k * 2))
            cfg['data']['cache_root'] = str(cache)
            cfg['data']['session_groups'] = [[1]]
            cfg['output_root'] = str(tmp / 'runs')
            cfg['train'].update(warmup_epochs=0, samples_per_class=2, seeds=[42])
            records, _ = load_records(cfg, [1])
            parts, _, audit = make_fold(cfg, records, 1)
            assert not (set(map(tuple, audit['train_trials'])) & set(map(tuple, audit['dev_trials'])))
            changed = copy.deepcopy(records)
            changed[1, 1]['x'] += 100000
            parts_changed, _, _ = make_fold(cfg, changed, 1)
            np.testing.assert_array_equal(parts['train'][2][0], parts_changed['train'][2][0])
            prototypes = torch.nn.functional.normalize(torch.randn(k, 512), dim=-1)
            for variant in ('P0', 'P1', 'P2', 'P3'):
                cfg['variant'] = variant
                run_fold(cfg, [1], 1, 42, torch.device('cpu'), prototypes, records,
                         SimpleNamespace(epochs=1, steps=2))
            metrics = list((tmp / 'runs' / dataset).rglob('metrics.json'))
            assert len(metrics) == 4
            for path in metrics:
                result = json.loads(path.read_text())
                assert np.shape(result['confusion_matrix']) == (k, k)
                assert result['n_samples'] == k * 2 * 3
                history = json.loads(path.with_name('history.json').read_text())
                if 'P1' in path.parts or 'P2' in path.parts:
                    assert history[0]['train_losses'][1] > 0, 'adversarial path was not exercised'
                if 'P2' in path.parts:
                    assert history[0]['train_losses'][2] > 0, 'decorrelation path was not exercised'
            checks.append(dict(classes=k, synthetic_training_variants=['P0', 'P1', 'P2'],
                               trial_disjoint=True, target_stats_isolated=True, dynamic_metrics=True))
    out = CODE_ROOT / 'results/seed_family_disentangle/verify'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checks.json').write_text(json.dumps(dict(status='passed', checks=checks,
        torch_version=torch.__version__, synthetic_only=True), indent=2), encoding='utf-8')
    print(f'PASS: {out / "checks.json"}')


if __name__ == '__main__':
    main()
