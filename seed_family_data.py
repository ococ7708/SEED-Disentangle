"""Trial-safe DE cache interface. No target labels are used in fitting statistics."""
from pathlib import Path
import re
import numpy as np


def load_records(config, sessions):
    root = Path(config['data']['cache_root'])
    records = {}
    for path in sorted(root.glob('session*_subject*.npz')):
        match = re.fullmatch(r'session(\d+)_subject(\d+)\.npz', path.name)
        session, subject = map(int, match.groups())
        if session not in sessions:
            continue
        with np.load(path, allow_pickle=False) as f:
            x, y, trial, window = [np.array(f[k]) for k in ('x', 'y', 'trial', 'window')]
        n = len(x)
        if x.shape != (n, 62, 5) or any(v.shape != (n,) for v in (y, trial, window)):
            raise ValueError(f'{path}: expected x[N,62,5] and y/trial/window[N]')
        if not np.isfinite(x).all() or not all(np.issubdtype(v.dtype, np.integer) for v in (y, trial, window)):
            raise ValueError(f'{path}: nonfinite features or noninteger metadata')
        if not set(y.tolist()) <= set(range(len(config['data']['class_names']))):
            raise ValueError(f'{path}: labels do not match configured class order')
        keys = np.stack([trial, window], 1)
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError(f'{path}: duplicate trial/window keys')
        for t in np.unique(trial):
            if len(np.unique(y[trial == t])) != 1:
                raise ValueError(f'{path}: trial {t} changes class; explicitly segment it before import')
        if (session, subject) in records:
            raise ValueError('duplicate record')
        records[session, subject] = dict(x=x.astype(np.float32), y=y, trial=trial, window=window, path=str(path))
    if not records:
        raise FileNotFoundError(f'No DE caches for sessions {sessions} in {root}')
    subjects = sorted({r for _, r in records})
    if any((s, r) not in records for s in sessions for r in subjects):
        raise ValueError('Incomplete subject/session matrix; set an explicit complete cache root')
    return records, subjects


def make_fold(config, records, target):
    """Source validation: one whole trial/class/subject, across selected sessions.

    A subject remains a single source domain when multiple sessions are pooled.
    """
    length = config['model']['sample_length']
    classes = len(config['data']['class_names'])
    train_keys, dev_keys = set(), set()
    subjects = sorted({r for _, r in records if r != target})
    if len(subjects) < 2:
        raise ValueError('Need at least two source subjects plus a target')
    for r in subjects:
        for c in range(classes):
            trials = sorted((s, r, int(t)) for (s, rr), d in records.items() if rr == r
                            for t in np.unique(d['trial'][d['y'] == c]))
            if len(trials) < 2:
                raise ValueError(f'subject {r} class {c}: need >=2 trials for source validation')
            dev_keys.add(trials[-1])
            train_keys.update(trials[:-1])
    mode = config['data']['normalization']
    if mode == 'source_global_zscore':
        values = np.concatenate([d['x'][np.isin(d['trial'], [t for ss, rr, t in train_keys if (ss, rr) == (s, r)])]
                                 for (s, r), d in records.items() if r != target])
        mean, scale = values.mean(0), values.std(0).clip(1e-5)
    elif mode not in ('subject_zscore', 'record_minmax', 'none'):
        raise ValueError(f'Unknown normalization: {mode}')
    partitions = {name: {} for name in ('train', 'dev', 'target')}
    audit = {'train_trials': sorted(train_keys), 'dev_trials': sorted(dev_keys),
             'normalization': mode, 'target_statistics_used': mode in ('subject_zscore', 'record_minmax')}
    for (s, r), d in records.items():
        x = d['x']
        if mode == 'source_global_zscore':
            x = (x - mean) / scale
        elif mode == 'subject_zscore':
            x = (x - x.mean(0)) / x.std(0).clip(1e-5)
        elif mode == 'record_minmax':
            x = 2 * (x - x.min(0)) / (x.max(0) - x.min(0)).clip(1e-5) - 1
        for t in np.unique(d['trial']):
            idx = np.flatnonzero(d['trial'] == t)
            idx = idx[np.argsort(d['window'][idx])]
            dest = 'target' if r == target else ('dev' if (s, r, int(t)) in dev_keys else 'train')
            bucket = partitions[dest].setdefault(r, [[], []])
            for start in range(0, len(idx) - length + 1, config['data'].get('stride', 1)):
                ids = idx[start:start + length]
                if not np.all(np.diff(d['window'][ids]) == 1):
                    continue
                bucket[0].append(x[ids]); bucket[1].append(int(d['y'][ids[0]]))
    for name, part in partitions.items():
        for r, (xs, ys) in part.items():
            if not xs:
                raise ValueError(f'{name}/{r}: no valid windows')
            part[r] = (np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64))
            if set(ys) != set(range(classes)):
                raise ValueError(f'{name}/{r}: missing classes after windowing')
    return partitions, subjects, audit
