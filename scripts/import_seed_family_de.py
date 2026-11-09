"""Convert explicitly indexed MAT DE trials to the common cache contract.

CSV columns: path,key,layout,session,subject,trial,label.
layout is CTF (62,time,5) or TCF (time,62,5). label is a canonical class name.
Relative MAT paths are resolved against the CSV directory. Never guesses labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-description', required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    names = cfg['data']['class_names']; groups = {}; seen = set(); sources = {}
    with args.index.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            s, r, t = [int(row[k]) for k in ('session', 'subject', 'trial')]
            if (s, r, t) in seen:
                raise ValueError(f'duplicate trial {(s, r, t)}')
            seen.add((s, r, t))
            path = Path(row['path'])
            if not path.is_absolute():
                path = args.index.parent / path
            feature = np.asarray(loadmat(path)[row['key']], dtype=np.float32)
            if row['layout'] == 'CTF':
                feature = feature.transpose(1, 0, 2)
            elif row['layout'] != 'TCF':
                raise ValueError('layout must explicitly be CTF or TCF')
            if feature.ndim != 3 or feature.shape[1:] != (62, 5) or not np.isfinite(feature).all():
                raise ValueError(f'{path}:{row["key"]}: expected finite DE [time,62,5]')
            label = names.index(row['label'])
            n = len(feature)
            groups.setdefault((s, r), []).append((feature, np.full(n, label, np.int64),
                                                 np.full(n, t, np.int64), np.arange(n)))
            if str(path) not in sources:
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not groups:
        raise ValueError('Empty trial index')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / f'session{s}_subject{r:02d}.npz' for s, r in groups]
    if (args.output / 'manifest.json').exists() or any(path.exists() for path in paths):
        raise FileExistsError('Import refuses to overwrite an existing cache')
    for (s, r), rows in sorted(groups.items()):
        arrays = [np.concatenate([row[i] for row in rows]) for i in range(4)]
        np.savez_compressed(args.output / f'session{s}_subject{r:02d}.npz',
                            **dict(zip(('x', 'y', 'trial', 'window'), arrays)))
    manifest = dict(dataset=cfg['data']['dataset'], classes=dict(enumerate(names)),
                    feature_description=args.feature_description, label_source='explicit trial-index CSV',
                    index=str(args.index.resolve()), source_sha256=sources,
                    channel_order='must match the configured standard 62-channel DE caches',
                    note='No filtering/LDS/raw EEG extraction or normalization performed by this importer')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Imported {len(groups)} session-subject records to {args.output}')


if __name__ == '__main__':
    main()
