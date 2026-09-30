#!/usr/bin/env python3
"""Merge jersey_ocr_probe reports (e.g. part1 + part2 batches).

Pools per_sample reads by fragment_id, dedupes on (fragment_id, frame),
recomputes consistency with the same rules as jersey_ocr_probe.py, and flags
pre-merge majority disagreements between inputs.

Usage (from sports-main/examples/soccer):
  python tools/merge_ocr_reports.py \\
    --report part1=data/jersey_ocr_v5_part1/jersey_ocr_report.json \\
    --report part2=data/jersey_ocr_v5_part2/jersey_ocr_report.json \\
    --out data/jersey_ocr_v5/jersey_ocr_report.json

Self-test (merged pct should match single report):
  python tools/merge_ocr_reports.py \\
    --report a=data/jersey_ocr_v5_part1/jersey_ocr_report.json \\
    --report b=data/jersey_ocr_v5_part1/jersey_ocr_report.json \\
    --out /tmp/merged_self.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SOC = Path(__file__).resolve().parents[1]
if str(_SOC) not in sys.path:
    sys.path.insert(0, str(_SOC))

from tools.jersey_ocr_probe import is_consistent, majority_stats


def _read_key(r: dict) -> tuple:
    return (int(r['number']), round(float(r['confidence']), 3))


def merge_per_sample(samples: list[dict]) -> list[dict]:
    """Dedupe frames; combine reads on the same frame across inputs."""
    by_frame: dict[int, dict] = {}
    for ps in samples:
        fnum = int(ps['frame'])
        reads = list(ps.get('reads') or [])
        if fnum not in by_frame:
            by_frame[fnum] = {
                'frame': fnum,
                'h': ps.get('h'),
                'reads': [],
            }
            seen: set[tuple] = set()
        else:
            seen = {_read_key(r) for r in by_frame[fnum]['reads']}
        for r in reads:
            k = _read_key(r)
            if k not in seen:
                by_frame[fnum]['reads'].append(r)
                seen.add(k)
    return [by_frame[f] for f in sorted(by_frame.keys())]


def fragment_from_pooled(base: dict, per_sample: list[dict]) -> dict:
    all_reads: list[tuple[int, float]] = []
    for ps in per_sample:
        for r in ps.get('reads') or []:
            all_reads.append((int(r['number']), float(r['confidence'])))

    consistent, maj, maj_count, n_reads = is_consistent(all_reads)
    maj_num, _, maj_mean_conf = majority_stats(all_reads)

    out = {
        'fragment_id': int(base['fragment_id']),
        'team': base.get('team'),
        'class': base.get('class'),
        'n_large_frames': base.get('n_large_frames'),
        'n_samples': len(per_sample),
        'n_successful_reads': n_reads,
        'all_reads': [
            {'number': n, 'confidence': round(c, 3)} for n, c in all_reads
        ],
        'majority_number': maj_num,
        'majority_count': maj_count,
        'majority_mean_confidence': (
            round(maj_mean_conf, 3) if maj_num is not None else None
        ),
        'consistent': consistent,
        'per_sample': per_sample,
    }
    return out


def team_number_distribution(fragments: list[dict]) -> dict[str, dict[str, int]]:
    by_team: dict[str, Counter] = defaultdict(Counter)
    for frag in fragments:
        maj = frag.get('majority_number')
        if maj is None:
            continue
        team = frag.get('team')
        key = 'team_' + str(team) if team is not None else 'team_unknown'
        by_team[key][int(maj)] += 1
    return {k: dict(v) for k, v in sorted(by_team.items())}


def merge_reports(labeled: list[tuple[str, dict]]) -> tuple[dict, list[dict]]:
    """Return (merged_report_dict, conflict_rows)."""
    if not labeled:
        raise ValueError('no reports')

    frag_by_label: list[tuple[str, dict[int, dict]]] = []
    for label, report in labeled:
        m = {int(f['fragment_id']): f for f in report.get('fragments') or []}
        frag_by_label.append((label, m))

    all_ids = sorted({fid for _, m in frag_by_label for fid in m.keys()})

    conflicts: list[dict] = []
    for fid in all_ids:
        majors: list[tuple[str, int | None]] = []
        for label, m in frag_by_label:
            if fid not in m:
                continue
            majors.append((label, m[fid].get('majority_number')))
        nums = {maj for _, maj in majors if maj is not None}
        if len(nums) > 1:
            conflicts.append({
                'fragment_id': fid,
                'majorities': {lab: maj for lab, maj in majors},
            })

    merged_frags: list[dict] = []
    for fid in all_ids:
        pooled_samples: list[dict] = []
        base: dict | None = None
        for _label, m in frag_by_label:
            if fid not in m:
                continue
            fr = m[fid]
            if base is None:
                base = fr
            pooled_samples.extend(fr.get('per_sample') or [])
        if base is None:
            continue
        per_sample = merge_per_sample(pooled_samples)
        merged_frags.append(fragment_from_pooled(base, per_sample))

    n_probed = len(merged_frags)
    n_any = sum(1 for f in merged_frags if f['n_successful_reads'] > 0)
    n_cons = sum(1 for f in merged_frags if f['consistent'])
    pct_any = 100.0 * n_any / n_probed if n_probed else 0.0
    pct_cons = 100.0 * n_cons / n_probed if n_probed else 0.0

    if pct_cons >= 30.0:
        verdict = 'PASS (>=30% consistent) — worth a pipeline experiment'
    elif pct_cons < 10.0:
        verdict = (
            'STOP (<10% consistent) — jersey OCR unlikely to anchor identity here'
        )
    else:
        verdict = (
            f'MARGINAL ({pct_cons:.1f}% consistent) — eyeball contact sheet '
            f'before continuing'
        )

    first = labeled[0][1]
    merged = {
        'dump': first.get('dump'),
        'video': first.get('video'),
        'params': first.get('params'),
        'crop_settings': first.get('crop_settings'),
        'height_diagnostics': first.get('height_diagnostics'),
        'merge_sources': [lab for lab, _ in labeled],
        'summary': {
            'fragments_probed': n_probed,
            'any_read': n_any,
            'consistent': n_cons,
            'pct_any_read': round(pct_any, 2),
            'pct_consistent': round(pct_cons, 2),
            'verdict': verdict,
        },
        'team_number_distribution': team_number_distribution(merged_frags),
        'fragments': merged_frags,
        'merge_conflicts': conflicts,
    }
    return merged, conflicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--report', action='append', required=True, metavar='LABEL=PATH',
        help='Input report (repeatable), e.g. part1=data/.../jersey_ocr_report.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    labeled: list[tuple[str, dict]] = []
    for spec in args.report:
        if '=' not in spec:
            raise SystemExit(f'--report must be LABEL=PATH, got: {spec!r}')
        label, path_s = spec.split('=', 1)
        path = Path(path_s)
        if not path.is_file():
            raise SystemExit(f'missing report: {path}')
        report = json.loads(path.read_text(encoding='utf-8'))
        labeled.append((label.strip(), report))

    merged, conflicts = merge_reports(labeled)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding='utf-8')

    s = merged['summary']
    print('\n=== Merged jersey OCR report ===')
    print(f'  Sources                       : {merged["merge_sources"]}')
    print(f'  Output                        : {args.out}')
    print(f'  Fragments probed              : {s["fragments_probed"]}')
    print(f'  Any digit read                : {s["any_read"]} ({s["pct_any_read"]}%)')
    print(f'  Consistent (3+ & >=60% agree) : {s["consistent"]} ({s["pct_consistent"]}%)')
    print(f'  Verdict                       : {s["verdict"]}')
    print('\n  Majority number distribution by team (fragment counts):')
    for team_key in sorted(merged['team_number_distribution'].keys()):
        dist = merged['team_number_distribution'][team_key]
        c = Counter({int(k): v for k, v in dist.items()})
        print(f'    {team_key}: {len(c)} distinct numbers, top={c.most_common(12)}')
    if conflicts:
        print(f'\n  Pre-merge majority conflicts  : {len(conflicts)} fragment(s)')
        for row in conflicts[:20]:
            print(f'    frag {row["fragment_id"]}: {row["majorities"]}')
        if len(conflicts) > 20:
            print(f'    ... and {len(conflicts) - 20} more (see merge_conflicts in JSON)')
    else:
        print('\n  Pre-merge majority conflicts  : none')


if __name__ == '__main__':
    main()
