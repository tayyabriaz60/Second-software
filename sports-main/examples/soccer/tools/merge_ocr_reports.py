#!/usr/bin/env python3
"""Merge jersey_ocr_probe reports (e.g. part1 + part2 batches).

Pools per_sample reads by fragment_id; first source to sample a frame wins
(no double-count across batches). Recomputes consistency with the same rules
as jersey_ocr_probe.py, and flags pre-merge majority disagreements.

Usage (from sports-main/examples/soccer):
  python tools/merge_ocr_reports.py \\
    --report part1=data/jersey_ocr_v5_part1/jersey_ocr_report.json \\
    --report part2=data/jersey_ocr_v5_part2/jersey_ocr_report.json \\
    --out data/jersey_ocr_v5/jersey_ocr_report.json
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

_COMPAT_PARAMS = (
    'ocr_engine', 'upscale', 'ocr_preprocess', 'min_h', 'min_large_frames',
    'max_samples', 'torso_width_margin', 'no_ocr_paragraph',
    'include_referees', 'referee_class',
)


def merge_per_sample(samples: list[dict]) -> list[dict]:
    """First source to sample a frame wins; skip later sources for that frame."""
    by_frame: dict[int, dict] = {}
    for ps in samples:
        fnum = int(ps['frame'])
        if fnum in by_frame:
            continue
        by_frame[fnum] = {
            'frame': fnum,
            'h': ps.get('h'),
            'reads': list(ps.get('reads') or []),
        }
    return [by_frame[f] for f in sorted(by_frame.keys())]


def check_compatible(labeled: list[tuple[str, dict]]) -> list[str]:
    if len(labeled) < 2:
        return []
    problems: list[str] = []
    ref_label, ref = labeled[0]
    ref_params = ref.get('params') or {}
    ref_crop = ref.get('crop_settings')
    ref_dump = Path(ref.get('dump') or '').name
    for label, report in labeled[1:]:
        params = report.get('params') or {}
        for key in _COMPAT_PARAMS:
            if params.get(key) != ref_params.get(key):
                problems.append(
                    f'{label} vs {ref_label}: params.{key} '
                    f'{params.get(key)!r} != {ref_params.get(key)!r}')
        if (report.get('crop_settings') or {}) != (ref_crop or {}):
            problems.append(f'{label} vs {ref_label}: crop_settings differ')
        dump_name = Path(report.get('dump') or '').name
        if dump_name != ref_dump:
            problems.append(
                f'{label} vs {ref_label}: dump {dump_name!r} != {ref_dump!r}')
    return problems


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

    source_count: Counter[int] = Counter()
    for _label, m in frag_by_label:
        for fid in m:
            source_count[fid] += 1
    n_multi_source = sum(1 for c in source_count.values() if c > 1)

    all_ids = sorted(source_count.keys())

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
            'fragments_in_multiple_sources': n_multi_source,
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
    ap.add_argument('--allow-mismatch', action='store_true',
                    help='Merge even if params/crop/dump differ between reports')
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

    problems = check_compatible(labeled)
    if problems:
        for p in problems:
            print(p)
        if not args.allow_mismatch:
            raise SystemExit(
                'Refusing to pool reports with mismatched settings '
                '(use --allow-mismatch)')

    merged, conflicts = merge_reports(labeled)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding='utf-8')

    s = merged['summary']
    print('\n=== Merged jersey OCR report ===')
    print(f'  Sources                       : {merged["merge_sources"]}')
    print(f'  Output                        : {args.out}')
    print(f'  Fragments probed              : {s["fragments_probed"]}')
    print(f'  ...in more than one source    : {s["fragments_in_multiple_sources"]}')
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
