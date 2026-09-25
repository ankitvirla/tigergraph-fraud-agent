#!/usr/bin/env python3
"""Fast, read-only audit of final held-out fraud probability predictions.

Run from project root:
    python audit_final_probability_model.py

Input: analysis/final_probability_model_predictions.csv
Outputs (under analysis/):
    final_probability_audit_summary.json
    final_probability_audit_notes.txt
    final_probability_error_cases.csv
    final_probability_calibration.csv

This audit does not retrain, recalibrate, or change the model. Thresholds below
are diagnostics for inspection, NOT business decision thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

INPUT = Path('analysis/final_probability_model_predictions.csv')
OUT = Path('analysis')
BINS = np.linspace(0, 1, 11)
LOW = 0.10       # inspect confident missed fraud
HIGH = 0.90      # inspect confident cleared cases
MIN_BAND_N = 30  # small bands cannot support stable calibration conclusions


def scalar(value):
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    if not INPUT.is_file():
        raise SystemExit(f'Missing {INPUT}. Run build_final_probability_model.py first.')
    df = pd.read_csv(INPUT)
    required = {'actual_fraud', 'fraud_probability'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f'Missing columns: {missing}; available: {list(df.columns)}')
    if df.empty:
        raise SystemExit('Prediction file is empty.')
    df['actual_fraud'] = pd.to_numeric(df['actual_fraud'], errors='coerce')
    df['fraud_probability'] = pd.to_numeric(df['fraud_probability'], errors='coerce')
    if df[list(required)].isna().any().any():
        raise SystemExit('actual_fraud / fraud_probability contain missing or invalid values.')
    if not df['actual_fraud'].isin([0, 1]).all():
        raise SystemExit('actual_fraud must be binary (0/1).')
    if not df['fraud_probability'].between(0, 1).all():
        raise SystemExit('fraud_probability must lie within [0,1].')
    df['actual_fraud'] = df['actual_fraud'].astype(int)

    y = df['actual_fraud'].to_numpy()
    p = df['fraud_probability'].to_numpy()
    n = len(df)
    fraud_n = int(y.sum())
    overall_rate = fraud_n / n
    brier = float(brier_score_loss(y, p))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None

    # Explicit interval semantics: [lower, upper), except last [0.9,1].
    band_idx = np.minimum(np.searchsorted(BINS, p, side='right') - 1, 9)
    rows = []
    ece = 0.0
    for i in range(10):
        sub = df.iloc[np.flatnonzero(band_idx == i)]
        count = len(sub)
        pred = float(sub['fraud_probability'].mean()) if count else None
        obs = float(sub['actual_fraud'].mean()) if count else None
        err = abs(pred - obs) if count else None
        if count:
            ece += (count / n) * err
        rows.append({
            'band': f'{BINS[i]:.2f}-{BINS[i+1]:.2f}',
            'count': count,
            'share_of_test': count / n,
            'mean_predicted': pred,
            'observed_fraud_rate': obs,
            'absolute_error': err,
            'small_sample': count < MIN_BAND_N,
        })
    calibration = pd.DataFrame(rows)

    # These are diagnostic groups, not operational decision rules.
    low_misses = df[(df.actual_fraud == 1) & (df.fraud_probability < LOW)].copy()
    high_false_alarms = df[(df.actual_fraud == 0) & (df.fraud_probability >= HIGH)].copy()
    middle = df[(df.fraud_probability >= LOW) & (df.fraud_probability < HIGH)].copy()
    high = df[df.fraud_probability >= HIGH].copy()
    low = df[df.fraud_probability < LOW].copy()
    high['audit_group'] = np.where(high.actual_fraud.eq(0), 'high_probability_cleared', 'high_probability_fraud')
    low['audit_group'] = np.where(low.actual_fraud.eq(1), 'low_probability_fraud', 'low_probability_cleared')
    middle['audit_group'] = 'middle_probability_case'

    # Save all confidently wrong cases plus middle cases for rapid evidence review.
    review = pd.concat([low_misses.assign(audit_group='low_probability_fraud'),
                        high_false_alarms.assign(audit_group='high_probability_cleared'),
                        middle.assign(audit_group='middle_probability_case')], ignore_index=True)
    review['confidence_error'] = np.where(review.actual_fraud.eq(1),
                                          1 - review.fraud_probability,
                                          review.fraud_probability)
    review = review.sort_values('confidence_error', ascending=False)

    # Monotonicity is descriptive; compare only populated adjacent bands with >=30 cases.
    stable = calibration[calibration['count'] >= MIN_BAND_N]
    stable_rates = stable['observed_fraud_rate'].to_numpy(dtype=float)
    monotonic_violations = int((np.diff(stable_rates) < 0).sum()) if len(stable_rates) > 1 else None

    # No leakage or production-prevalence audit is possible from predictions alone.
    flags = []
    if len(low_misses):
        flags.append(f'{len(low_misses)} confirmed fraud cases predicted below {LOW:.0%}; inspect their evidence.')
    if len(high_false_alarms):
        flags.append(f'{len(high_false_alarms)} cleared cases predicted at or above {HIGH:.0%}; inspect their evidence.')
    if (calibration['small_sample'] & calibration['count'].gt(0)).any():
        flags.append('Some probability bands contain fewer than 30 cases; avoid interpreting their observed rates as stable.')
    if monotonic_violations:
        flags.append(f'{monotonic_violations} decrease(s) in observed fraud rate across adequately populated adjacent bands.')
    flags.append('Historical closed-case test prevalence is not representative of arbitrary production transactions without separate validation.')
    flags.append('Predictions alone cannot verify feature leakage, evidence provenance, or training/serving feature parity.')

    summary = {
        'input': str(INPUT), 'rows': n, 'fraud_cases': fraud_n,
        'cleared_cases': n - fraud_n, 'test_fraud_rate': overall_rate,
        'roc_auc': auc, 'brier_score': brier, 'ece_10_equal_width': ece,
        'probability_distribution': {
            'min': float(p.min()), 'median': float(np.median(p)),
            'mean': float(p.mean()), 'p90': float(np.quantile(p, .9)),
            'max': float(p.max()),
        },
        'diagnostic_groups': {
            'fraud_below_10_percent': len(low_misses),
            'cleared_at_least_90_percent': len(high_false_alarms),
            'all_below_10_percent': len(low),
            'all_10_to_below_90_percent': len(middle),
            'all_at_least_90_percent': len(high),
        },
        'adequately_populated_band_count': len(stable),
        'observed_rate_decreases_among_adequately_populated_bands': monotonic_violations,
        'calibration_bands': rows,
        'flags': flags,
        'scope': 'held-out historical closed cases; case-level outcome probability',
        'thresholds_are_diagnostics_not_decision_rules': True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(OUT / 'final_probability_calibration.csv', index=False)
    review.to_csv(OUT / 'final_probability_error_cases.csv', index=False)
    (OUT / 'final_probability_audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    notes = [
        'FINAL PROBABILITY AUDIT — RAPID REVIEW',
        '=' * 50,
        f'Test cases: {n:,} | Fraud: {fraud_n:,} ({overall_rate:.2%}) | Cleared: {n-fraud_n:,}',
        f'ROC-AUC: {auc:.6f}' if auc is not None else 'ROC-AUC: unavailable (single class)',
        f'Brier: {brier:.6f} | ECE (10 equal-width bands): {ece:.6f}',
        f'Fraud below 10%: {len(low_misses)} | Cleared at/above 90%: {len(high_false_alarms)}',
        f'Middle [10%,90%) cases: {len(middle)} | High [90%,100%] cases: {len(high)}',
        '', 'CALIBRATION BANDS (n<30 = sparse)',
        calibration.to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '', 'FLAGS / LIMITATIONS',
        *[f'- {flag}' for flag in flags],
        '', 'NEXT ACTION (TIME-BOXED)',
        'Inspect the highest confidence_error rows in final_probability_error_cases.csv.',
        'If no feature leakage or obvious serving mismatch appears, freeze this model for the demo.',
        'Integrate using exactly the same feature derivation and feature ordering as training.',
        'Label outputs as historical closed-case, case-level estimates; do not claim production calibration.',
    ]
    (OUT / 'final_probability_audit_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')

    print('\n'.join(notes[:7]))
    print('\nFiles created:')
    for filename in ['final_probability_calibration.csv', 'final_probability_error_cases.csv',
                     'final_probability_audit_summary.json', 'final_probability_audit_notes.txt']:
        print(' ', OUT / filename)


if __name__ == '__main__':
    main()
