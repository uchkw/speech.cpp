#!/usr/bin/env python3
"""Numerical and admission checks for precision.py.

The numerical fixtures use NumPy (already pinned inside the isolated MLX
Python 3.12 environment; run this file with that interpreter) and nothing
else: source leaves are synthetic exact BF16 memory words and destinations
are NumPy float16 arrays, so there is no MLX import, no subprocess, no model
and no file dataset anywhere here. The admission checks exercise the
audit-artifact contract with plain dictionaries: stale, foreign, partial,
forged or missing evidence must be refused. No test asserts any measurement
result; the eligibility assertions only restate the audited numeric rule.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precision  # noqa: E402

try:
    import numpy as np
except ImportError:  # the admission checks below still run without numpy
    np = None

PORT = {'commit': 'cde96d263f56aad75c1de70282f71a91562092dd', 'tree': 'e' * 40,
        'worktree_clean': True}
MODEL = {'revision': '15ef4f11618c6a1ee50e0eafaae13beded8ecdf9',
         'files': {'model.safetensors': {'sha256': '7' * 64, 'size': 1231236320},
                   'tokenizer.model': {'sha256': '9' * 64, 'size': 512352},
                   'config.json': {'sha256': '6' * 64, 'size': 812,
                                   'git_blob_sha1': '638a6704459b3041d8122c08ae872f90464edd49'}},
         'manifest_sha256': 'a' * 64}
RUNNER = {'resident_sha256': 'b' * 64, 'precision_sha256': 'c' * 64}
RUNTIME = {'mlx': {'version': '0.32.4', 'mlx_metal_version': '0.32.4', 'device': 'gpu',
                   'device_name': 'Apple M2 Ultra'},
           'python': {'version': '3.12.11', 'machine': 'arm64'}}


def expected():
    return {'port': PORT, 'model': MODEL, 'runner': RUNNER, 'runtime': RUNTIME}


def per_tensor_row(name, elements=8, **counts):
    row = {'name': name, 'elements': elements, 'numeric_unequal': 0, 'underflow_to_zero': 0,
           'fp16_subnormal': 0, 'overflow_to_infinity': 0, 'nonfinite_before': 0,
           'nonfinite_after': 0, 'max_finite_abs_difference': 0.0}
    row.update(counts)
    return row


def per_tensor_rows():
    """731 synthetic rows consistent with the pinned totals: 730 small rows
    plus one large symbolic final row carrying every nonzero count, so the
    declared 615871520 elements, the count fields and the max all match the
    rows exactly."""
    return [per_tensor_row(f'p{i}') for i in range(730)] + [
        per_tensor_row('final', 615871520 - 730 * 8, numeric_unequal=12345,
                       underflow_to_zero=12, fp16_subnormal=34,
                       max_finite_abs_difference=0.25)]


def artifact(**overrides):
    """A finished, honest, eligible audit artifact for the pinned FP16 profile.

    The pinned totals are written as literals on purpose: drifting constants
    must fail admission here rather than slip through."""
    base = {
        'schema': 'haas-mlx-fp16-audit-1', 'status': 'complete', 'eligible': True,
        'profile': 'fp16-parameters', 'cast': True,
        'counts': {'tensors': 731, 'elements': 615871520},
        'dtypes': {'source': 'bfloat16', 'parameters': 'float16', 'master': None,
                   'arithmetic': precision.ARITHMETIC_NOTE},
        'numeric': {'tensors_compared': 731, 'elements_compared': 615871520,
                    'numeric_unequal': 12345, 'underflow_to_zero': 12, 'fp16_subnormal': 34,
                    'overflow_to_infinity': 0, 'nonfinite_before': 0, 'nonfinite_after': 0,
                    'max_finite_abs_difference': 0.25,
                    'per_tensor': per_tensor_rows(),
                    'diagnostics': []},
        'identity': {'port': PORT, 'model': MODEL, 'runtime': RUNTIME, 'runner': RUNNER,
                     'haas_mlx_module': 'haas_mlx/__init__.py', 'module_path_verified': True},
        'created_utc': '2026-11-07T00:00:00+00:00',
    }
    base.update(overrides)
    return base


def bf16_words(values):
    """Exact BF16 memory words for values representable in BF16."""
    return (np.array(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def as_f16(values):
    """The observed destination: float32 values converted to float16."""
    return np.array(values, dtype=np.float32).astype(np.float16)


@unittest.skipIf(np is None, 'numpy is pinned only in the isolated MLX environment')
class WidenedComparisonTest(unittest.TestCase):
    """compare_widened_leaves on synthetic BF16 words and NumPy float16."""

    def compare(self, sources, destinations):
        return precision.compare_widened_leaves(np, bf16_words(sources), as_f16(destinations))

    def test_signed_zeros_and_exact_values(self):
        # 65280, unlike 65504, is exactly representable in both BF16 and float16
        row = self.compare([0.0, -0.0, 1.0, -1.0, 0.5, 65280.0, 2.0 ** -14],
                           [0.0, 0.0, 1.0, -1.0, 0.5, 65280.0, 2.0 ** -14])
        for key in precision.TOTAL_KEYS:
            self.assertEqual(row[key], 0, key)  # a -0 source with a +0 destination stays equal
        self.assertEqual(row['max_finite_abs_difference'], 0.0)

    def test_fp16_subnormal_2_to_minus_24_is_kept_exactly(self):
        row = self.compare([2.0 ** -24, -2.0 ** -24], [2.0 ** -24, -2.0 ** -24])
        self.assertEqual(row['fp16_subnormal'], 2)
        self.assertEqual(row['numeric_unequal'], 0)
        self.assertEqual(row['underflow_to_zero'], 0)
        self.assertEqual(row['max_finite_abs_difference'], 0.0)
        self.assertEqual(as_f16([2.0 ** -24]).view(np.uint16).tolist(),
                         [0x0001])  # a real stored subnormal, not a zero

    def test_bf16_2_to_minus_25_underflows_to_signed_or_unsigned_zero(self):
        row = self.compare([2.0 ** -25, -2.0 ** -25], [2.0 ** -25, -2.0 ** -25])
        self.assertEqual(row['underflow_to_zero'], 2)
        self.assertEqual(row['numeric_unequal'], 2)
        self.assertEqual(row['fp16_subnormal'], 0)
        self.assertEqual(as_f16([2.0 ** -25, -2.0 ** -25]).view(np.uint16).tolist(),
                         [0x0000, 0x8000])  # unsigned and signed zero, sign preserved

    def test_overflow_at_65536_stops_the_audit(self):
        self.assertTrue(np.isinf(as_f16([65536.0])[0]))  # BF16-exact 65536 overflows float16
        row = precision.compare_widened_leaves(np, bf16_words([65536.0]),
                                               np.array([np.inf], dtype=np.float16))
        self.assertEqual(row['overflow_to_infinity'], 1)
        self.assertEqual(row['nonfinite_after'], 1)
        self.assertEqual(row['numeric_unequal'], 0)  # overflow is not a finite-unequal pair
        rows = [{'name': 'w', **row}]
        self.assertFalse(precision.audit_eligible(precision.numeric_totals(rows)))
        self.assertEqual([diagnostic['kind'] for diagnostic in precision.numeric_diagnostics(rows)],
                         ['overflow_to_infinity', 'nonfinite_after'])  # one overflow reports both

    def test_nonfinite_source_or_destination_stops_the_audit(self):
        words = np.array([0x7F80, 0x7FC0], dtype=np.uint16)  # BF16 +inf and a NaN
        row = precision.compare_widened_leaves(
            np, words, np.array([np.inf, np.nan], dtype=np.float16))
        self.assertEqual((row['nonfinite_before'], row['nonfinite_after']), (2, 2))
        rows = [{'name': 'w', **row}]
        self.assertFalse(precision.audit_eligible(precision.numeric_totals(rows)))
        self.assertEqual({diagnostic['kind'] for diagnostic in precision.numeric_diagnostics(rows)},
                         {'nonfinite_before', 'nonfinite_after'})

    def test_max_finite_abs_difference_is_exact_in_f64(self):
        row = self.compare([1.0], [1.0 + 2.0 ** -10])  # the float16 neighbor of 1.0
        self.assertEqual(row['numeric_unequal'], 1)
        self.assertEqual(row['max_finite_abs_difference'], 2.0 ** -10)

    def test_totals_across_tensors_and_underflow_eligibility(self):
        rows = [{'name': f'w{i}',
                 **precision.compare_widened_leaves(np, bf16_words([2.0 ** -25]),
                                                    np.array([0.0], dtype=np.float16))}
                for i in range(3)]
        totals = precision.numeric_totals(rows)
        self.assertEqual(totals['tensors_compared'], 3)
        self.assertEqual(totals['elements_compared'], 3)
        self.assertEqual(totals['underflow_to_zero'], 3)
        self.assertEqual(totals['numeric_unequal'], 3)
        self.assertTrue(precision.audit_eligible(totals))  # underflow is reported, not disqualifying
        self.assertEqual(precision.numeric_diagnostics(rows), [])


class AuditAdmissionTest(unittest.TestCase):
    """verify_audit_artifact: exact identities, counts and eligibility only."""

    def admit(self, audit):
        return precision.verify_audit_artifact(audit, expected())

    def test_admits_a_completed_honest_audit(self):
        summary = self.admit(artifact())
        self.assertTrue(summary['eligible'])
        self.assertEqual(summary['counts'], {'tensors': 731, 'elements': 615871520})
        self.assertEqual(summary['numeric']['numeric_unequal'], 12345)  # changed values, still eligible
        self.assertEqual(summary['runner'], RUNNER)

    def test_rejects_bad_partial_or_missing_artifacts(self):
        for audit in [artifact(schema='haas-mlx-fp16-audit-0'),
                      artifact(status='partial'),
                      artifact(eligible=False),
                      artifact(profile='bf16-parameters'),
                      artifact(cast=False),
                      artifact(counts={'tensors': 730, 'elements': 615871520}),
                      artifact(counts={'tensors': 731, 'elements': 615871521}),
                      artifact(counts={'tensors': 731}),  # absent element count
                      {'schema': precision.AUDIT_SCHEMA},
                      None]:
            with self.assertRaises(precision.PrecisionError):
                self.admit(audit)

    def test_rejects_wrong_or_absent_dtype_and_master_claims(self):
        honest = artifact()['dtypes']
        for dtypes in [None,
                       {**honest, 'source': 'float32'},
                       {**honest, 'parameters': 'bfloat16'},
                       {**honest, 'master': 'bfloat16'},
                       {**honest, 'arithmetic': 'all arithmetic float16'}]:
            with self.assertRaises(precision.PrecisionError):
                self.admit(artifact(dtypes=dtypes))

    def test_rejects_disqualifying_numeric_evidence_even_if_marked_eligible(self):
        for key, value in [('overflow_to_infinity', 1), ('nonfinite_before', 1),
                           ('nonfinite_after', 2), ('tensors_compared', 730),
                           ('elements_compared', 615871519), ('numeric_unequal', -1),
                           ('max_finite_abs_difference', float('inf')),
                           ('diagnostics', [{'tensor': 'w', 'kind': 'overflow_to_infinity',
                                             'count': 1}])]:
            audit = artifact()
            audit['numeric'] = {**audit['numeric'], key: value}
            with self.assertRaises(precision.PrecisionError):
                self.admit(audit)
        audit = artifact()
        audit['numeric'] = {**audit['numeric'],
                            'per_tensor': audit['numeric']['per_tensor'][:-1]}
        with self.assertRaises(precision.PrecisionError):
            self.admit(audit)

    def test_rejects_corrupted_per_tensor_rows(self):
        def corrupted(rows, **declared):
            audit = artifact()
            audit['numeric'] = {**audit['numeric'], 'per_tensor': rows, **declared}
            return audit
        empty_row = per_tensor_rows()
        empty_row[3] = {}  # a present but empty entry: no name and no counts
        duplicate = per_tensor_rows()
        duplicate[7]['name'] = duplicate[6]['name']
        outside = per_tensor_rows()
        outside[2]['numeric_unequal'] = outside[2]['elements'] + 1  # beyond its own tensor
        for audit in [corrupted(empty_row),
                      corrupted(duplicate),
                      corrupted(outside),
                      corrupted(per_tensor_rows(), numeric_unequal=12346)]:  # declared != rows
            with self.assertRaises(precision.PrecisionError):
                self.admit(audit)

    def test_rejects_stale_or_foreign_identities(self):
        stale_runner = artifact()
        stale_runner['identity'] = {**stale_runner['identity'],
                                    'runner': {**RUNNER, 'precision_sha256': 'd' * 64}}
        stale_port = artifact()
        stale_port['identity'] = {**stale_port['identity'],
                                  'port': {**PORT, 'commit': 'f' * 40}}
        stale_model = artifact()
        stale_model['identity'] = {**stale_model['identity'],
                                   'model': {**MODEL, 'manifest_sha256': 'e' * 64}}
        foreign_runtime = artifact()
        foreign_runtime['identity'] = {
            **foreign_runtime['identity'],
            'runtime': {'mlx': {**RUNTIME['mlx'], 'version': '0.32.5'},
                        'python': RUNTIME['python']}}
        foreign_machine = artifact()
        foreign_machine['identity'] = {
            **foreign_machine['identity'],
            'runtime': {'mlx': RUNTIME['mlx'],
                        'python': {**RUNTIME['python'], 'machine': 'x86_64'}}}
        unverified_path = artifact()
        unverified_path['identity'] = {**unverified_path['identity'],
                                       'module_path_verified': False}
        for audit in [stale_runner, stale_port, stale_model, foreign_runtime,
                      foreign_machine, unverified_path]:
            with self.assertRaises(precision.PrecisionError):
                self.admit(audit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
