#!/usr/bin/env python3
"""Reviewed source-only FP16-parameter support for the Haas MLX timing pilot.

Three pieces live here so that the resident startup inside the timed pilot and
the separate standalone audit share literally one cast helper:

- cast_parameters_to_float16: the narrow, strict BF16 -> FP16 parameter cast.
  It flattens the parameter tree by stable dot names with
  mlx.utils.tree_flatten, requires every leaf to be bfloat16 before the pinned
  MLX nn.Module.set_dtype(mx.float16) call and float16 after it, requires the
  same names and shapes before and after, refuses any tree other than the
  pinned model totals, and returns an honest profile (master_dtype is None:
  no BF16 master copy is retained; operator structure, chunking and decode
  policy are unchanged, while operator arithmetic follows the actual dtypes
  of its inputs). There is no alternate code path when validation fails; the
  helper raises.
- verify_audit_artifact: the small admission check the pilot controller runs
  on a supplied --mlx-precision-audit artifact before any benchmark child
  starts, against the identities the pilot verified itself.
- run_audit: the standalone, untimed, offline numeric-audit CLI. The pilot
  never reruns it on timed startup.

Importing this module (directly or via resident/pilot) works on the system
Python 3.9 with no MLX and no numpy: mlx, numpy, the port and pilot are
imported lazily inside functions, and pilot is imported only lazily to avoid
a circular eager import. Diagnostics are portable: no absolute personal
paths, no tensor values and no tensor data are stored, only names, shapes,
counts and digests.

Numeric treatment of the BF16 -> FP16 parameter cast (audit only), defined
once so the counts cannot be read two ways:

- The source leaf is widened exactly: BF16 memory words are read with
  array.view(mx.uint16), shifted left by 16 as NumPy uint32 and reinterpreted
  as float32 (NumPy has no universal bfloat16). The destination leaf is
  widened exactly with np.asarray(tensor).astype(np.float32), which preserves
  observed FP16 subnormals. Stored values are compared directly, on the CPU,
  with array-wise reductions, one tensor of temporary data at a time.
- numeric_unequal: pairs finite on both sides whose float32 values differ;
  +0.0 and -0.0 compare equal (IEEE 754 equality).
- underflow_to_zero: finite nonzero source whose destination is exactly zero,
  signed or unsigned; a subset of numeric_unequal.
- fp16_subnormal: finite nonzero destination with magnitude below 2^-14 that
  the cast kept as a subnormal; may or may not be unequal.
- overflow_to_infinity: finite source whose destination is infinite; not a
  finite-unequal pair.
- nonfinite_before / nonfinite_after: nonfinite values on either side, however
  they arose (NaN or infinity in the source, or produced by the cast).
- max_finite_abs_difference: maximum |source - destination| computed in
  float64 over pairs finite on both sides; 0.0 when there is no such pair.
- Eligibility: any nonfinite source or destination value, or any overflow,
  records a diagnostic and exits nonzero; such a cast is never eligible for
  the timing pilot. Underflow, kept subnormals and changed-but-finite values
  are reported as counts and never affect eligibility; the audit is numeric
  evidence only, independent of ASR, and claims no accuracy, CER or timing
  result.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path

HERE = Path(__file__).resolve().parent

PRECISION_PROFILE = 'fp16-parameters'
AUDIT_SCHEMA = 'haas-mlx-fp16-audit-1'
SOURCE_PARAMETER_DTYPE = 'bfloat16'
CAST_PARAMETER_DTYPE = 'float16'
ARITHMETIC_NOTE = ('parameter storage only; operator structure, chunking and decode policy are '
                   'unchanged, while operator arithmetic follows the actual dtypes of its inputs '
                   '(no claim that all arithmetic is unchanged or float16)')
PINNED_PARAMETER_TENSORS = 731
PINNED_PARAMETER_ELEMENTS = 615871520
SHAPE_SAMPLE_SIZE = 3
FP16_SUBNORMAL_BOUND = 2.0 ** -14
TOTAL_KEYS = ('numeric_unequal', 'underflow_to_zero', 'fp16_subnormal',
              'overflow_to_infinity', 'nonfinite_before', 'nonfinite_after')
EXPECTED_AUDIT_DTYPES = {'source': SOURCE_PARAMETER_DTYPE, 'parameters': CAST_PARAMETER_DTYPE,
                         'master': None, 'arithmetic': ARITHMETIC_NOTE}
AUDIT_EXIT_OK = 0
AUDIT_EXIT_REFUSED = 2      # verification, import, identity or cast refusal
AUDIT_EXIT_INELIGIBLE = 3   # nonfinite source/destination or overflow in the cast


class PrecisionError(RuntimeError):
    """A refused cast or a refused audit artifact; there is no fallback path."""


def cast_parameters_to_float16(model, mx, tree_flatten,
                               expected_tensors=None, expected_elements=None):
    """Cast every parameter leaf of the pinned all-BF16 HaasModel to float16.

    This is the one shared cast, used by the resident startup and by the
    standalone audit. It calls the pinned MLX nn.Module.set_dtype(mx.float16)
    and validates strictly: every leaf must be bfloat16 before and float16
    after, the stable tree_flatten names and shapes must be unchanged, and
    the totals must be the pinned model's. The BF16 leaves are referenced
    only inside this call, so the caller retains no old arrays. Returns the
    honest profile dict for the ready message / audit artifact."""
    if expected_tensors is None:
        expected_tensors = PINNED_PARAMETER_TENSORS
    if expected_elements is None:
        expected_elements = PINNED_PARAMETER_ELEMENTS
    before = list(tree_flatten(model.parameters()))
    if not before:
        raise PrecisionError('refusing to cast an empty parameter tree')
    for name, leaf in before:
        if leaf.dtype != mx.bfloat16:
            raise PrecisionError(
                f'parameter {name} is {leaf.dtype!r}, not bfloat16; the pinned checkpoint is all BF16')
    shapes = {name: list(leaf.shape) for name, leaf in before}
    model.set_dtype(mx.float16)
    after = list(tree_flatten(model.parameters()))
    if [name for name, _ in after] != [name for name, _ in before]:
        raise PrecisionError('parameter names changed across the cast')
    for (name, leaf), shape in zip(after, shapes.values()):
        if leaf.dtype != mx.float16:
            raise PrecisionError(f'parameter {name} stayed {leaf.dtype!r}; every leaf must be float16')
        if list(leaf.shape) != shape:
            raise PrecisionError(f'parameter {name} changed shape across the cast')
    tensors = len(after)
    elements = sum(math.prod(shape) for shape in shapes.values())
    if (tensors, elements) != (expected_tensors, expected_elements):
        raise PrecisionError(
            f'parameter totals {tensors}/{elements} do not match the pinned model '
            f'{expected_tensors}/{expected_elements}')
    return {
        'profile': PRECISION_PROFILE,
        'cast': True,
        'source_dtype': SOURCE_PARAMETER_DTYPE,
        'parameter_dtype': CAST_PARAMETER_DTYPE,
        'master_dtype': None,
        'arithmetic': ARITHMETIC_NOTE,
        'tensor_count': tensors,
        'element_count': elements,
        'shape_sample': [[name, shapes[name]] for name, _ in before[:SHAPE_SAMPLE_SIZE]],
    }


def mlx_runtime_identity(mx):
    """The honest MLX and Python runtime identity, shared by the resident
    ready message and the audit (the audit strips the executable path)."""
    device = mx.default_device()
    kind = getattr(device, 'type', device)
    kind = getattr(kind, 'name', None) or str(kind)
    return {
        'mlx': {'version': distribution_version('mlx'),
                'mlx_metal_version': distribution_version('mlx-metal'),
                'device': str(kind), 'device_name': mx.device_info()['device_name']},
        'python': {'executable': sys.executable, 'version': platform.python_version(),
                   'machine': platform.machine()},
    }


def verify_runtime_identity(mx):
    """Build the runtime identity and validate it with the pilot's ready pins
    (MLX/metal versions, GPU default device, machine and Python version)."""
    import pilot  # lazy: no circular eager import

    identity = mlx_runtime_identity(mx)
    pilot.verify_mlx_runtime({'type': 'ready', 'runtime': 'haas-mlx',
                              'protocol': pilot.MLX_PROTOCOL, **identity})
    return identity


def compare_widened_leaves(np, source_words, destination_f16):
    """Compare one BF16 source leaf (uint16 memory words) with its cast FP16
    destination leaf, both widened exactly to float32; see the module
    docstring for the defined treatment of each count."""
    source = (np.asarray(source_words).astype(np.uint32) << np.uint32(16)).view(np.float32)
    destination = np.asarray(destination_f16).astype(np.float32)
    if source.shape != destination.shape:
        raise PrecisionError('source and destination leaves disagree in shape')
    finite_source = np.isfinite(source)
    finite_destination = np.isfinite(destination)
    both_finite = finite_source & finite_destination
    counts = {
        'elements': int(source.size),
        'numeric_unequal': int(np.count_nonzero(both_finite & (source != destination))),
        'underflow_to_zero': int(np.count_nonzero(
            finite_source & (source != 0.0) & (destination == 0.0))),
        'fp16_subnormal': int(np.count_nonzero(
            finite_destination & (destination != 0.0) & (np.abs(destination) < FP16_SUBNORMAL_BOUND))),
        'overflow_to_infinity': int(np.count_nonzero(finite_source & np.isinf(destination))),
        'nonfinite_before': int(np.count_nonzero(~finite_source)),
        'nonfinite_after': int(np.count_nonzero(~finite_destination)),
    }
    if np.any(both_finite):
        difference = (source[both_finite].astype(np.float64)
                      - destination[both_finite].astype(np.float64))
        counts['max_finite_abs_difference'] = float(np.max(np.abs(difference)))
    else:
        counts['max_finite_abs_difference'] = 0.0
    return counts


def numeric_totals(rows):
    totals = {key: sum(row[key] for row in rows) for key in TOTAL_KEYS}
    totals['max_finite_abs_difference'] = max(
        (row['max_finite_abs_difference'] for row in rows), default=0.0)
    totals['tensors_compared'] = len(rows)
    totals['elements_compared'] = sum(row['elements'] for row in rows)
    return totals


def numeric_diagnostics(rows, limit=8):
    """Small recorded diagnostics for disqualifying tensors; no values stored."""
    found = []
    for row in rows:
        for kind in ('overflow_to_infinity', 'nonfinite_before', 'nonfinite_after'):
            if row[kind]:
                found.append({'tensor': row['name'], 'kind': kind, 'count': row[kind]})
                if len(found) >= limit:
                    return found
    return found


def audit_eligible(totals):
    return not (totals['overflow_to_infinity'] or totals['nonfinite_before'] or totals['nonfinite_after'])


def verify_audit_artifact(audit, expected):
    """Strict admission of a supplied precision-audit artifact.

    expected carries the controller's own already-verified identities:
    port and model (the pilot.verify_mlx_source / verify_mlx_model results),
    runner (SHA-256 of the current resident.py and precision.py, so a stale
    audit is refused) and runtime (the pinned MLX/Python/device identity).
    Every per-tensor row must be a uniquely named, well-formed accounting
    (positive integer elements, non-boolean integer counts within it,
    underflow within numeric_unequal, a finite nonnegative max), and
    numeric_totals of those rows must reproduce the declared tensors,
    elements, counts and max exactly. A bad, partial, stale or missing audit
    is refused; there is no alternate acceptance path. Returns a small
    admitted summary."""
    if not isinstance(audit, dict):
        raise PrecisionError('the precision audit artifact must be a JSON object')
    if audit.get('schema') != AUDIT_SCHEMA:
        raise PrecisionError(f'audit schema {audit.get("schema")!r} is not {AUDIT_SCHEMA!r}')
    if audit.get('status') != 'complete' or audit.get('eligible') is not True:
        raise PrecisionError('the precision audit artifact is partial or not eligible')
    if audit.get('profile') != PRECISION_PROFILE or audit.get('cast') is not True:
        raise PrecisionError(f'the audit must record the {PRECISION_PROFILE} cast')
    counts = audit.get('counts')
    if counts != {'tensors': PINNED_PARAMETER_TENSORS, 'elements': PINNED_PARAMETER_ELEMENTS}:
        raise PrecisionError(f'audit counts {counts!r} are not the pinned model totals '
                             f'{PINNED_PARAMETER_TENSORS}/{PINNED_PARAMETER_ELEMENTS}')
    dtypes = audit.get('dtypes')
    if dtypes != EXPECTED_AUDIT_DTYPES:
        raise PrecisionError(f'audit dtypes {dtypes!r} do not record the honest cast profile')
    numeric = audit.get('numeric')
    if not isinstance(numeric, dict):
        raise PrecisionError('the audit artifact lacks its numeric section')
    for key in TOTAL_KEYS:
        value = numeric.get(key)
        if type(value) is not int or value < 0:
            raise PrecisionError(f'audit numeric {key} must be a nonnegative integer')
    for key in ('overflow_to_infinity', 'nonfinite_before', 'nonfinite_after'):
        if numeric[key]:
            raise PrecisionError(f'audit reports {key}; such a cast is never eligible')
    if (numeric.get('tensors_compared') != PINNED_PARAMETER_TENSORS
            or numeric.get('elements_compared') != PINNED_PARAMETER_ELEMENTS):
        raise PrecisionError('the audit did not compare every pinned parameter tensor and element')
    per_tensor = numeric.get('per_tensor')
    if not isinstance(per_tensor, list) or len(per_tensor) != PINNED_PARAMETER_TENSORS:
        raise PrecisionError('the audit per-tensor accounting is incomplete')
    seen_names = set()
    for row in per_tensor:
        if not isinstance(row, dict):
            raise PrecisionError('each per-tensor entry must be a JSON object')
        name = row.get('name')
        if not isinstance(name, str) or not name:
            raise PrecisionError('each per-tensor entry needs a nonempty string name')
        if name in seen_names:
            raise PrecisionError(f'per-tensor entry {name!r} appears more than once')
        seen_names.add(name)
        elements = row.get('elements')
        if type(elements) is not int or elements <= 0:
            raise PrecisionError(f'tensor {name!r} needs a positive integer element count')
        for key in TOTAL_KEYS:
            value = row.get(key)
            if type(value) is not int or not 0 <= value <= elements:
                raise PrecisionError(f'tensor {name!r} {key} must be an integer in '
                                     f'[0, {elements}]')
        if row['underflow_to_zero'] > row['numeric_unequal']:
            raise PrecisionError(f'tensor {name!r} underflow_to_zero exceeds numeric_unequal')
        difference = row.get('max_finite_abs_difference')
        if (isinstance(difference, bool) or not isinstance(difference, (int, float))
                or not math.isfinite(difference) or difference < 0):
            raise PrecisionError(f'tensor {name!r} max_finite_abs_difference must be a '
                                 f'finite nonnegative number')
    difference = numeric.get('max_finite_abs_difference')
    if (isinstance(difference, bool) or not isinstance(difference, (int, float))
            or not math.isfinite(difference) or difference < 0):
        raise PrecisionError('audit max_finite_abs_difference must be a finite nonnegative number')
    totaled = numeric_totals(per_tensor)
    for key in (*TOTAL_KEYS, 'max_finite_abs_difference', 'tensors_compared', 'elements_compared'):
        if numeric[key] != totaled[key]:
            raise PrecisionError(f'audit numeric {key} is {numeric[key]!r} but the per-tensor '
                                 f'rows total {totaled[key]!r}; a partial or forged accounting '
                                 f'is refused even when the declared totals look right')
    if numeric.get('diagnostics'):
        raise PrecisionError('an eligible audit cannot carry disqualifying diagnostics')
    identity = audit.get('identity')
    if not isinstance(identity, dict):
        raise PrecisionError('the audit artifact lacks its identity section')
    for section in ('port', 'model'):
        if identity.get(section) != expected[section]:
            raise PrecisionError(f'the audit {section} identity does not match the verified '
                                 f'{section}; it is stale or foreign')
    if identity.get('runner') != expected['runner']:
        raise PrecisionError('the audit was produced by different resident/precision sources; '
                             'it is stale')
    runtime = identity.get('runtime')
    if not isinstance(runtime, dict):
        raise PrecisionError('the audit artifact lacks its runtime identity')
    mlx = runtime.get('mlx')
    if not isinstance(mlx, dict) or 'gpu' not in str(mlx.get('device', '')).lower():
        raise PrecisionError('the audit did not run on a GPU device')
    for key, want in (('version', expected['runtime']['mlx']['version']),
                      ('mlx_metal_version', expected['runtime']['mlx']['mlx_metal_version']),
                      ('device_name', expected['runtime']['mlx']['device_name'])):
        if mlx.get(key) != want:
            raise PrecisionError(f'audit runtime mlx.{key} is {mlx.get(key)!r}, not the pinned {want!r}')
    python = runtime.get('python')
    if (not isinstance(python, dict)
            or python.get('version') != expected['runtime']['python']['version']
            or python.get('machine') != expected['runtime']['python']['machine']):
        raise PrecisionError('the audit Python identity is not the pinned MLX environment')
    if identity.get('module_path_verified') is not True:
        raise PrecisionError('the audit did not verify the port module path against the checkout')
    return {'schema': AUDIT_SCHEMA, 'profile': PRECISION_PROFILE, 'eligible': True,
            'counts': dict(counts),
            'numeric': {key: numeric[key] for key in (*TOTAL_KEYS, 'max_finite_abs_difference')},
            'runner': dict(expected['runner']),
            'created_utc': audit.get('created_utc')}


def _refuse(message):
    print(f'precision: {message}', file=sys.stderr, flush=True)
    return AUDIT_EXIT_REFUSED


def _portable_runtime(identity):
    """Drop the interpreter path; the artifact carries no personal paths."""
    python = {key: value for key, value in identity['python'].items() if key != 'executable'}
    return {'mlx': identity['mlx'], 'python': python}


def run_audit(argv=None):
    """The standalone untimed offline numeric audit; see the module docstring."""
    parser = argparse.ArgumentParser(
        description='Untimed offline numeric audit of the reviewed BF16->FP16 parameter '
                    'cast; writes one fresh audit artifact and never runs inside the pilot.')
    parser.add_argument('--mlx-source', required=True, help='pinned Haas MLX port Git checkout')
    parser.add_argument('--mlx-model-dir', required=True, help='local converted MLX model directory')
    parser.add_argument('--mlx-model-manifest', required=True, help='verified pinned model file manifest')
    parser.add_argument('--out', required=True, help='fresh output directory (never overwritten)')
    args = parser.parse_args(argv)

    for label, value, kind in [('mlx-source', args.mlx_source, 'directory'),
                               ('mlx-model-dir', args.mlx_model_dir, 'directory'),
                               ('mlx-model-manifest', args.mlx_model_manifest, 'file')]:
        path = Path(value)
        if not (path.is_file() if kind == 'file' else path.is_dir()):
            raise SystemExit(f'precision: --{label} must be an existing {kind}: {value}')

    import pilot  # lazy: reuse the pilot's source/model verification, hashes and pins

    pilot.require_no_mlx_tweaks()
    source_id = pilot.verify_mlx_source(args.mlx_source)
    model_id = pilot.verify_mlx_model(args.mlx_model_manifest, args.mlx_model_dir)

    src_root = (Path(args.mlx_source) / 'src').resolve()
    sys.path.insert(0, str(src_root))  # only the requested checkout's port
    try:
        import mlx.core as mx
        import numpy as np
        import haas_mlx
        from haas_mlx.model import HaasModel
        from mlx.utils import tree_flatten
    except ImportError as error:
        return _refuse(f'the pinned port or its MLX environment could not be imported: {error}')

    # The public supplied port must really be the module tree being audited.
    module_file = Path(haas_mlx.__file__).resolve()
    try:
        module_relative = module_file.relative_to(src_root)
    except ValueError:
        return _refuse(f'haas_mlx was imported from {module_file}, not the requested checkout '
                       f'{src_root}; refusing to audit a foreign port')
    if not mx.metal.is_available():
        return _refuse('MLX reports no Metal device; the audit requires the real GPU')
    try:
        runtime = verify_runtime_identity(mx)
    except RuntimeError as error:
        return _refuse(f'the runtime identity is not the pinned MLX environment: {error}')

    try:
        model = HaasModel.load(args.mlx_model_dir)
    except Exception as error:
        return _refuse(f'the pinned model failed to load: {error}')
    before = list(tree_flatten(model.parameters()))
    mx.eval([leaf for _, leaf in before])  # stable BF16 values before the cast
    try:
        profile = cast_parameters_to_float16(model, mx, tree_flatten)
    except PrecisionError as error:
        return _refuse(f'the parameter cast was refused: {error}')
    after = dict(tree_flatten(model.parameters()))

    rows = []
    for name, source_leaf in before:  # one tensor of CPU temporaries at a time
        rows.append({'name': name,
                     **compare_widened_leaves(np, source_leaf.view(mx.uint16), after[name])})
    totals = numeric_totals(rows)
    diagnostics = numeric_diagnostics(rows)
    eligible = audit_eligible(totals)
    artifact = {
        'schema': AUDIT_SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete',
        'eligible': eligible,
        'profile': profile['profile'],
        'cast': profile['cast'],
        'counts': {'tensors': profile['tensor_count'], 'elements': profile['element_count']},
        'dtypes': {'source': profile['source_dtype'], 'parameters': profile['parameter_dtype'],
                   'master': profile['master_dtype'], 'arithmetic': profile['arithmetic']},
        'numeric': {
            'method': 'exact widened float32 CPU comparison of the stored values, array-wise, '
                      'one tensor at a time; signed zeros compare equal',
            **totals,
            'per_tensor': rows,
            'diagnostics': diagnostics,
        },
        'identity': {
            'port': source_id,
            'model': model_id,
            'runtime': _portable_runtime(runtime),
            'runner': {'resident_sha256': pilot.sha(HERE / 'resident.py'),
                       'precision_sha256': pilot.sha(HERE / 'precision.py')},
            'haas_mlx_module': str(module_relative),
            'module_path_verified': True,
        },
        'scope': 'untimed offline numeric audit of the reviewed BF16->FP16 parameter cast; '
                 'no accuracy, CER or timing evidence',
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # fresh output only, never overwritten
    artifact_path = out / 'audit.json'
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    print(f'precision audit {"eligible" if eligible else "ineligible"}: {artifact_path}', flush=True)
    if not eligible:
        for diagnostic in diagnostics:
            print(f"precision: {diagnostic['kind']} in {diagnostic['tensor']} "
                  f"({diagnostic['count']} values)", file=sys.stderr, flush=True)
        return AUDIT_EXIT_INELIGIBLE
    return AUDIT_EXIT_OK


def main(argv=None):
    return run_audit(argv)


if __name__ == '__main__':
    sys.exit(main())
