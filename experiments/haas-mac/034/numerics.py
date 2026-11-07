"""Checks finite stage arrays and reports SNR without changing a failing value or bound."""

import math

import numpy as np

from contract import require


def array(path, shape):
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and list(value.shape) == list(shape), "Unexpected F32 array: " + str(path))
    require(np.isfinite(value).all(), "Nonfinite reference array: " + str(path))
    return np.ascontiguousarray(value)


def difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.dtype == np.float32 and actual.shape == expected.shape and actual.size > 0, "Output dtype or shape differs.")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite output or reference.")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    signal = float(np.square(expected.astype(np.float64)).sum())
    noise = float(np.square(error).sum())
    # JSON has no infinity; exact equality is represented independently of SNR.
    exact = noise == 0
    snr = None if exact or signal == 0 else 10 * math.log10(signal / noise)
    return {"exact": exact, "bit_exact": actual.tobytes(order="C") == expected.tobytes(order="C"),
            "snr_db": snr, "zero_reference": signal == 0,
            "max_abs": float(np.abs(error).max()), "elements": int(actual.size)}


def passes(diff, threshold):
    return diff["exact"] or (diff["snr_db"] is not None and diff["snr_db"] > threshold)


def stage(actual, expected, threshold, label):
    diff = difference(actual, expected)
    require(passes(diff, threshold), f"{label} fails {threshold} dB: {diff}")
    return diff


def output_arrays(values, case, diagnostic=False, joint=False):
    projected = np.asarray(values["joint_encoded"])
    if joint:
        require(projected.shape == (1, case["u"], 640), "Isolated joint output shape differs.")
        return {"joint_encoded": projected[0]}
    length = np.asarray(values["encoded_length"])
    require(length.dtype == np.int32 and length.shape == (1,) and int(length[0]) == case["u"], "Encoded length differs.")
    require(projected.dtype == np.float32 and projected.shape == (1, case["q"], 640), "Composed output shape differs.")
    require(np.isfinite(projected).all(), "Nonfinite padded projection output.")
    result = {"joint_encoded": projected[0, :case["u"]]}
    if diagnostic:
        encoded = np.asarray(values["encoded"])
        require(encoded.dtype == np.float32 and encoded.shape == (1, 1024, case["q"]), "Encoder output shape differs.")
        require(np.isfinite(encoded).all(), "Nonfinite padded encoder output.")
        result["encoded"] = np.ascontiguousarray(encoded[0, :, :case["u"]].T)
    return result
