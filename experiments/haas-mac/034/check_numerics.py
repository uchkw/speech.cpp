"""Exercises the stage check's numeric failure cases with tiny arrays, without Core ML or Torch.

Usage: python -B check_numerics.py
Requires the separately resolved NumPy environment; this is not a model accuracy check.
"""

import unittest

import numpy as np

from numerics import difference, output_arrays, passes


class NumericChecks(unittest.TestCase):
    def test_exact_and_rounding_bound(self):
        expected = np.ones(8, dtype=np.float32)
        self.assertTrue(passes(difference(expected, expected), 90))
        self.assertTrue(passes(difference(expected + np.float32(1e-6), expected), 90))
        self.assertFalse(passes(difference(expected + np.float32(1e-2), expected), 90))

    def test_signed_zero_is_not_bit_exact(self):
        diff = difference(np.array([-0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
        self.assertTrue(diff["exact"])
        self.assertFalse(diff["bit_exact"])

    def test_zero_reference_does_not_pass_nonzero_error(self):
        self.assertFalse(passes(difference(np.ones(1, dtype=np.float32), np.zeros(1, dtype=np.float32)), 25))

    def test_nonfinite_output_and_reference_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            bad = np.array([value], dtype=np.float32)
            good = np.ones(1, dtype=np.float32)
            for actual, expected in ((bad, good), (good, bad)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    difference(actual, expected)

    def test_dtype_and_shape_fail(self):
        good = np.ones((2, 3), dtype=np.float32)
        for bad in (good.astype(np.float16), good.T, np.ones((0, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                difference(bad, good)

    def test_valid_length_crops_only_after_padded_validation(self):
        case = {"u": 1, "q": 2}
        values = {"joint_encoded": np.zeros((1, 2, 640), dtype=np.float32), "encoded_length": np.array([1], dtype=np.int32)}
        self.assertEqual(output_arrays(values, case)["joint_encoded"].shape, (1, 640))
        values["joint_encoded"][0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            output_arrays(values, case)

    def test_frozen_valid_length_fails(self):
        values = {"joint_encoded": np.zeros((1, 126, 640), dtype=np.float32), "encoded_length": np.array([126], dtype=np.int32)}
        with self.assertRaises(ValueError):
            output_arrays(values, {"u": 125, "q": 126})


if __name__ == "__main__":
    unittest.main()
