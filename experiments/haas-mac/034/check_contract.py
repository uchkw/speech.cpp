"""Checks reference identity, shape guards, the disk plan and the TDT admission gates with small standard-library fixtures.

Usage: python3 -B check_contract.py
No package installation, model, array computation, compiler, git query or subprocess runs.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from check_tdt import validate_admission, verify_coverage, verify_runtime_state
from contract import (CHECKER_SOURCES, DISK_PLAN, GIB, PINS, REFERENCE_FILES, disk_decision, load_manifest,
                      model_identity, output_directory, selected_cases, sha256, validate_meta,
                      verify_reference_files, verify_source_snapshot, write_json)


def meta(case):
    return {"reference_scope": "frontend-encoder-joint.enc", "model": copy.deepcopy(PINS["checkpoint"]),
            "dtype": "float32", "device": "cpu", "nemo": "2.7.3", "sample_rate": 16000, "samples": 1000,
            "frames": case["t"], "encoded_frames": case["u"], "encoder_input_frames": case["p"], "encoder_output_frames": case["q"],
            "arrays": {"features": [case["t"], 128], "encoded": [case["u"], 1024], "joint_encoded": [case["u"], 640]}}


def admission_records():
    identity = {"model.mil": "0" * 64}
    manifest = {"cases": [{"id": "one"}, {"id": "two"}], "model_identities": {"f32": {"normal": identity}}}
    checked = {"manifest_sha256": "m", "compute_units": "CPU_AND_NE", "model_identities": {"f32/normal": identity}}
    native = {"manifest_sha256": "m", "compute_units": "CPU_AND_NE", "precision": "f32", "mode": "predict",
              "model_identity": identity}
    parity = {"manifest_sha256": "m", "all_exact": True, "precision": "f32", "compute_units": "CPU_AND_NE",
              "rows": [{"id": "one", "bit_exact": True, "python_npy_sha256": "1" * 64, "native_raw_sha256": "2" * 64},
                       {"id": "two", "bit_exact": True, "python_npy_sha256": "3" * 64, "native_raw_sha256": "4" * 64}]}
    return manifest, checked, native, parity


class ContractChecks(unittest.TestCase):
    def test_fixed_pair_preserves_two_valid_lengths(self):
        cases = selected_cases("fixed")
        self.assertEqual([(row["p"], row["t"], row["u"]) for row in cases], [(1008, 996, 125), (1008, 1002, 126)])
        for case in cases:
            validate_meta(case, meta(case))

    def test_enumerated_shapes_keep_all_eleven_inputs(self):
        cases = selected_cases("enumerated")
        self.assertEqual(len(cases), 11)
        self.assertEqual(len({row["p"] for row in cases}), 10)
        for case in cases:
            validate_meta(case, meta(case))

    def test_bad_reference_is_rejected(self):
        case = selected_cases("fixed")[0]
        mutations = {
            "scope": lambda row: row.update(reference_scope="full-model"),
            "precision": lambda row: row.update(dtype="float16"),
            "device": lambda row: row.update(device="cuda"),
            "version": lambda row: row.update(nemo="3.0.0"),
            "checkpoint": lambda row: row["model"].update(revision="other"),
            "frozen_length": lambda row: row.update(frames=1002),
            "frozen_output_length": lambda row: row.update(encoded_frames=126),
            "unpadded_input": lambda row: row.update(encoder_input_frames=996),
            "cropped_output_shape": lambda row: row.update(encoder_output_frames=125),
            "empty_audio": lambda row: row.update(samples=0),
            "transposed_features": lambda row: row["arrays"].update(features=[128, case["t"]]),
            "wrong_projection_width": lambda row: row["arrays"].update(joint_encoded=[case["u"], 1024]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = meta(case)
                mutate(value)
                with self.assertRaises(ValueError):
                    validate_meta(case, value)

    def test_manifest_does_not_collapse_shared_shapes(self):
        value = {"schema": 1, "experiment": "034", "pins": PINS, "shape_set": "fixed", "cases": selected_cases("fixed")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json(path, value)
            self.assertEqual(len(load_manifest(path)["cases"]), 2)
            value["cases"] = value["cases"][:1]
            other = Path(directory) / "missing.json"
            write_json(other, value)
            with self.assertRaises(ValueError):
                load_manifest(other)

    def test_source_directory_cannot_hold_artifacts(self):
        with self.assertRaises(ValueError):
            output_directory(Path(__file__).resolve().parent / "outputs" / "not-created")

    def test_disk_plan_budgets_each_invocation_once(self):
        # Every production invocation is budgeted for what it alone still adds; the already-present
        # archive and dumps are budgeted by no step, and export is the step that writes the packages.
        self.assertEqual(set(DISK_PLAN), {"environment", "export", "check", "native", "instruments"})
        for step, plan in DISK_PLAN.items():
            self.assertGreater(plan["reserve"], 0, step)
            self.assertNotIn("weights", plan, step)
        self.assertGreater(DISK_PLAN["export"]["packages"], 0)
        self.assertGreater(DISK_PLAN["export"]["conversion_temp"], 0)

    def test_each_step_rejects_free_space_just_below_its_plan(self):
        for step, plan in DISK_PLAN.items():
            boundary = sum(plan.values()) * GIB
            with self.subTest(step=step):
                self.assertEqual(disk_decision(step, boundary)["total_bytes"], boundary)
                with self.assertRaisesRegex(ValueError, step):
                    disk_decision(step, boundary - 1)
        with self.assertRaisesRegex(ValueError, "disk plan"):
            disk_decision("unknown-step", 10 ** 15)

    def test_package_identity_detects_tampering_and_rejects_empty_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "model.mlpackage"
            (package / "weights").mkdir(parents=True)
            (package / "model.mil").write_text("mil", encoding="utf-8")
            (package / "weights" / "weight.bin").write_bytes(b"weights")
            identity = model_identity(package)
            self.assertEqual(set(identity), {"model.mil", "weights/weight.bin"})
            (package / "weights" / "weight.bin").write_bytes(b"tampered")
            self.assertNotEqual(model_identity(package), identity)
            empty = Path(directory) / "empty.mlpackage"
            empty.mkdir()
            with self.assertRaises(ValueError):
                model_identity(empty)

    def test_reference_file_tampering_is_rejected(self):
        case = {"reference_files": {}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in REFERENCE_FILES:
                (directory / name).write_bytes(name.encode())
                case["reference_files"][name] = sha256(directory / name)
            verify_reference_files(case, directory)
            (directory / "features.npy").write_bytes(b"other")
            with self.assertRaises(ValueError):
                verify_reference_files(case, directory)
            (directory / "audio.npy").unlink()
            with self.assertRaises(ValueError):
                verify_reference_files(case, directory)

    def test_source_snapshot_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            snapshot = {}
            for name in CHECKER_SOURCES:
                (directory / name).write_text(name, encoding="utf-8")
                snapshot[name] = sha256(directory / name)
            manifest = {"source_sha256": snapshot}
            verify_source_snapshot(manifest, directory)
            (directory / "contract.py").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_source_snapshot(manifest, directory)
            (directory / "pins.json").unlink()
            with self.assertRaises(ValueError):
                verify_source_snapshot(manifest, directory)
            incomplete = {"source_sha256": {name: value for name, value in snapshot.items() if name != "check.py"}}
            with self.assertRaises(ValueError):
                verify_source_snapshot(incomplete, directory)

    def test_tdt_admission_rejects_broken_or_mismatched_records(self):
        pinned = "b" * 64
        manifest, checked, native, parity = admission_records()
        self.assertEqual(validate_admission(manifest, checked, native, parity, "m", pinned, pinned)["cases"], ["one", "two"])
        with self.assertRaises(ValueError):
            validate_admission(*admission_records(), "other", pinned, pinned)
        with self.assertRaises(ValueError):
            validate_admission(*admission_records(), "m", "c" * 64, pinned)
        with self.assertRaises(ValueError):
            validate_admission(*admission_records(), "m", pinned, None)

        def rejects(message, mutate):
            manifest, checked, native, parity = admission_records()
            mutate(manifest, checked, native, parity)
            with self.subTest(message), self.assertRaises(ValueError):
                validate_admission(manifest, checked, native, parity, "m", pinned, pinned)

        rejects("parity not exact", lambda *rows: rows[3].update(all_exact=False))
        rejects("parity precision", lambda *rows: rows[3].update(precision="f16"))
        rejects("parity units", lambda *rows: rows[3].update(compute_units="ALL"))
        rejects("parity mode", lambda *rows: rows[2].update(mode="trace"))
        rejects("native units", lambda *rows: rows[2].update(compute_units="CPU_ONLY"))
        rejects("manifest mismatch", lambda *rows: rows[2].update(manifest_sha256="other"))
        rejects("parity from another manifest", lambda *rows: rows[3].update(manifest_sha256="other"))
        rejects("missing last case", lambda *rows: rows[3]["rows"].pop())
        rejects("swapped cases", lambda *rows: rows[3]["rows"].reverse())
        rejects("duplicate case", lambda *rows: rows[3]["rows"].append(dict(rows[3]["rows"][0])))
        rejects("inexact case", lambda *rows: rows[3]["rows"][0].update(bit_exact=False))
        rejects("package identity", lambda *rows: rows[2].update(model_identity={"model.mil": "9" * 64}))

    def test_tdt_output_coverage_requires_both_sides_and_true_decisions_in_order(self):
        pairs = [(Path("/d/one/features.npy"), Path("/c/one/joint_encoded.npy")),
                 (Path("/d/two/features.npy"), Path("/c/two/joint_encoded.npy"))]

        def stdout(mutate):
            lines = [
                {"source": "ggml_normal_encoder", "input": str(pairs[0][0])},
                {"source": "coreml_normal_encoder", "input": str(pairs[0][1])},
                {"input": str(pairs[0][0]), "exact_tdt_decisions": True},
                {"source": "ggml_normal_encoder", "input": str(pairs[1][0])},
                {"source": "coreml_normal_encoder", "input": str(pairs[1][1])},
                {"input": str(pairs[1][0]), "exact_tdt_decisions": True},
            ]
            mutate(lines)
            return "\n".join(json.dumps(row) for row in lines)

        verify_coverage(stdout(lambda lines: None), pairs)
        for message, mutate in (
                ("missing decision", lambda lines: lines.pop()),
                ("missing decoding", lambda lines: lines.pop(0)),
                ("false decision", lambda lines: lines[-1].update(exact_tdt_decisions=False)),
                ("swapped decision inputs", lambda lines: lines[2].update(input=str(pairs[1][0]))),
                ("swapped sides", lambda lines: lines[0].update(source="coreml_normal_encoder")),
                ("another input", lambda lines: lines[3].update(input=str(pairs[0][0]))),
        ):
            with self.subTest(message), self.assertRaises(ValueError):
                verify_coverage(stdout(mutate), pairs)

    def test_tdt_runtime_state_guards(self):
        def state(**changes):
            value = {"head": "e" * 40, "base_is_ancestor": True, "runtime_clean": True,
                     "runtime_at_base": True, "ggml_gitlink": PINS["ggml_base"],
                     "ggml_working": PINS["ggml_base"], "ggml_clean": True}
            value.update(changes)
            return value

        receipt = {"runtime": {"head": "e" * 40, "ggml_gitlink": PINS["ggml_base"],
                               "ggml_working": PINS["ggml_base"]}}
        verify_runtime_state(state())
        verify_runtime_state(state(), receipt)
        for message, changes in (
                ("not a descendant", {"base_is_ancestor": False}),
                ("dirty runtime", {"runtime_clean": False}),
                ("runtime differs from the pinned base", {"runtime_at_base": False}),
                ("ggml gitlink moved", {"ggml_gitlink": "0" * 40}),
                ("ggml working tree moved", {"ggml_working": "0" * 40}),
                ("dirty ggml", {"ggml_clean": False}),
        ):
            with self.subTest(message), self.assertRaises(ValueError):
                verify_runtime_state(state(**changes))
        with self.assertRaises(ValueError):
            verify_runtime_state(state(head="f" * 40), receipt)


if __name__ == "__main__":
    unittest.main()
