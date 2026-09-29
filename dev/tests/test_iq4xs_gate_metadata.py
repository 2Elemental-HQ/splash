"""Metadata screening only: these tests do not execute IQ4_XS Metal kernels."""

import tempfile
import unittest
from pathlib import Path

from dev.tests.test_gguf_metadata import GGML, fixture, loadable_tensors, write_gguf
from install import gguf, models


class Iq4XsGateMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def inputs(self, arch, layers):
        values = {
            key.replace("qwen35moe.", arch + "."): value
            for key, value in fixture().items()
            if arch == "qwen35moe" or not key.startswith("qwen35moe.expert")
        }
        values["general.architecture"] = arch
        values[arch + ".block_count"] = layers
        tensors = loadable_tensors(values, self.root)
        gates = [
            name
            for name in tensors
            if name.endswith((".ssm_alpha.weight", ".ssm_beta.weight"))
        ]
        self.assertEqual(len(gates), 2 * (layers - layers // 4))
        return values, tensors, gates

    def screen(self, values, tensors):
        path = write_gguf(self.root / "gates.gguf", values, tensors.items())
        gguf.require_loadable(gguf.Metadata(path, tensors=True))

    def test_matching_supported_pairs(self):
        for arch, layers in (("qwen35", 64), ("qwen35moe", 40)):
            for kind in ("IQ4_XS", "Q8_0", "F32", "BF16"):
                with self.subTest(architecture=arch, gate_type=kind):
                    values, tensors, gates = self.inputs(arch, layers)
                    for name in gates:
                        tensors[name] = GGML[kind]
                    self.screen(values, tensors)

    def test_mismatched_pairs_still_fail(self):
        for arch, layers in (("qwen35", 64), ("qwen35moe", 40)):
            with self.subTest(architecture=arch):
                values, tensors, gates = self.inputs(arch, layers)
                for name in gates:
                    tensors[name] = GGML["IQ4_XS"]
                tensors["blk.0.ssm_alpha.weight"] = GGML["Q8_0"]
                with self.assertRaisesRegex(models.ModelError, "of different types"):
                    self.screen(values, tensors)

    def test_unsupported_pairs_still_fail(self):
        for kind in ("F16", "Q5_0"):
            with self.subTest(gate_type=kind):
                values, tensors, _ = self.inputs("qwen35", 64)
                tensors["blk.0.ssm_alpha.weight"] = GGML[kind]
                tensors["blk.0.ssm_beta.weight"] = GGML[kind]
                with self.assertRaisesRegex(models.ModelError, "cannot load"):
                    self.screen(values, tensors)

    def test_missing_gate_still_fails(self):
        values, tensors, gates = self.inputs("qwen35", 64)
        for name in gates:
            tensors[name] = GGML["IQ4_XS"]
        del tensors["blk.0.ssm_alpha.weight"]
        with self.assertRaisesRegex(models.ModelError, "missing"):
            self.screen(values, tensors)


if __name__ == "__main__":
    unittest.main()
