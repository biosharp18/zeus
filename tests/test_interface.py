import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from zeus import cli, ipa, refold, rf, worker
from zeus.pdb import read_ca, write_backbone
from zeus.refold import designed_sequences, tied_positions


class InterfaceTests(unittest.TestCase):
    def test_backbone_round_trip_retains_physical_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backbone.pdb"
            coords = [[[float(i), float(a), 1.0] for a in range(37)] for i in range(6)]
            write_backbone(path, coords, subunit_length=3)
            chains = read_ca(path)
            self.assertEqual(list(chains), ["A", "B"])
            self.assertEqual([v["sequence"] for v in chains.values()], ["GGG", "GGG"])
            self.assertEqual(chains["B"]["ca"][0], [3.0, 1.0, 1.0])
            self.assertEqual(tied_positions(chains), [
                {"A": [1], "B": [1]}, {"A": [2], "B": [2]}, {"A": [3], "B": [3]}])

    def test_fasta_skips_native_and_keeps_multichain_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "seqs.fa"
            fasta.write_text(">backbone, score=1\nGGG/GGG\n>T=0.1, sample=1\nACD/\nACD\n>T=0.1, sample=2\nFEG/FEG\n")
            self.assertEqual(designed_sequences(fasta, 2, [3, 3]), ["ACD:ACD", "FEG:FEG"])
            with self.assertRaisesRegex(ValueError, "expected 3"):
                designed_sequences(fasta, 3, [3, 3])
            with self.assertRaisesRegex(ValueError, "chain lengths"):
                designed_sequences(fasta, 2, [6])

    def test_unequal_chains_are_not_silently_tied(self):
        with self.assertRaisesRegex(ValueError, "equal chain lengths"):
            tied_positions({"A": {"sequence": "AAA"}, "B": {"sequence": "AA"}})

    def test_refuse_nonempty_output_without_launching_a_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint = tmp / "weights.pt"
            checkpoint.touch()
            out = tmp / "output"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("existing work")
            with patch("zeus.cli.subprocess.run") as launch, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.main(["generate", "--model", "framediff", "--checkpoint", str(checkpoint), "--output", str(out)])
                launch.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing work")

    def test_dry_run_needs_no_model_imports_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint = tmp / "weights.pt"
            checkpoint.touch()
            out = tmp / "new-output"
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), patch("zeus.cli.subprocess.run") as launch:
                code = cli.main(["generate", "--model", "rfdiffusion", "--checkpoint", str(checkpoint), "--output", str(out), "--dry-run"])
                launch.assert_not_called()
            request = json.loads(capture.getvalue())["request"]
            self.assertEqual(code, 0)
            self.assertEqual(request["symmetry"], "C6")
            self.assertEqual(request["symmetry_order"], 6)
            self.assertEqual(request["steps"], 50)
            self.assertNotIn("device", request)
            self.assertFalse(out.exists())

    def test_worker_uses_visible_devices_and_preserves_environment(self):
        routes = [("generate", "framediff", ipa), ("generate", "foldflow", ipa),
                  ("generate", "rfdiffusion", rf), ("refold", None, refold)]
        devices = [("3,1", True, "cuda:0"), ("", False, "cpu"), (None, False, "cpu")]
        with tempfile.TemporaryDirectory() as tmp:
            for command, model, adapter in routes:
                for visible, available, expected_device in devices:
                    with self.subTest(command=command, model=model, visible=visible):
                        request = {"command": command, "cache_dir": tmp}
                        if model is not None:
                            request["model"] = model
                        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=available)))
                        with patch.dict(os.environ), patch.dict(sys.modules, {"torch": torch}), \
                                patch.object(sys, "argv", ["zeus.worker", json.dumps(request)]), \
                                patch.object(adapter, "run") as run:
                            if visible is None:
                                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = visible
                            worker.main()
                            run.assert_called_once_with(dict(request, device=expected_device))
                            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), visible)

    def test_rf_ablation_is_independent_of_torsion_flags(self):
        request = dict(symmetry="C3", symmetry_order=3, subunit_length=50, mode="ablation",
                       output="/tmp/designs", checkpoint="/tmp/weights.pt", num_designs=1,
                       seed=123, radius=10.0, cache_dir="/tmp/cache", chain_symmetric_features=False, steps=50)
        overrides = rf.overrides(request)
        self.assertIn("model.symmetry_mode=null_chainbreak_ablate", overrides)
        self.assertIn("inference.chain_symmetric_features=false", overrides)
        contigs = next(value.split("=", 1)[1] for value in overrides if value.startswith("contigmap.contigs="))
        self.assertEqual(json.loads(contigs), ["150"])
        self.assertFalse(any(value.startswith("model.symmetry_mode=") for value in rf.overrides(dict(request, mode="full"))))
        self.assertIn("model.symmetry_mode=restricted", rf.overrides(dict(request, mode="zeus")))


if __name__ == "__main__":
    unittest.main()
