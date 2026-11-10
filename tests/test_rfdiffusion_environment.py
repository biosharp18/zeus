import importlib.util
import os
import subprocess
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("zeus_rfdiffusion"), "requires the RFdiffusion environment")
class RFdiffusionEnvironmentTests(unittest.TestCase):
    def test_native_graph_operations_without_external_cuda_search_path(self):
        environment = os.environ.copy()
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
            environment.pop(key, None)
        environment.update(CUDA_VISIBLE_DEVICES="", DGLBACKEND="pytorch", OMP_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", """
import torch
import zeus_rfdiffusion.inference.model_runners
import dgl
import dgl.function as fn

graph = dgl.graph(([0, 1], [1, 0]))
graph.ndata['value'] = torch.tensor([[2.0], [3.0]])
graph.update_all(fn.copy_u('value', 'message'), fn.sum('message', 'total'))
assert torch.equal(graph.ndata['total'], torch.tensor([[3.0], [2.0]]))
assert not torch.cuda.is_initialized()
"""], env=environment, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
