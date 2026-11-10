"""Shared inference lifecycle for the independently packaged IPA runtimes."""
import importlib
import inspect
from pathlib import Path
import random


def load_runner(request):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    package = importlib.import_module("zeus_" + request["model"])
    module = importlib.import_module(package.__name__ + ".runner")
    conf = OmegaConf.load(Path(package.__file__).parent / "config.yaml")
    # These published checkpoints contain OmegaConf metadata as well as tensors.
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch.load(request["checkpoint"], **kwargs)
    if request["model"] == "framediff" and "conf" in checkpoint:
        # Resolve against the checkpoint's complete config before extracting
        # its model subtree; some older checkpoints refer to training fields.
        saved_model = OmegaConf.to_container(checkpoint["conf"].model, resolve=True)
        conf.model = OmegaConf.merge(conf.model, saved_model)
    # Keep the FoldFlow-1 architecture from its inference config. Its original
    # sampler also uses this config when loading the published base checkpoint.
    conf.model.symmetry_type = request["symmetry_type"]
    conf.model.symmetry = request["symmetry_order"]
    conf.model.efficient_symmetry = request["mode"] == "zeus"
    conf.model.diffuse_subunit_only = True
    conf.model.symmetry_origin_translation = [request["radius"], 0.0, 0.0]
    conf.model.face_offset_radius = request["face_offset_radius"]
    conf.data.num_t = request["steps"]
    if request["model"] == "framediff":
        conf.diffuser.so3.cache_dir = str(Path(request["cache_dir"]) / "framediff")
    np.random.seed(request["seed"])
    torch.manual_seed(request["seed"])
    runner = module.Runner(conf)
    weights = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
    if request["model"] == "foldflow":
        weights = {key.replace("score_model.", "vectorfield."): value for key, value in weights.items()}
    runner.model.load_state_dict(weights, strict=True)
    runner.model.to(request["device"]).eval()
    return runner, conf


def initial_features(runner, request):
    import numpy as np
    import torch

    length = request["subunit_length"] * request["symmetry_order"]
    dynamics = runner.diffuser if request["model"] == "framediff" else runner.flow_matcher
    feats = {
        "res_mask": np.ones(length),
        "seq_idx": torch.arange(1, length + 1),
        "fixed_mask": np.zeros(length),
        "torsion_angles_sin_cos": np.zeros((length, 7, 2)),
        "sc_ca_t": np.zeros((length, 3)),
        **dynamics.sample_ref(n_samples=length, as_tensor_7=True),
    }
    if request["model"] == "foldflow":
        feats["aatype"] = torch.zeros(length, dtype=torch.int32)
        feats["chain_idx"] = torch.zeros(length, dtype=torch.int32)
    return {key: torch.as_tensor(value)[None].to(request["device"]) for key, value in feats.items()}


def run(request):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from .pdb import write_backbone

    runner, conf = load_runner(request)
    output = Path(request["output"])
    OmegaConf.save(conf, output / "config.yaml", resolve=True)
    for index in range(request["num_designs"]):
        seed = request["seed"] + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        feats = initial_features(runner, request)
        sample = runner.inference_fn(feats, num_t=request["steps"], min_t=conf.data.min_t,
                                     aux_traj=False, noise_scale=0.1)
        destination = output / f"design_{index:04d}.pdb"
        write_backbone(destination, sample["prot_traj"][0, 0], request["subunit_length"])
        print("Wrote " + str(destination), flush=True)

