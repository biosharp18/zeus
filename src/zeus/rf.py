"""Map the common generation request to RFdiffusion's native Hydra runner."""
from pathlib import Path
import json
import sys


def overrides(request):
    symmetry = request["symmetry"].lower()
    symmetry = {"t": "tetrahedral", "o": "octahedral", "i": "icosahedral"}.get(symmetry, symmetry)
    mode = {"full": "null", "zeus": "restricted", "ablation": "null_chainbreak_ablate"}[request["mode"]]
    output = Path(request["output"])
    args = [
        "inference.symmetry=" + symmetry,
        "contigmap.contigs=" + json.dumps([str(request["symmetry_order"] * request["subunit_length"])]),
        "inference.ckpt_override_path=" + json.dumps(request["checkpoint"]),
        "inference.output_prefix=" + json.dumps(str(output / "design")),
        "inference.num_designs=" + str(request["num_designs"]),
        "inference.design_startnum=0", "inference.write_trajectory=false",
        "inference.deterministic=true", "inference.seed=" + str(request["seed"]),
        "inference.radius=" + str(request["radius"]),
        "inference.schedule_directory_path=" + json.dumps(str(Path(request["cache_dir"]) / "rfdiffusion")),
        "inference.chain_symmetric_features=" + str(request["chain_symmetric_features"]).lower(),
        "diffuser.T=" + str(request["steps"]),
        "hydra.run.dir=" + json.dumps(str(output / "hydra")),
        "hydra.job.chdir=false",
    ]
    # The native checkpoint loader casts explicit overrides to their current
    # Python type; casting an explicit null through NoneType would fail.
    # Full mode is already the packaged default (null), so leave it unset.
    if request["mode"] != "full":
        args.append("model.symmetry_mode=" + mode)
    return args


def run(request):
    from zeus_rfdiffusion.run import main
    sys.argv = ["zeus-rfdiffusion"] + overrides(request)
    main()
