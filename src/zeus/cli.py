"""The CLI intentionally imports no model or tensor libraries."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def symmetry_info(value):
    value = value.upper()
    match = re.fullmatch(r"C([1-9][0-9]*)", value)
    if match:
        return "C", int(match.group(1))
    if value in {"T", "O", "I"}:
        return value, {"T": 12, "O": 24, "I": 60}[value]
    raise ValueError("symmetry must be Ck (e.g. C3), T, O, or I")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    device_help = "Set CUDA_VISIBLE_DEVICES to select a GPU, or to an empty string for CPU execution."
    p = argparse.ArgumentParser(description="ZeUS symmetric protein generation and refolding", epilog=device_help)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate symmetric backbone PDBs", epilog=device_help)
    g.add_argument("--model", required=True, choices=["framediff", "foldflow", "rfdiffusion"])
    g.add_argument("--mode", choices=["full", "zeus", "ablation"], default="zeus")
    g.add_argument("--symmetry", default="C6", help="Ck, T, O, or I (default: C6)")
    g.add_argument("--subunit-length", type=positive_int, default=50)
    g.add_argument("--num-designs", type=positive_int, default=1)
    g.add_argument("--checkpoint", required=True, type=Path)
    g.add_argument("--steps", type=positive_int, help="Default: 500 for FrameDiff, 50 otherwise")
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--radius", type=float, help="IPA symmetry pivot x-coordinate (default 11.55 A); RF symmetry radius (default 10 A)")
    g.add_argument("--face-offset-radius", type=float, default=0.0, help="IPA polyhedral placement parameter")
    g.add_argument("--chain-symmetric-features", action="store_true", help="RFdiffusion: rebuild termini/torsions per chain; independent of --mode")
    r = sub.add_parser("refold", help="Design sequences with ProteinMPNN and refold with ESMFold", epilog=device_help)
    r.add_argument("--input", required=True, type=Path, help="A backbone PDB or directory of backbone PDBs")
    r.add_argument("--mpnn-weights", required=True, type=Path, help="Directory containing v_48_020.pt")
    r.add_argument("--num-sequences", type=positive_int, default=8)
    r.add_argument("--temperature", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=38)
    r.add_argument("--untied", action="store_true", help="Design chains independently; default ties matching positions across equal-length chains")
    r.add_argument("--chunk-size", type=positive_int, default=128, help="ESMFold axial attention chunk size")
    r.add_argument("--recycles", type=int, default=3, help="ESMFold recycling iterations")
    for cmd in [g, r]:
        cmd.add_argument("--output", required=True, type=Path, help="New or empty output directory")
        cmd.add_argument("--python", default=sys.executable, help="Python interpreter in the selected runtime environment")
        cmd.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "zeus")
        cmd.add_argument("--dry-run", action="store_true", help="Print the request and command without loading models or writing files")
    return p


def request_from_args(args):
    request = {k: str(v.expanduser().resolve()) if isinstance(v, Path) else v
               for k, v in vars(args).items() if k not in {"python", "dry_run"}}
    if args.command == "generate":
        kind, order = symmetry_info(args.symmetry)
        if order > 62 or args.subunit_length > 9999 or 4 * order * args.subunit_length + order > 99999:
            raise ValueError("requested structure exceeds this CLI's PDB format limits")
        request.update(symmetry=args.symmetry.upper(), symmetry_type=kind, symmetry_order=order)
        request["steps"] = args.steps or (500 if args.model == "framediff" else 50)
        request["radius"] = args.radius if args.radius is not None else (10.0 if args.model == "rfdiffusion" else 11.55)
        if request["steps"] < 2:
            raise ValueError("at least two sampling steps are required")
        if args.model == "rfdiffusion" and request["steps"] < 15:
            raise ValueError("RFdiffusion's discrete schedule requires at least 15 sampling steps")
        if not all(math.isfinite(request[key]) for key in ["radius", "face_offset_radius"]):
            raise ValueError("placement parameters must be finite")
        if args.model != "rfdiffusion" and (args.mode == "ablation" or args.chain_symmetric_features):
            raise ValueError("ablation and chain-symmetric-features apply only to RFdiffusion")
        if args.model == "rfdiffusion" and args.face_offset_radius != 0:
            raise ValueError("face-offset-radius applies only to FrameDiff and FoldFlow")
        if args.model == "rfdiffusion" and order * args.subunit_length > 9999:
            raise ValueError("RFdiffusion's PDB residue numbering supports at most 9999 total residues")
        if not Path(request["checkpoint"]).is_file():
            raise ValueError("checkpoint does not exist: " + request["checkpoint"])
    else:
        if not math.isfinite(args.temperature) or args.temperature <= 0 or args.recycles < 0:
            raise ValueError("temperature must be positive and recycles must be nonnegative")
        source = Path(request["input"])
        pdbs = [source] if source.is_file() and source.suffix.lower() == ".pdb" else sorted(source.glob("*.pdb")) if source.is_dir() else []
        if not pdbs:
            raise ValueError("input must contain at least one backbone PDB")
        request["pdbs"] = [str(p) for p in pdbs]
        if not (Path(request["mpnn_weights"]) / "v_48_020.pt").is_file():
            raise ValueError("mpnn-weights must contain v_48_020.pt")
    return request


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        request = request_from_args(args)
        command = [args.python, "-m", "zeus.worker", json.dumps(request)]
        if args.dry_run:
            print(json.dumps({"request": request, "command": command}, indent=2))
            return 0
        output = Path(request["output"])
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("output must be a new or empty directory: " + str(output))
        output.mkdir(parents=True, exist_ok=True)
        metadata_path = output / (args.command + ".json")
        metadata = dict(request, status="running")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt):
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            raise
        metadata["status"] = "complete"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print("Wrote " + str(output))
        return 0
    except (ValueError, OSError) as error:
        p.error(str(error))
    except subprocess.CalledProcessError as error:
        print("Runtime failed; see the error above. Check that --python selects the installed runtime environment.", file=sys.stderr)
        return error.returncode

