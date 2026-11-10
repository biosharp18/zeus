"""ProteinMPNN sequence design followed by complex-level ESMFold refolding."""
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from .pdb import read_ca


def tied_positions(chains):
    lengths = {len(data["sequence"]) for data in chains.values()}
    if len(lengths) != 1:
        raise ValueError("tied homomer design requires equal chain lengths; use --untied for independent chains")
    return [
        {chain: [i] for chain in chains} for i in range(1, next(iter(lengths)) + 1)]


def designed_sequences(path, expected, chain_lengths):
    """Skip ProteinMPNN's native template; preserve its '/' chain separators."""
    records, header, chunks = [], None, []
    for line in Path(path).read_text().splitlines() + [">"]:
        if line.startswith(">"):
            if header is not None and (header.startswith("T=") or ", T=" in header):
                records.append("".join(chunks).replace("/", ":"))
            header, chunks = line[1:], []
        elif line.strip():
            chunks.append(line.strip())
    if len(records) != expected:
        raise ValueError(f"ProteinMPNN produced {len(records)} sequences; expected {expected}")
    for sequence in records:
        if [len(chain) for chain in sequence.split(":")] != chain_lengths:
            raise ValueError("designed sequence chain lengths do not match the input backbone")
        if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY:"):
            raise ValueError("ProteinMPNN returned unsupported amino acid symbols")
    return records


def score_complex(reference, prediction):
    import numpy as np
    from tmtools import tm_align

    original = read_ca(reference)
    folded = read_ca(prediction)
    if [len(v["sequence"]) for v in original.values()] != [len(v["sequence"]) for v in folded.values()]:
        raise ValueError("refolded structure does not match the reference chain lengths")
    xyz1 = np.asarray([xyz for chain in original.values() for xyz in chain["ca"]], dtype=np.float64)
    xyz2 = np.asarray([xyz for chain in folded.values() for xyz in chain["ca"]], dtype=np.float64)
    seq1 = "".join(chain["sequence"] for chain in original.values())
    seq2 = "".join(chain["sequence"] for chain in folded.values())
    result = tm_align(xyz1, xyz2, seq1, seq2)
    return {"scRMSD": float(result.rmsd), "scTM": float(max(result.tm_norm_chain1, result.tm_norm_chain2))}


def design_sequences(request, pdb_path, work):
    chains = read_ca(pdb_path)
    # ProteinMPNN writes designed chains in sorted chain-label order. Reorder
    # the reference to the same order for subsequent complex-level alignment.
    chain_order = sorted(chains)
    source = work / "backbones"
    source.mkdir(parents=True)
    backbone = source / "backbone.pdb"
    lines = Path(pdb_path).read_text().split("ENDMDL", 1)[0].splitlines()
    backbone.write_text("\n".join(
        line for chain in chain_order for line in lines
        if line.startswith(("ATOM  ", "TER   ")) and line[21:22] == chain
    ) + "\nEND\n")
    parsed = work / "parsed_pdbs.jsonl"
    subprocess.run([sys.executable, "-m", "zeus_protein_mpnn.helper_scripts.parse_multiple_chains",
                    "--input_path", str(source), "--output_path", str(parsed)], check=True)
    command = [sys.executable, "-m", "zeus_protein_mpnn.protein_mpnn_run",
               "--jsonl_path", str(parsed), "--out_folder", str(work / "mpnn"),
               "--path_to_model_weights", request["mpnn_weights"], "--model_name", "v_48_020",
               "--num_seq_per_target", str(request["num_sequences"]), "--batch_size", "1",
               "--sampling_temp", str(request["temperature"]), "--seed", str(request["seed"])]
    if not request["untied"] and len(chains) > 1:
        tied = work / "tied_positions.jsonl"
        tied.write_text(json.dumps({"backbone": tied_positions(chains)}) + "\n")
        command.extend(["--tied_positions_jsonl", str(tied)])
    subprocess.run(command, check=True)
    fasta = work / "mpnn" / "seqs" / "backbone.fa"
    sequences = designed_sequences(fasta, request["num_sequences"], [len(chains[c]["sequence"]) for c in chain_order])
    if not request["untied"] and len(chains) > 1 and any(len(set(seq.split(":"))) != 1 for seq in sequences):
        raise ValueError("ProteinMPNN failed to produce tied homomer sequences")
    shutil.copy2(fasta, work / "sequences.fa")
    return backbone, sequences


def run(request):
    import numpy as np
    import torch
    import esm

    output = Path(request["output"])
    rows, summary = [], []
    model = None
    for index, pdb_path in enumerate(request["pdbs"]):
        work = output / f"{index:04d}_{Path(pdb_path).stem}"
        work.mkdir()
        backbone, sequences = design_sequences(request, pdb_path, work)
        if model is None:
            random.seed(request["seed"])
            np.random.seed(request["seed"])
            torch.manual_seed(request["seed"])
            model = esm.pretrained.esmfold_v1().eval().to(request["device"])
            model.set_chunk_size(request["chunk_size"])
        predictions = work / "refolds"
        predictions.mkdir()
        scores = []
        for sample_index, sequence in enumerate(sequences):
            destination = predictions / f"sample_{sample_index:02d}.pdb"
            with torch.no_grad():
                pdb_string = model.infer_pdb(sequence, num_recycles=request["recycles"])
            destination.write_text(pdb_string)
            score = score_complex(backbone, destination)
            row = {"backbone": str(pdb_path), "sample": sample_index, "sequence": sequence,
                   "refold": str(destination.relative_to(output)), **score}
            rows.append(row)
            scores.append(row)
            _write_csv(output / "scores.csv", rows)
        best = min(scores, key=lambda row: row["scRMSD"])
        summary.append({"backbone": str(pdb_path), "best_scRMSD": best["scRMSD"],
                        "best_scTM": max(row["scTM"] for row in scores),
                        "best_rmsd_sample": best["sample"], "designable_scRMSD_lt_2": best["scRMSD"] < 2.0})
        _write_csv(output / "summary.csv", summary)
        print(f"Refolded {pdb_path}: best scRMSD={best['scRMSD']:.3f}", flush=True)


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
