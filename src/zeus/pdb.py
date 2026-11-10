"""Backbone PDB I/O shared by generation and refolding."""
from pathlib import Path
import math
import string

CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits
AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))


def write_backbone(path, atom37, subunit_length):
    """Write glycine backbones, with one chain per physical subunit."""
    if len(atom37) % subunit_length:
        raise ValueError("backbone length is not divisible by subunit length")
    lines, serial = [], 1
    for i, residue in enumerate(atom37):
        chain = CHAIN_IDS[i // subunit_length]
        resnum = i % subunit_length + 1
        # OpenFold atom37 order: N, CA, C, CB, O, ...
        for name, index in [("N", 0), ("CA", 1), ("C", 2), ("O", 4)]:
            x, y, z = map(float, residue[index])
            if not all(math.isfinite(v) and -999.999 <= v <= 9999.999 for v in [x, y, z]):
                raise ValueError("nonfinite coordinates or coordinates outside PDB field limits")
            lines.append(f"ATOM  {serial:5d} {name:^4s} GLY {chain}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {name[0]:>2s}  ")
            serial += 1
        if resnum == subunit_length:
            lines.append(f"TER   {serial:5d}      GLY {chain}{resnum:4d}")
            serial += 1
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def read_ca(path):
    """Return chains in file order, each with sequence and CA coordinates."""
    chains, seen = {}, set()
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA" or line[16] not in {" ", "A"}:
            continue
        chain = line[21]
        key = (chain, line[22:27])
        if key in seen:
            continue
        seen.add(key)
        data = chains.setdefault(chain, {"sequence": "", "ca": []})
        data["sequence"] += AA3.get(line[17:20], "X")
        data["ca"].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not chains:
        raise ValueError("no alpha carbons found in " + str(path))
    return chains

