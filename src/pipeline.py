import sys
import os
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
VINA_PATH = os.path.join(BASE_DIR, "..", "vina")

DEFAULT_RECEPTOR_PATH = os.path.join(DATA_DIR, "structures", "receptor.pdbqt")
DEFAULT_BOX_CENTER = (22.014, 0.253, 52.794)

DOCK_CACHE_PATH = os.path.join(DATA_DIR, "docking_cache.json")

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
model = joblib.load(os.path.join(DATA_DIR, "models", "egfr_model.pkl"))


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(fp_gen.GetFingerprint(mol))


def largest_fragment(smiles):
    return max(smiles.split("."), key=len)


def get_molecule_image(smiles, size=(300, 300)):
    """
    Render a 2D skeletal structure image for a SMILES string using RDKit.
    Returns a PIL Image, or None if the SMILES can't be parsed.
    """
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def parse_atom_coords(pdb_text):
    """
    Extract (x, y, z) coordinates from every ATOM/HETATM line in a PDB or
    PDBQT text block.
    """
    coords = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    return coords


def parse_atoms(pdb_text):
    """
    Extract per-atom info (coordinates, element, residue name/number/chain)
    from every ATOM/HETATM line in a PDB or PDBQT text block. Falls back to
    guessing the element from the atom-name column if the dedicated element
    column (cols 77-78) is blank or missing, which PDBQT files often do.
    """
    atoms = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                atom_name = line[12:16].strip()
                letters = "".join(c for c in atom_name if c.isalpha())
                element = letters[:1].upper() if letters else "?"

            atoms.append({
                "coord": (x, y, z),
                "element": element,
                "resname": line[17:20].strip() if len(line) >= 20 else "",
                "chain": line[21].strip() if len(line) >= 22 else "",
                "resnum": line[22:26].strip() if len(line) >= 26 else "",
            })
    return atoms


def find_close_contacts_detailed(receptor_text, ligand_pose_text, cutoff=4.0):
    """
    Like find_close_contacts, but returns richer per-contact info: which
    receptor residue is involved, and a rough polar/hydrophobic guess based
    on element (both atoms N or O => "polar", otherwise "hydrophobic").
    This is still a geometric heuristic, not a validated interaction
    classification (see find_close_contacts docstring) -- "polar" here just
    means "both atoms are the kind that COULD hydrogen-bond", not that a
    bond is confirmed.

    Returns a list of dicts: {ligand_coord, receptor_coord, distance,
    residue (e.g. "ASP594"), chain, contact_type ("polar"/"hydrophobic")}.
    """
    receptor_atoms = parse_atoms(receptor_text)
    ligand_atoms = parse_atoms(ligand_pose_text)

    contacts = []
    for latom in ligand_atoms:
        lx, ly, lz = latom["coord"]
        best_dist = None
        best_atom = None
        for ratom in receptor_atoms:
            rx, ry, rz = ratom["coord"]
            dist = ((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_atom = ratom

        if best_dist is not None and best_dist <= cutoff:
            polar_elements = {"N", "O"}
            is_polar = latom["element"] in polar_elements and best_atom["element"] in polar_elements
            contacts.append({
                "ligand_coord": (lx, ly, lz),
                "receptor_coord": best_atom["coord"],
                "distance": best_dist,
                "residue": f"{best_atom['resname']}{best_atom['resnum']}",
                "chain": best_atom["chain"],
                "contact_type": "polar" if is_polar else "hydrophobic",
            })

    return contacts


def find_close_contacts(receptor_text, ligand_pose_text, cutoff=4.0):
    """
    For each ligand atom, find its single nearest receptor atom. If that
    distance is within `cutoff` (Angstroms), record it as a "close contact".

    This is a geometric PROXIMITY heuristic only -- it does not classify
    contacts by type (hydrogen bond, hydrophobic, salt bridge, etc.), which
    would require a dedicated tool like PLIP. It's meant for a quick visual
    sense of where a docked molecule sits close to the receptor, not a
    validated interaction analysis.

    Returns a list of (ligand_coord, receptor_coord, distance) tuples.
    """
    receptor_coords = parse_atom_coords(receptor_text)
    ligand_coords = parse_atom_coords(ligand_pose_text)

    contacts = []
    for lx, ly, lz in ligand_coords:
        best_dist = None
        best_coord = None
        for rx, ry, rz in receptor_coords:
            dist = ((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_coord = (rx, ry, rz)
        if best_dist is not None and best_dist <= cutoff:
            contacts.append(((lx, ly, lz), best_coord, best_dist))

    return contacts


def tanimoto_similarity(fp1, fp2):
    """
    Tanimoto similarity between two binary fingerprint arrays (as produced
    by smiles_to_fp). 1.0 = identical fingerprint, 0.0 = no shared bits.
    This is the standard chemical similarity metric used to compare
    molecular structures.
    """
    if fp1 is None or fp2 is None:
        return None
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def read_pocket(pocket_path):
    """Read a 'cx,cy,cz' pocket-center file written by prepare_target_structure.py."""
    with open(pocket_path) as f:
        cx, cy, cz = (float(v) for v in f.read().strip().split(","))
    return (cx, cy, cz)


def _load_dock_cache():
    if os.path.exists(DOCK_CACHE_PATH):
        with open(DOCK_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_dock_cache(cache):
    with open(DOCK_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _dock_cache_key(smiles, receptor_path):
    return f"{os.path.basename(receptor_path)}::{smiles}"


def get_pose_path(name, receptor_path):
    """
    Standardized output path for a docked pose, namespaced by both the
    receptor used and the molecule name, so the same molecule docked
    against different targets never overwrites a previous pose file.
    """
    receptor_tag = os.path.splitext(os.path.basename(receptor_path))[0]
    return os.path.join(DATA_DIR, "structures", f"{receptor_tag}__{name}_docked.pdbqt")


def extract_top_pose(pdbqt_path):
    """
    Extract just the top-scoring pose (MODEL 1) from a multi-model Vina
    output file, returning the ATOM/HETATM lines only (no MODEL/ENDMDL
    wrapper), ready to feed into a 3D viewer.
    """
    lines = []
    in_model_1 = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model_1 = line.split()[1] == "1"
                continue
            if line.startswith("ENDMDL"):
                if in_model_1:
                    break
                continue
            if in_model_1 and (line.startswith("ATOM") or line.startswith("HETATM")):
                lines.append(line)
    return "".join(lines)


def dock_smiles(smiles, name, receptor_path=None, box_center=None, box_size=(20, 20, 20),
                use_cache=True):
    """
    Dock a SMILES string against a given receptor at a given pocket center.
    Results are cached on disk (data/docking_cache.json) keyed by receptor +
    SMILES, so re-docking the same molecule/target pair is instant. The pose
    file itself is namespaced by receptor+name (see get_pose_path) so it is
    never silently overwritten by a different target/molecule pairing.
    """
    if receptor_path is None:
        receptor_path = DEFAULT_RECEPTOR_PATH
    if box_center is None:
        box_center = DEFAULT_BOX_CENTER

    smiles = largest_fragment(smiles)
    out_path = get_pose_path(name, receptor_path)

    cache = _load_dock_cache() if use_cache else {}
    cache_key = _dock_cache_key(smiles, receptor_path)
    if use_cache and cache_key in cache and os.path.exists(out_path):
        return cache[cache_key]

    smi_path = os.path.join(DATA_DIR, "structures", f"{name}.smi")
    pdbqt_path = os.path.join(DATA_DIR, "structures", f"{name}.pdbqt")

    with open(smi_path, "w") as f:
        f.write(smiles + "\n")
    subprocess.run(["obabel", smi_path, "-O", pdbqt_path, "--gen3d"],
                    check=True, capture_output=True)

    cx, cy, cz = box_center
    sx, sy, sz = box_size
    result = subprocess.run([
        VINA_PATH, "--receptor", receptor_path, "--ligand", pdbqt_path,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--out", out_path
    ], capture_output=True, text=True)

    score = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("1"):
            score = float(line.split()[1])
            break

    if use_cache and score is not None:
        cache[cache_key] = score
        _save_dock_cache(cache)

    return score


def run_pipeline(csv_path, top_n):
    df = pd.read_csv(csv_path)
    df["fingerprint"] = df["smiles"].apply(smiles_to_fp)
    df = df.dropna(subset=["fingerprint"])
    X = np.stack(df["fingerprint"].values)
    df["predicted_prob_active"] = model.predict_proba(X)[:, 1]
    top = df.sort_values("predicted_prob_active", ascending=False).head(top_n)

    print(f"Screened {len(df)} candidates. Docking top {top_n}...\n")
    results = []
    for _, row in top.iterrows():
        safe_name = "".join(c if c.isalnum() else "_" for c in str(row["name"]))[:30]
        score = dock_smiles(row["smiles"], safe_name)
        results.append((row["name"], row["predicted_prob_active"], score))
        print(f"{row['name']:30s}  screening={row['predicted_prob_active']:.3f}  docking={score}")

    return results


if __name__ == "__main__":
    csv_path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_pipeline(csv_path, top_n)
