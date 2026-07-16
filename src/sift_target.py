import sys
import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from train_target import train_target
from pipeline import dock_smiles, read_pocket
from name_to_smiles import name_to_smiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def safe(target_name):
    return "".join(c if c.isalnum() else "_" for c in target_name.lower())


def ensure_model(target_name):
    model_path = os.path.join(DATA_DIR, "models", f"{safe(target_name)}_model.pkl")
    if not os.path.exists(model_path):
        print(f"No existing model for '{target_name}' -- training now...")
        train_target(target_name)
    return joblib.load(model_path)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(fp_gen.GetFingerprint(mol))


def get_target_structure(target_name, status=print):
    """
    Returns (receptor_path, box_center) for a target. If no structure is
    prepared yet, automatically tries to find and prepare one via RCSB
    search (see prepare_target_structure.ensure_structure) before giving
    up and returning (None, None).

    status: a callable used to report auto-preparation progress (defaults
            to print for CLI use; the Streamlit UI passes a status box's
            .write method instead so progress shows up live on screen).
    """
    receptor_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_receptor.pdbqt")
    pocket_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_pocket.txt")

    if not (os.path.exists(receptor_path) and os.path.exists(pocket_path)):
        from prepare_target_structure import ensure_structure
        ensure_structure(target_name, status=status)

    if os.path.exists(receptor_path) and os.path.exists(pocket_path):
        return receptor_path, read_pocket(pocket_path)
    return None, None


def screen_one(target_name, molecule_name):
    model = ensure_model(target_name)
    smiles, source = name_to_smiles(molecule_name)
    if smiles is None:
        print(f"Could not resolve '{molecule_name}' to a structure.")
        return
    print(f"Resolved '{molecule_name}' -> {smiles} (via {source})")

    fp = smiles_to_fp(smiles)
    if fp is None:
        print("RDKit could not parse the resulting SMILES.")
        return
    prob = model.predict_proba(fp.reshape(1, -1))[0][1]
    print(f"Screening vs {target_name}: predicted probability of activity = {prob:.3f}")

    receptor_path, box_center = get_target_structure(target_name)
    if receptor_path:
        score = dock_smiles(smiles, safe(molecule_name)[:30],
                             receptor_path=receptor_path, box_center=box_center)
        print(f"Docking vs {target_name}: predicted binding affinity = {score} kcal/mol")
    else:
        print(f"(No prepared structure for '{target_name}' yet -- run prepare_target_structure.py first for docking.)")


def top_candidates(target_name, csv_path, top_n=10):
    model = ensure_model(target_name)
    df = pd.read_csv(csv_path)
    df["fingerprint"] = df["smiles"].apply(smiles_to_fp)
    df = df.dropna(subset=["fingerprint"])
    X = np.stack(df["fingerprint"].values)
    df["predicted_prob_active"] = model.predict_proba(X)[:, 1]
    top = df.sort_values("predicted_prob_active", ascending=False).head(top_n)
    print(f"Top {top_n} candidates for {target_name}:")
    for _, row in top.iterrows():
        print(f"  {row['name']:30s}  screening={row['predicted_prob_active']:.3f}")


if __name__ == "__main__":
    target_name = sys.argv[1]
    candidates_path = os.path.join(DATA_DIR, "candidates.csv")
    if len(sys.argv) > 2:
        molecule_name = sys.argv[2]
        screen_one(target_name, molecule_name)
    else:
        top_candidates(target_name, candidates_path)
