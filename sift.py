import sys
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from name_to_smiles import name_to_smiles
from pipeline import dock_smiles

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
model = joblib.load("egfr_model.pkl")

def run(name):
    smiles, source = name_to_smiles(name)
    if smiles is None:
        print(f"Could not resolve '{name}' to a structure.")
        return

    print(f"Resolved '{name}' -> {smiles}  (via {source})")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("RDKit could not parse the resulting SMILES.")
        return

    fp = np.array(fp_gen.GetFingerprint(mol)).reshape(1, -1)
    prob = model.predict_proba(fp)[0][1]
    print(f"Screening: predicted probability of EGFR activity = {prob:.3f}")

    safe_name = "".join(c if c.isalnum() else "_" for c in name)[:30]
    score = dock_smiles(smiles, safe_name)
    print(f"Docking: predicted binding affinity = {score} kcal/mol")

if __name__ == "__main__":
    run(sys.argv[1])
