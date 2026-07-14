import sys
import time
import requests
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def find_target_id(target_name):
    r = requests.get(
        "https://www.ebi.ac.uk/chembl/api/data/target/search.json",
        params={"q": target_name, "organism": "Homo sapiens"}
    )
    targets = r.json()["targets"]
    if not targets:
        return None
    for t in targets:
        if t["target_type"] == "SINGLE PROTEIN":
            return t["target_chembl_id"], t["pref_name"]
    return targets[0]["target_chembl_id"], targets[0]["pref_name"]

def fetch_bioactivity(target_chembl_id, max_records=5000):
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    all_activities = []
    offset = 0
    limit = 1000
    while offset < max_records:
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": "IC50",
            "limit": limit,
            "offset": offset
        }
        r = requests.get(url, params=params)
        batch = r.json()["activities"]
        if not batch:
            break
        all_activities.extend(batch)
        offset += limit
        time.sleep(0.5)
    return all_activities

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(fp_gen.GetFingerprint(mol))

def train_target(target_name):
    print(f"Looking up ChEMBL target ID for '{target_name}'...")
    result = find_target_id(target_name)
    if result is None:
        print(f"No ChEMBL target found for '{target_name}'.")
        return
    target_id, pref_name = result
    print(f"Found: {target_id} ({pref_name})")

    print("Fetching bioactivity data...")
    activities = fetch_bioactivity(target_id)
    print(f"Fetched {len(activities)} records.")

    df = pd.DataFrame(activities)
    df = df[["canonical_smiles", "standard_value", "standard_units"]]
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df = df[df["standard_units"] == "nM"]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])

    # Collapse each molecule to its MEDIAN IC50 across all records, instead of
    # treating every individual assay record as a separate training example.
    # This resolves conflicting measurements (different assays/variants) rather
    # than feeding the model contradictory labels for the same molecule.
    before = len(df)
    df = df.groupby("canonical_smiles", as_index=False)["standard_value"].median()
    print(f"Collapsed {before} raw records into {len(df)} molecules (median IC50 per molecule).")

    df["active"] = (df["standard_value"] <= 1000).astype(int)

    print("Computing fingerprints...")
    df["fingerprint"] = df["canonical_smiles"].apply(smiles_to_fp)
    df = df.dropna(subset=["fingerprint"])
    print(f"Final training set: {len(df)} molecules "
          f"({df['active'].sum()} active / {(df['active']==0).sum()} inactive)")

    X = np.stack(df["fingerprint"].values)
    y = df["active"].values

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"Cross-validated ROC-AUC: {scores.mean():.3f}")

    clf.fit(X, y)
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())
    model_path = f"{safe_name}_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_target(sys.argv[1])
