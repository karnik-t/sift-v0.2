import sys
import os
import time
import requests
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "data", "models")

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _target_synonyms(t):
    """Flatten every synonym across all components of a target (gene
    symbols like 'CDK9' usually live here, not in pref_name -- especially
    for targets ChEMBL models as a family or complex)."""
    syns = []
    for comp in t.get("target_components", []):
        for s in comp.get("target_component_synonyms", []):
            val = s.get("component_synonym")
            if val:
                syns.append(val)
    return syns


def _normalize(s):
    return s.lower().replace("-", "").replace(" ", "").strip()


def _exact_match(query, candidate_name, synonyms):
    """Strict: the WHOLE candidate name or WHOLE synonym string must equal
    the query exactly (after normalizing case/spacing/hyphens). This is
    what correctly picks 'CDK9' as a synonym on the real CDK9 complexes,
    while rejecting a longer descriptive synonym like 'Major CDK9
    elongation factor-associated protein' (a different protein, AFF4,
    that merely mentions CDK9) -- that phrase is not equal to 'CDK9', it
    just contains it."""
    query_norm = _normalize(query)
    for candidate in [candidate_name] + synonyms:
        if _normalize(candidate) == query_norm:
            return True
    return False


def _loose_match(query, candidate_name):
    """Fallback only: shares a whole word with the query. Deliberately
    weaker and only used if NO exact match exists anywhere in the results,
    since substring/token checks are what let 'CDK9' match inside an
    unrelated protein's descriptive synonym last time."""
    query_tokens = set(query.lower().replace("-", " ").split())
    name_tokens = set(candidate_name.lower().replace("-", " ").split())
    return bool(query_tokens & name_tokens)


def find_target_id(target_name):
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/target/search.json",
            params={"q": target_name, "organism": "Homo sapiens"},
            timeout=30,
        )
        r.raise_for_status()
        targets = r.json()["targets"]
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        # Covers connection errors, timeouts, non-2xx responses, and bodies
        # that aren't valid JSON (or valid JSON missing "targets") -- any of
        # which previously crashed the whole page with an unhandled
        # JSONDecodeError instead of failing this one target cleanly.
        print(f"ChEMBL target search failed for '{target_name}': {e}")
        return None

    if not targets:
        return None

    exact = [t for t in targets if _exact_match(target_name, t["pref_name"], _target_synonyms(t))]

    if exact:
        single_proteins = [t for t in exact if t["target_type"] == "SINGLE PROTEIN"]
        chosen = single_proteins[0] if single_proteins else exact[0]
        if not single_proteins:
            print(f"Note: '{target_name}' matched as part of a {chosen['target_type'].lower()} "
                  f"('{chosen['pref_name']}'), not a standalone single-protein entry -- "
                  f"bioactivity data may reflect the complex, not the isolated protein.")
        return chosen["target_chembl_id"], chosen["pref_name"]

    # No exact match anywhere -- fall back to loose matching, but treat it
    # as much less trustworthy and say so.
    loose = [t for t in targets if _loose_match(target_name, t["pref_name"])]
    if loose:
        chosen = loose[0]
        print(f"Note: no exact ChEMBL match for '{target_name}' -- falling back to a loose "
              f"name match: '{chosen['pref_name']}' ({chosen['target_chembl_id']}). "
              f"Verify this is the intended target.")
        return chosen["target_chembl_id"], chosen["pref_name"]

    best_guess = targets[0]
    print(f"WARNING: no ChEMBL target name or synonym closely matching '{target_name}' was "
          f"found. Closest available result was '{best_guess['pref_name']}' "
          f"({best_guess['target_chembl_id']}), which does not clearly correspond to your "
          f"query. Refusing to auto-train against a possibly-wrong target -- please verify "
          f"the target name or search ChEMBL manually.")
    return None

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
        raise ValueError(
            f"No ChEMBL bioactivity target found for '{target_name}'. This can mean the "
            f"target has no name/synonym match in ChEMBL, or that ChEMBL's search API "
            f"returned an error for this query. Try a different name, or check ChEMBL "
            f"directly at https://www.ebi.ac.uk/chembl/"
        )
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

    n_active = int((y == 1).sum())
    n_inactive = int((y == 0).sum())
    min_class_count = min(n_active, n_inactive)

    if min_class_count < 5 or len(y) < 20:
        print(f"WARNING: only {len(y)} molecules ({n_active} active / {n_inactive} inactive) -- "
              f"too few for reliable cross-validation. Training without a validation score. "
              f"Treat this model's predictions with extra caution.")
    else:
        cv_folds = min(5, min_class_count)
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")
        print(f"Cross-validated ROC-AUC: {scores.mean():.3f}" + (f" (cv={cv_folds})" if cv_folds != 5 else ""))

    clf.fit(X, y)
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_target(sys.argv[1])
