import sys
import os
import subprocess
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
STRUCTURES_DIR = os.path.join(DATA_DIR, "structures")


def _structure_meta_path(safe_name):
    return os.path.join(STRUCTURES_DIR, f"{safe_name}_structure_meta.json")


def write_structure_meta(safe_name, meta):
    """
    Record where a target's structure came from (real PDB crystal structure
    vs. an AlphaFold prediction) and any relevant confidence info, so the UI
    can show an honest "how much to trust this" signal instead of treating
    every prepared structure the same. Overwrites any previous meta file for
    this target -- callers should call this exactly once per successful
    structure preparation, after the receptor/pocket files are written.
    """
    import json
    with open(_structure_meta_path(safe_name), "w") as f:
        json.dump(meta, f, indent=2)


def read_structure_meta(target_name):
    """
    Returns the metadata dict for a target's prepared structure, or a
    generic RCSB-style fallback dict if no meta file exists (covers
    structures prepared before this metadata system existed).
    """
    import json
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())
    meta_path = _structure_meta_path(safe_name)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"source": "rcsb", "note": "prepared before structure-metadata tracking existed"}


def prepare_structure(target_name, pdb_id, ligand_code):
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())

    pdb_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        print(f"Using already-downloaded {pdb_id}.pdb")
    else:
        print(f"Downloading {pdb_id}...")
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(r.text)

    receptor_pdb = os.path.join(STRUCTURES_DIR, f"{safe_name}_receptor.pdb")
    ligand_pdb = os.path.join(STRUCTURES_DIR, f"{safe_name}_ligand.pdb")

    with open(pdb_path) as f, open(receptor_pdb, "w") as rec, open(ligand_pdb, "w") as lig:
        for line in f:
            if line.startswith("ATOM"):
                rec.write(line)
            elif line.startswith("HETATM") and ligand_code in line:
                lig.write(line)

    receptor_pdbqt = os.path.join(STRUCTURES_DIR, f"{safe_name}_receptor.pdbqt")
    ligand_pdbqt = os.path.join(STRUCTURES_DIR, f"{safe_name}_ligand.pdbqt")

    subprocess.run(["obabel", receptor_pdb, "-O", receptor_pdbqt, "-xr"], check=True)
    subprocess.run(["obabel", ligand_pdb, "-O", ligand_pdbqt], check=True)

    coords = []
    with open(ligand_pdb) as f:
        for line in f:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)

    print(f"Receptor: {receptor_pdbqt}")
    print(f"Pocket center: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}")

    pocket_path = os.path.join(STRUCTURES_DIR, f"{safe_name}_pocket.txt")
    with open(pocket_path, "w") as f:
        f.write(f"{cx:.3f},{cy:.3f},{cz:.3f}\n")
    print(f"Saved pocket coordinates to {pocket_path}")

    write_structure_meta(safe_name, {
        "source": "rcsb",
        "pdb_id": pdb_id,
        "ligand_code": ligand_code,
    })


if __name__ == "__main__":
    target_name = sys.argv[1]
    pdb_id = sys.argv[2]
    ligand_code = sys.argv[3]
    prepare_structure(target_name, pdb_id, ligand_code)


# --- Automatic structure discovery via RCSB search ---

# Common crystallization additives, ions, buffer components, and cryoprotectants
# that show up as HETATM records but are not meaningful drug-like ligands.
_LIGAND_BLOCKLIST = {
    "HOH", "GOL", "EDO", "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "MN", "K",
    "ACT", "TRS", "PEG", "DMS", "BME", "MPD", "IMD", "FMT", "NO3", "UNX", "1PE",
    "P6G", "PGE", "MES", "BOG", "CIT", "NI", "CO", "FE", "CU", "BR", "IOD",
    "SCN", "NH4", "CD", "HG", "PB", "IPA", "EOH", "MOH", "ACY", "ACE", "GSH",
    "NAG", "MAN", "BMA", "FUC", "GAL", "SIN", "TLA", "PG4", "PGO", "CAC", "FLC",
}


def find_ligand_for_target(target_name, max_entries_to_check=8, status=print):
    """
    Search RCSB for a PDB structure of target_name that has a real bound
    ligand (not just a water/ion/buffer component). Uses only two endpoints:
    the search API (to find candidate PDB IDs) and direct .pdb file downloads
    (to inspect HETATM records), avoiding the Data API's per-entity metadata
    endpoint, whose entity-ID format and field names are easy to get wrong.

    Returns (pdb_id, ligand_code) or (None, None) if nothing suitable is found.

    status: a callable (e.g. print, or a Streamlit UI callback) used to
            report progress as this runs, since it can take a while.
    """
    status(f"Searching RCSB for structures of '{target_name}'...")
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    search_body = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": target_name},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_entries_to_check},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }

    try:
        resp = requests.post(search_url, json=search_body, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("result_set", [])
    except Exception as e:
        status(f"RCSB search request failed: {e}")
        return None, None

    if not results:
        status(f"No PDB entries found for '{target_name}'.")
        return None, None

    status(f"Found {len(results)} candidate PDB entries. Checking each for a usable bound ligand...")
    os.makedirs(STRUCTURES_DIR, exist_ok=True)

    for hit in results:
        pdb_id = hit["identifier"]
        status(f"Checking {pdb_id}...")
        try:
            pdb_resp = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
            pdb_resp.raise_for_status()
            pdb_text = pdb_resp.text
        except Exception:
            continue

        residue_counts = {}
        for line in pdb_text.splitlines():
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname and resname != "HOH":
                    residue_counts[resname] = residue_counts.get(resname, 0) + 1

        candidates = {
            name: count for name, count in residue_counts.items()
            if name not in _LIGAND_BLOCKLIST and count >= 8
        }

        if candidates:
            best_ligand = max(candidates, key=candidates.get)
            status(f"Found candidate structure: {pdb_id}, ligand: {best_ligand} ({candidates[best_ligand]} atoms)")
            cached_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.pdb")
            with open(cached_path, "w") as f:
                f.write(pdb_text)
            return pdb_id, best_ligand

    status(f"Checked {len(results)} PDB entries for '{target_name}' but found no suitable bound ligand "
           f"(only waters, ions, or buffer components in each).")
    return None, None


def find_uniprot_id(target_name):
    """
    Look up the UniProt accession for a human gene/protein name. Returns
    the accession string (e.g. 'P50750') or None if nothing reviewed matches.
    """
    try:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"gene:{target_name} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "fields": "accession,id,protein_name",
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return None

    if not results:
        return None
    return results[0]["primaryAccession"]


def get_alphafold_structure(uniprot_id):
    """
    Fetch AlphaFold DB's prediction metadata for a UniProt accession and
    return (pdb_url, mean_plddt) using the prediction API's own URL field
    (never construct the filename/version ourselves -- AlphaFold DB
    version numbers change over time and a hardcoded version will break).
    Returns (None, None) if no prediction exists for this accession.
    """
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=30)
        r.raise_for_status()
        entries = r.json()
    except Exception:
        return None, None

    if not entries:
        return None, None

    entry = entries[0]
    return entry.get("pdbUrl"), entry.get("globalMetricValue")


def prepare_alphafold_structure(target_name, status=print):
    """
    Fallback structure source for targets with no usable RCSB entry:
    UniProt lookup -> AlphaFold DB predicted structure -> receptor PDBQT.

    IMPORTANT: this prepares the RECEPTOR ONLY. AlphaFold structures are
    predictions with no bound ligand, so there is no pocket centroid to
    compute the way prepare_structure() does from a real co-crystallized
    ligand. No _pocket.txt is written here -- pocket detection on this
    apo structure needs a dedicated tool (fpocket), not yet wired in.
    Docking against this target will stay unavailable until that pocket
    file exists, which is intentional: no honest way to guess a docking
    box center without either a bound ligand or real pocket detection.

    Returns True if the receptor file was prepared, False otherwise.
    """
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())

    status(f"No RCSB structure found for '{target_name}' -- trying AlphaFold DB...")
    uniprot_id = find_uniprot_id(target_name)
    if uniprot_id is None:
        status(f"No UniProt entry found for '{target_name}'.")
        return False

    pdb_url, mean_plddt = get_alphafold_structure(uniprot_id)
    if pdb_url is None:
        status(f"No AlphaFold DB prediction found for UniProt {uniprot_id}.")
        return False

    status(f"Found AlphaFold prediction for {uniprot_id} (mean pLDDT: {mean_plddt}). Downloading...")
    try:
        r = requests.get(pdb_url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        status(f"AlphaFold download failed: {e}")
        return False

    receptor_pdb = os.path.join(STRUCTURES_DIR, f"{safe_name}_alphafold.pdb")
    with open(receptor_pdb, "w") as f:
        f.write(r.text)

    receptor_pdbqt = os.path.join(STRUCTURES_DIR, f"{safe_name}_receptor.pdbqt")
    try:
        subprocess.run(["obabel", receptor_pdb, "-O", receptor_pdbqt, "-xr"], check=True)
    except subprocess.CalledProcessError as e:
        status(f"obabel conversion failed: {e}")
        return False

    status(f"AlphaFold receptor ready: {receptor_pdbqt} (mean pLDDT: {mean_plddt}, "
           f"predicted structure, not a real crystal structure).")

    pocket_found = detect_pocket_with_fpocket(receptor_pdb, safe_name, status=status)
    if not pocket_found:
        status(f"No pocket detected for '{target_name}' -- docking will remain "
               f"unavailable until fpocket is available or finds a usable pocket.")

    write_structure_meta(safe_name, {
        "source": "alphafold",
        "uniprot_id": uniprot_id,
        "mean_plddt": mean_plddt,
        "pocket_detected": pocket_found,
    })

    return True


def run_fpocket(pdb_path, status=print):
    """
    Run fpocket against a structure and return the path to its output
    directory (<name>_out/), or None if fpocket isn't available or failed.
    Requires the 'fpocket-env' conda environment's fpocket binary --
    resolved via shutil.which so this works regardless of which
    environment the calling Python process is running in.
    """
    import shutil
    import subprocess as sp

    fpocket_bin = shutil.which("fpocket")
    if fpocket_bin is None:
        status("fpocket not found on PATH -- activate the 'fpocket-env' conda "
               "environment, or run fpocket separately and point at its output.")
        return None

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    out_dir = os.path.join(os.path.dirname(pdb_path), f"{base}_out")

    if os.path.isdir(out_dir):
        status(f"Using existing fpocket output: {out_dir}")
        return out_dir

    status(f"Running fpocket on {os.path.basename(pdb_path)}...")
    try:
        sp.run([fpocket_bin, "-f", pdb_path], check=True, capture_output=True)
    except sp.CalledProcessError as e:
        status(f"fpocket failed: {e}")
        return None

    return out_dir if os.path.isdir(out_dir) else None


def parse_fpocket_pockets(fpocket_out_dir):
    """
    Parse fpocket's <name>_info.txt into a list of dicts, one per pocket:
    {"number": int, "score": float, "druggability": float}, in the order
    fpocket reports them (already sorted best-first by fpocket itself).
    """
    info_files = [f for f in os.listdir(fpocket_out_dir) if f.endswith("_info.txt")]
    if not info_files:
        return []

    info_path = os.path.join(fpocket_out_dir, info_files[0])
    pockets = []
    current = None

    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pocket "):
                if current:
                    pockets.append(current)
                current = {"number": int(line.split()[1])}
            elif current is not None:
                if line.startswith("Score :"):
                    current["score"] = float(line.split(":")[1].strip())
                elif line.startswith("Druggability Score :"):
                    current["druggability"] = float(line.split(":")[1].strip())

    if current:
        pockets.append(current)
    return pockets


def compute_pocket_centroid(fpocket_out_dir, pocket_number):
    """Read a specific pocket's atom file and return its (x, y, z) centroid."""
    atom_path = os.path.join(fpocket_out_dir, "pockets", f"pocket{pocket_number}_atm.pdb")
    if not os.path.exists(atom_path):
        return None

    coords = []
    with open(atom_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue

    if not coords:
        return None
    n = len(coords)
    return (sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n, sum(c[2] for c in coords) / n)


def detect_pocket_with_fpocket(pdb_path, safe_name, status=print):
    """
    Run fpocket on an apo (no-ligand) structure, pick the best pocket by
    druggability score (falling back to fpocket's general Score if every
    druggability score is 0, which happens on low-confidence/non-globular
    predictions), and write a _pocket.txt in the same format prepare_structure()
    produces from a real bound ligand. Returns True if a pocket file was written.
    """
    out_dir = run_fpocket(pdb_path, status=status)
    if out_dir is None:
        return False

    pockets = parse_fpocket_pockets(out_dir)
    if not pockets:
        status("fpocket ran but found no pockets meeting its detection thresholds.")
        return False

    # Prefer the pocket with the best druggability score; if every pocket
    # scored 0 on druggability (common on low-confidence apo structures),
    # fall back to fpocket's general (non-druggability) Score instead.
    best_by_drug = max(pockets, key=lambda p: p.get("druggability", 0))
    if best_by_drug.get("druggability", 0) > 0:
        chosen = best_by_drug
        rank_note = f"druggability score {chosen['druggability']:.3f}"
    else:
        chosen = max(pockets, key=lambda p: p.get("score", 0))
        rank_note = f"general score {chosen['score']:.3f} (all druggability scores were 0)"

    centroid = compute_pocket_centroid(out_dir, chosen["number"])
    if centroid is None:
        status(f"Could not compute a centroid for pocket {chosen['number']}.")
        return False

    cx, cy, cz = centroid
    pocket_path = os.path.join(STRUCTURES_DIR, f"{safe_name}_pocket.txt")
    with open(pocket_path, "w") as f:
        f.write(f"{cx:.3f},{cy:.3f},{cz:.3f}\n")

    status(f"Pocket detected via fpocket: pocket {chosen['number']} ({rank_note}), "
           f"out of {len(pockets)} candidate pockets. Center: x={cx:.3f}, y={cy:.3f}, z={cz:.3f}. "
           f"Saved to {pocket_path}")
    return True


def ensure_structure(target_name, status=print):
    """
    Automatically find and prepare a docking structure for target_name if
    one doesn't already exist. Returns True if a structure is now ready
    (either already existed or was freshly prepared), False otherwise.

    status: a callable used to report progress (see find_ligand_for_target).
    """
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())
    receptor_pdbqt = os.path.join(STRUCTURES_DIR, f"{safe_name}_receptor.pdbqt")
    pocket_file = os.path.join(STRUCTURES_DIR, f"{safe_name}_pocket.txt")

    if os.path.exists(receptor_pdbqt) and os.path.exists(pocket_file):
        return True

    status(f"No prepared structure for '{target_name}' yet -- searching RCSB for one...")
    pdb_id, ligand_code = find_ligand_for_target(target_name, status=status)
    if pdb_id is not None:
        try:
            status(f"Downloading and preparing {pdb_id} (ligand {ligand_code})...")
            prepare_structure(target_name, pdb_id, ligand_code)
            status(f"Structure ready: {pdb_id} / {ligand_code}")
            return True
        except Exception as e:
            status(f"Auto-preparation failed for '{target_name}' using {pdb_id}/{ligand_code}: {e}")
            # Fall through to AlphaFold rather than giving up entirely.

    # No usable RCSB structure (or preparation failed) -- try AlphaFold DB.
    # prepare_alphafold_structure() now also attempts fpocket-based pocket
    # detection internally, so a real pocket file may exist afterward --
    # check for it rather than assuming failure.
    prepare_alphafold_structure(target_name, status=status)
    return os.path.exists(receptor_pdbqt) and os.path.exists(pocket_file)
