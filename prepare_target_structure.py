import sys
import subprocess
import requests

def prepare_structure(target_name, pdb_id, ligand_code):
    safe_name = "".join(c if c.isalnum() else "_" for c in target_name.lower())

    print(f"Downloading {pdb_id}...")
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    pdb_path = f"{pdb_id}.pdb"
    with open(pdb_path, "w") as f:
        f.write(r.text)

    receptor_pdb = f"{safe_name}_receptor.pdb"
    ligand_pdb = f"{safe_name}_ligand.pdb"

    with open(pdb_path) as f, open(receptor_pdb, "w") as rec, open(ligand_pdb, "w") as lig:
        for line in f:
            if line.startswith("ATOM"):
                rec.write(line)
            elif line.startswith("HETATM") and ligand_code in line:
                lig.write(line)

    receptor_pdbqt = f"{safe_name}_receptor.pdbqt"
    ligand_pdbqt = f"{safe_name}_ligand.pdbqt"

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

    with open(f"{safe_name}_pocket.txt", "w") as f:
        f.write(f"{cx:.3f},{cy:.3f},{cz:.3f}\n")
    print(f"Saved pocket coordinates to {safe_name}_pocket.txt")

if __name__ == "__main__":
    target_name = sys.argv[1]
    pdb_id = sys.argv[2]
    ligand_code = sys.argv[3]
    prepare_structure(target_name, pdb_id, ligand_code)
