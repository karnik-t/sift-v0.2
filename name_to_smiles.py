import sys
import requests
from py2opsin import py2opsin

def name_to_smiles(name):
    # Try PubChem first -- best for common/trade names (e.g. "aspirin")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/ConnectivitySMILES/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data["PropertyTable"]["Properties"][0]["ConnectivitySMILES"], "PubChem"
    except requests.exceptions.RequestException:
        pass

    # Fall back to OPSIN -- handles valid systematic IUPAC names PubChem doesn't index
    smiles = py2opsin(name)
    if smiles:
        return smiles, "OPSIN"

    return None, None

if __name__ == "__main__":
    name = sys.argv[1]
    smiles, source = name_to_smiles(name)
    if smiles is None:
        print(f"Could not resolve '{name}' via PubChem or OPSIN.")
    else:
        print(f"{smiles}  (source: {source})")
