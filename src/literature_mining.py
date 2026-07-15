"""
Literature mining for Sift: pull PubMed abstracts for a target, then use
Gemini to extract known ligands (drugs vs endogenous) and disease context.
Results are cached to data/literature/{target}_literature.json.
"""
import os
import json
import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_pubmed_abstracts(target_name, max_results=20, api_key=None):
    """
    Search PubMed for abstracts mentioning target_name, and fetch their text.
    Returns a list of dicts: [{"pmid": ..., "title": ..., "abstract": ...}, ...]
    """
    search_params = {
        "db": "pubmed",
        "term": f"{target_name} AND (inhibitor OR ligand OR bioactivity)",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        search_params["api_key"] = api_key

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=search_params, timeout=30)
    resp.raise_for_status()
    pmids = resp.json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params, timeout=30)
    resp.raise_for_status()

    return parse_pubmed_xml(resp.text)


def parse_pubmed_xml(xml_text):
    """Parse efetch XML into a list of {pmid, title, abstract} dicts."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")

        pmid = pmid_el.text if pmid_el is not None else None
        title = title_el.text if title_el is not None else ""
        abstract = " ".join(a.text for a in abstract_parts if a.text)

        if pmid and abstract:
            articles.append({"pmid": pmid, "title": title, "abstract": abstract})

    return articles


# --- Gemini extraction layer ---

def get_gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def extract_ligands_from_abstract(client, target_name, abstract_text):
    """
    Ask Gemini to extract known ligands from a single PubMed abstract,
    tagging each as a therapeutic drug/inhibitor or a natural (endogenous)
    ligand, plus a one-sentence disease/context blurb.
    """
    prompt = (
        "You are extracting structured data from a scientific abstract about "
        f'the protein target "{target_name}".\n\n'
        f'Abstract:\n"""{abstract_text}"""\n\n'
        "Return ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:\n"
        '{"ligands": [{"name": "...", "type": "drug"}], "context": "one sentence"}\n\n'
        "Rules:\n"
        '- Each ligand needs a "type" of either "drug" (a synthetic or therapeutic '
        'compound/inhibitor/antibody used or developed as medicine) or "endogenous" '
        "(the target's natural biological binding partner, e.g. a growth factor or hormone).\n"
        '- Do not include the target protein itself as a ligand.\n'
        '- If no specific named ligands are mentioned, return an empty list.\n'
        '- Keep "context" to one concise sentence.\n'
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ligands": [], "context": "", "_raw_error": text}


def mine_literature(target_name, max_abstracts=10, ncbi_api_key=None):
    """
    Full literature mining pipeline for a target: fetch abstracts, extract
    ligands + context from each via Gemini, aggregate into one result.
    """
    client = get_gemini_client()
    abstracts = fetch_pubmed_abstracts(target_name, max_results=max_abstracts, api_key=ncbi_api_key)

    drugs = set()
    endogenous = set()
    contexts = []
    per_abstract_results = []

    for a in abstracts:
        extracted = extract_ligands_from_abstract(client, target_name, a["abstract"])
        per_abstract_results.append({
            "pmid": a["pmid"],
            "title": a["title"],
            **extracted,
        })
        for lig in extracted.get("ligands", []):
            name = lig.get("name", "").strip()
            if not name:
                continue
            if lig.get("type") == "endogenous":
                endogenous.add(name)
            else:
                drugs.add(name)
        if extracted.get("context"):
            contexts.append(extracted["context"])

    return {
        "target": target_name,
        "num_abstracts": len(abstracts),
        "drugs": sorted(drugs),
        "endogenous_ligands": sorted(endogenous),
        "contexts": contexts,
        "per_abstract": per_abstract_results,
    }


# --- Caching layer ---

def _literature_path(target_name):
    from sift_target import safe, DATA_DIR
    lit_dir = os.path.join(DATA_DIR, "literature")
    os.makedirs(lit_dir, exist_ok=True)
    return os.path.join(lit_dir, f"{safe(target_name)}_literature.json")


def get_literature(target_name, max_abstracts=10, ncbi_api_key=None, force_refresh=False):
    """
    Returns cached literature mining results for a target if present,
    otherwise runs mine_literature() and caches the result to disk.
    """
    path = _literature_path(target_name)
    if not force_refresh and os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    result = mine_literature(target_name, max_abstracts=max_abstracts, ncbi_api_key=ncbi_api_key)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "EGFR"
    result = get_literature(target, max_abstracts=5)

    print("=" * 50)
    print(f"  {result['target']}  ---  {result['num_abstracts']} abstracts mined")
    print("=" * 50)

    print("\nKnown drugs / inhibitors:")
    if result["drugs"]:
        for d in result["drugs"]:
            print(f"  - {d}")
    else:
        print("  (none found)")

    print("\nNatural (endogenous) ligands:")
    if result["endogenous_ligands"]:
        for e in result["endogenous_ligands"]:
            print(f"  - {e}")
    else:
        print("  (none found)")

    print("\nContext summary:")
    for c in result["contexts"]:
        print(f"  - {c}")
    print()
