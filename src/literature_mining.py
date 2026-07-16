"""
Literature mining for Sift: pull PubMed abstracts for a target, then use
Gemini to extract known ligands (drugs vs endogenous) and disease context.
Results are cached to data/literature/{target}_literature.json.

Optimizations:
- All abstracts for a target are sent to Gemini in a single batched request
  (instead of one request per abstract), to conserve free-tier daily quota.
- Gemini calls are wrapped with a small manual retry/backoff for transient
  network or server errors (503s, dropped connections), on top of the
  google-genai client's own internal retries.
"""
import os
import json
import time
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


def _generate_with_retry(client, model, prompt, max_attempts=3, base_delay=5):
    """
    Call Gemini with a small manual retry/backoff, on top of whatever
    retries the google-genai client already does internally. Handles
    transient issues (503 high demand, dropped connections) that the
    free tier is prone to.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                delay = base_delay * attempt
                print(f"Gemini call failed (attempt {attempt}/{max_attempts}): {e}")
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


# Tried in order. Flash-Lite variants have a much larger free-tier daily
# quota than the preview model we were on. A model-not-found error (the
# model doesn't exist / isn't available yet) is instant and costs no
# quota, so falling through the list is safe. A quota (429) or transient
# (503) error on a real model is NOT a reason to fall through -- that
# model is fine, we're just rate-limited on it right now.
MODEL_CANDIDATES = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]


def _is_model_missing_error(e):
    msg = str(e)
    return "NOT_FOUND" in msg or "not found" in msg.lower() or "no longer available" in msg.lower()


def _generate_with_model_fallback(client, prompt, max_attempts=3, base_delay=5):
    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            # Probe with a single attempt first (no retry delay) so a
            # permanently-dead model name fails in ~1 call, not after
            # 3 retries with growing backoff. Retries only make sense
            # for transient errors (503, dropped connections), and a
            # 404 model-not-found will never succeed on retry anyway.
            return _generate_with_retry(client, model, prompt, max_attempts=1)
        except Exception as e:
            if _is_model_missing_error(e):
                print(f"Model '{model}' unavailable, trying next candidate...")
                last_error = e
                continue
            # Real (likely transient) error on a model that DOES exist --
            # now it's worth the full retry/backoff treatment.
            try:
                return _generate_with_retry(client, model, prompt, max_attempts=max_attempts, base_delay=base_delay)
            except Exception as e2:
                last_error = e2
                if _is_model_missing_error(e2):
                    continue
                raise
    raise last_error


def extract_batch(client, target_name, abstracts):
    """
    Send ALL abstracts for a target to Gemini in a single request, asking
    for a JSON array of results (one per abstract, in order). This is the
    main quota-saving optimization: mining N abstracts costs 1 API call
    instead of N.

    Returns a list of dicts, one per abstract:
        {"ligands": [{"name": ..., "type": "drug"|"endogenous"}], "context": "..."}
    """
    if not abstracts:
        return []

    numbered = "\n\n".join(
        f"Abstract {i + 1}:\n\"\"\"{a['abstract']}\"\"\""
        for i, a in enumerate(abstracts)
    )

    prompt = (
        "You are extracting structured data from multiple scientific abstracts, "
        f'all about the protein target "{target_name}".\n\n'
        f"{numbered}\n\n"
        "Return ONLY valid JSON, no markdown fences, no preamble: a JSON array "
        f"with EXACTLY {len(abstracts)} objects, one per abstract above, in the "
        "same order. Each object must look exactly like this:\n"
        '{"ligands": [{"name": "...", "type": "drug"}], "context": "one sentence"}\n\n'
        "Rules:\n"
        '- Each ligand needs a "type" of either "drug" (a synthetic or therapeutic '
        'compound/inhibitor/antibody used or developed as medicine) or "endogenous" '
        "(the target's natural biological binding partner, e.g. a growth factor or hormone).\n"
        '- Do not include the target protein itself as a ligand.\n'
        '- If an abstract names no specific ligands, use an empty list for that abstract.\n'
        '- Keep each "context" to one concise sentence.\n'
        f"- The output array must have exactly {len(abstracts)} elements, matching "
        "the abstracts in order. Do not skip or merge any.\n"
    )

    response = _generate_with_model_fallback(client, prompt)

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        results = []

    # Defensive: pad or trim to match the number of abstracts, so downstream
    # code can always zip abstracts 1:1 with results.
    if len(results) < len(abstracts):
        results += [{"ligands": [], "context": ""}] * (len(abstracts) - len(results))
    elif len(results) > len(abstracts):
        results = results[:len(abstracts)]

    return results


def mine_literature(target_name, max_abstracts=10, ncbi_api_key=None):
    """
    Full literature mining pipeline for a target: fetch abstracts, extract
    ligands + context for all of them in a single batched Gemini call,
    aggregate into one result.
    """
    client = get_gemini_client()
    abstracts = fetch_pubmed_abstracts(target_name, max_results=max_abstracts, api_key=ncbi_api_key)

    extracted_list = extract_batch(client, target_name, abstracts)

    drugs = set()
    endogenous = set()
    contexts = []
    per_abstract_results = []

    for a, extracted in zip(abstracts, extracted_list):
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
