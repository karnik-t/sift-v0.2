## What this is

Sift is a personal project I'm building over summer 2026 to learn, hands-on, how AI and bioinformatics tools are actually used in early-stage drug discovery. I'm a biomedicine student, and going in I knew the theory around drug discovery only from lectures and so, this project is my way of testing the waters with the real, practical tools researchers use, using EGFR (a well-studied cancer drug target) as a concrete starting case study, entirely with free/public data and tools.

This isn't meant to be by any means a novel scientific contribution rather it's more a documented exploration of a real workflow, and a way to identify where my own interests and skill gaps are before committing to a research direction.

## What it does

Given a target protein name, Sift chains together three real drug-discovery steps:

1. **Bioactivity screening** — trains a random forest classifier on real ChEMBL bioactivity data for that target and ranks candidate molecules by predicted probability of activity
2. **Structural docking** — docks top-ranked candidates against the target's real binding pocket using AutoDock Vina, if a structure has been prepared for it
3. **Literature mining** — pulls recent PubMed abstracts for that target and uses an LLM (Gemini) to extract known drugs/inhibitors (kept separate from the target's natural endogenous ligands) plus a plain-language disease context summary

Two supporting pieces tie it together:
- A **SMILES translator** (`name_to_smiles.py`) that resolves a plain drug/molecule name (common or IUPAC) into its chemical structure via PubChem or OPSIN, used throughout the pipeline whenever a molecule needs to go from "name" to "structure"
- A **Streamlit UI** (`src/app.py`) that brings all of the above into one interface, including a bridge between the Sift and Literature tabs — a drug the literature step finds can be screened and docked inline with a single click, directly checking the model's predictions against real, literature-confirmed ligands

It started as an EGFR-only proof of concept but has since generalized to arbitrary targets — validated so far on EGFR, BRAF, JAK2, mTOR, and GLUT4 — as long as ChEMBL has bioactivity data for the target name given.

## Try it

```bash
streamlit run src/app.py
```

or from the command line:

```bash
python src/sift_target.py "TARGET_NAME"
python src/sift_target.py "TARGET_NAME" "molecule name"
python src/literature_mining.py "TARGET_NAME"
python src/name_to_smiles.py "molecule name"
```

See the Commands tab in the app for the full, current list of CLI entry points.

## Roadmap

- [x] v0.1 — Bioactivity screening model, validated against approved EGFR drugs
- [x] v0.2 — Structural docking validation (AutoDock Vina against EGFR's binding pocket)
- [x] v0.2.1 — Generalized screening + docking to arbitrary targets (validated on BRAF, JAK2, mTOR, GLUT4), plus a SMILES translator and repo reorganization
- [x] v0.2.2 — Streamlit UI (Sift, SMILES translator, Commands tabs; docking scores shown alongside screening scores)
- [x] v0.3 — LLM-based literature mining (PubMed context + known-ligand extraction, via Gemini)
- [x] v0.3.1 — Literature mining wired into the UI (Literature tab, Sift↔Literature bridge, disk caching)
- [ ] v1.0 — Full write-up, polish, and (optional) Flask UI rebuild

## What I learned / open questions (log)

**v0.1 — Bioactivity screening**
- Seeing the model rediscover real drugs (lapatinib, erlotinib) purely from fingerprint patterns made the textbook idea of "structure-activity relationship" click for the first time
- how much of the model's success is genuine chemistry versus it picking up on the kind of molecule ChEMBL happens to have lots of data for

**v0.2 — Structural docking**
- Docking made the screening model's predictions feel less abstract — going from "the model thinks this looks similar to known drugs" to "here's how it would physically sit in the actual binding pocket"
- Real-world data is messy: salt forms (e.g. lapatinib ditosylate) needed extra handling before docking would work at all — a good reminder that public datasets need cleaning even when they're already curated
- how sensitive are the docking scores to the exact box size/position I chose, and would a different validated pocket definition change the ranking of candidates

**v0.2.1 — Generalizing beyond EGFR**
- Writing `train_target.py` and `prepare_target_structure.py` to generalize the screening and docking steps (instead of hardcoding EGFR everywhere) was the point where this stopped feeling like a one-off script and started feeling like an actual tool
- Validated each generalization independently and on a real second target before trusting it: BRAF screening (ROC-AUC 0.827) and BRAF docking (redocking vemurafenib scored -8.705 kcal/mol, in the same strong-binder range as EGFR's erlotinib/gefitinib redocking)
- `sift_target.py` (a single entry point: give it a target name, it auto-trains if needed and shows top candidates or screens a specific molecule) was then stress-tested across very different targets and dataset sizes — JAK2 (fresh training, ROC-AUC 0.904), and GLUT4 (only 129 molecules after collapsing duplicates, ROC-AUC 0.831) — which was a good check that the approach doesn't just work on cancer-heavy, data-rich targets like EGFR/BRAF
- The repo reorganization (moving 19 loose root files into `src/`, `data/models/`, `data/structures/`, `notebooks/`) surfaced a real lesson about relative paths: `"../data/..."`-style paths only worked if scripts were run from inside `src/`, and I had to switch to `__file__`-based absolute paths to make the tool actually runnable from the project root
- at what point (how little ChEMBL data) does the screening model stop being trustworthy, versus just returning a number that looks confident but isn't

**v0.2.2 — Streamlit UI**
- Mocking up the visual design (cream/olive/copper theme, Instrument Serif + Manrope fonts) interactively before writing Streamlit code was much faster than iterating through Streamlit's reload cycle
- Found a real functional gap during review, not before: the "show top candidates" view was only ever displaying screening scores, never docking results, even for targets that already had a prepared docking structure — fixed by having that view check for a structure and dock each top candidate too
- whether it's worth the extra build effort to move off Streamlit to a hand-built Flask frontend for full design control, versus how much of that time would be better spent deepening the science

**v0.3 / v0.3.1 — Literature mining**
- Chose Gemini's free API for the LLM extraction step over Anthropic's paid API, mainly to keep the whole pipeline free end to end
- Model naming churned mid-build: `gemini-2.5-flash` (what most tutorials still reference) was already deprecated for new API keys, and I had to switch to `gemini-3-flash-preview` to get a working free-tier model — a small but real reminder that "copy the docs example" doesn't always work with fast-moving APIs
- Hit a genuinely annoying debugging stretch getting a Python script with embedded multi-line f-strings and triple-quotes written correctly through nested shell heredocs — several attempts got silently corrupted mid-paste before I gave up on incremental patches and just rewrote the whole file cleanly each time. Lesson: when quoting gets three levels deep (shell → heredoc → Python string), it's faster to regenerate the whole file than to patch it
- Also hit the free tier's real-world flakiness firsthand: a 503 "high demand" error on my first real test run, resolved by just waiting and retrying, and later ran into a genuine resource exhaustion mid-session — a reminder that "free" often means "best-effort, rate-limited," not "unreliable," but you do need to plan sessions around it
- The most useful design decision was splitting extracted ligands into "drug" vs "endogenous" categories — without it, EGFR's own natural ligands (EGF, epiregulin, epigen) got mixed in with actual therapeutic drugs (cetuximab, panitumumab), which would have been misleading in a write-up
- The Sift↔Literature bridge (screening/docking a literature-found drug with one click) turned out to be the most satisfying feature so far — it's a direct, visible version of the same validation logic used throughout the project (does the pipeline correctly score things it should already know are good candidates?), except now the "should be good" list comes from real papers instead of my own hand-picked drug names
- how much would extraction quality change with a stronger (paid-tier) model, and is it worth spending real money on this one step even if the rest of the pipeline stays free
