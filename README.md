# Sift — EGFR bioactivity screening + docking (v0.2)

## What this is

Sift is a personal project I'm building over summer 2026 to learn, hands-on, how AI and bioinformatics tools are actually used in early-stage drug discovery. I'm a biomedicine student, and going in I knew the theory around drug discovery only from lectures and so, this project is my way of testing the waters with the real, practical tools researchers use, using EGFR (a well-studied cancer drug target) as a concrete case study, entirely with free/public data and tools.

This isn't meant to be by any means a novel scientific contribution rather it's more a documented exploration of a real workflow, and a way to identify where my own interests and skill gaps are before committing to a research direction.

## What it does

Given a molecule (as a SMILES string), Sift predicts the probability that it would be active against EGFR (epidermal growth factor receptor), a protein targeted by several real cancer drugs.

It works by:
1. Pulling ~5,000 real, historical bioactivity measurements against EGFR from ChEMBL, a public pharmacology database
2. Converting each molecule's structure into a numerical fingerprint (RDKit, Morgan fingerprints)
3. Training a random forest classifier on this data (cross-validated ROC-AUC: 0.847)
4. Validating it against 2,686 real approved drugs it had never seen — it correctly ranked known EGFR drugs (lapatinib, neratinib, erlotinib, gefitinib, afatinib) at the very top, despite never being told which drugs those were

## Try it

\`\`\`bash
python predict.py "CC(=O)Oc1ccccc1C(=O)O"
\`\`\`

## Roadmap

- [x] v0.1 — Bioactivity screening model, validated against approved drugs
- [x] v0.2 — Structural docking validation (AutoDock Vina against EGFR's binding pocket)
- [ ] v0.3 — LLM-based literature mining (PubMed context + known-ligand extraction)
- [ ] v1.0 — Combined into a single tool with a simple interface

## What I learned / open questions (log)

**v0.1 — Bioactivity screening**
- Seeing the model rediscover real drugs (lapatinib, erlotinib) purely from fingerprint patterns made the textbook idea of "structure-activity relationship" click for the first time
- Open question: how much of the model's success is genuine chemistry versus it picking up on the kind of molecule ChEMBL happens to have lots of data for

**v0.2 — Structural docking**
- Docking made the screening model's predictions feel less abstract — going from "the model thinks this looks similar to known drugs" to "here's how it would physically sit in the actual binding pocket"
- Real-world data is messy: salt forms (e.g. lapatinib ditosylate) needed extra handling before docking would work at all — a good reminder that public datasets need cleaning even when they're already curated
- Open question: how sensitive are the docking scores to the exact box size/position I chose, and would a different validated pocket definition change the ranking of candidates
