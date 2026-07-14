# Sift — EGFR bioactivity screening (v0.1)

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
- [ ] v0.2 — Structural docking validation (AutoDock Vina against EGFR's binding pocket)
- [ ] v0.3 — LLM-based literature mining (PubMed context + known-ligand extraction)
- [ ] v1.0 — Combined into a single tool with a simple interface

## What I learned / open questions

Building this made the "structure-activity relationship" idea; a phrase I'd only seen in lectures actually click: seeing the model rediscover real drugs like lapatinib and erlotinib purely from fingerprint patterns, without ever being told which drugs they were, made it obvious why so much of drug discovery now leans on machine learning rather than blind lab testing. It also raised a question I want to dig into more: how much of the model's success is genuine chemistry versus it just picking up on the kind of molecule ChEMBL happens to have lots of data for. Going forward, I'm curious to see whether adding structural docking changes which candidates look promising, and I'd like to make the tool itself more usable one example for instance, adding a name-to-SMILES translator so it doesn't assume the user already has a molecule's SMILES string on hand.
