import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sift_target import ensure_model, smiles_to_fp, safe, DATA_DIR
from name_to_smiles import name_to_smiles
from pipeline import dock_smiles
from literature_mining import get_literature

st.set_page_config(page_title="SIFT: A Drug Discovery Pipeline", page_icon="🧬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600&display=swap');

.stApp { background-color: #F2F0EF; font-family: 'Manrope', sans-serif; }
h2, h3, p, label, .stMarkdown { color: #606C38 !important; font-family: 'Manrope', sans-serif; }

.stTextInput input {
    background-color: #ffffff;
    color: #606C38;
    border: 1px solid #dcdad7;
    border-radius: 8px;
    font-family: 'Manrope', sans-serif;
}
.stButton>button {
    background-color: #BC6C25;
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1.5em;
    font-family: 'Manrope', sans-serif;
}
.stButton>button:hover { background-color: #a35a1f; color: #FFFFFF !important; }
.stButton>button p { color: #FFFFFF !important; }

.stRadio label p { color: #606C38 !important; font-family: 'Manrope', sans-serif; }
.stCheckbox label p { color: #606C38 !important; font-family: 'Manrope', sans-serif; }
.stTabs [data-baseweb="tab"] { color: #a3a29e; font-family: 'Manrope', sans-serif; }
.stTabs [aria-selected="true"] {
    color: #606C38 !important;
    border-bottom-color: #BC6C25 !important;
}
[data-testid="stMetricValue"] { color: #BC6C25; font-family: 'Manrope', sans-serif; }
[data-testid="stMetricLabel"] { color: #606C38; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #ffffff;
    border-radius: 10px;
    border: none;
}
pre, code {
    background-color: #ffffff !important;
    color: #BC6C25 !important;
    border: 1px solid #dcdad7 !important;
}

/* Slider track + thumb */
[data-testid="stSlider"] [role="slider"] {
    background-color: #BC6C25 !important;
    border-color: #BC6C25 !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color: #BC6C25 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='display:flex; align-items:baseline; gap:12px; margin-bottom:8px;'>"
    "<span style=\"font-family:'Instrument Serif',serif; font-size:44px; letter-spacing:1px; color:#606C38;\">SIFT</span>"
    "<span style='font-size:13px; color:#8a9264;'>A Drug Discovery Pipeline</span>"
    "</div>",
    unsafe_allow_html=True
)

tab_app, tab_lit, tab_translate, tab_commands = st.tabs(["Sift", "Literature", "SMILES translator", "Commands"])

with tab_app:
    with st.container(border=True):
        st.markdown("**Target protein**")
        target_name = st.text_input("Target protein", label_visibility="collapsed",
                                     placeholder="EGFR, BRAF, JAK2, mTOR...", key="target_sift")

        st.markdown("**What do you want to do?**")
        mode = st.radio("mode", ["Show top candidates", "Screen a specific molecule"],
                         label_visibility="collapsed")

        if mode == "Show top candidates":
            top_n = st.slider("How many candidates?", 5, 20, 10)
            run_screen = st.button("Run screening")
        else:
            molecule_name = st.text_input("Molecule name", placeholder="common or IUPAC name")
            run_screen = st.button("Screen this molecule")

    if mode == "Show top candidates" and run_screen and target_name:
        with st.spinner(f"Screening candidates against {target_name}..."):
            model = ensure_model(target_name)
            df = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
            df["fingerprint"] = df["smiles"].apply(smiles_to_fp)
            df = df.dropna(subset=["fingerprint"])
            X = np.stack(df["fingerprint"].values)
            df["predicted_prob_active"] = model.predict_proba(X)[:, 1]
            top = df.sort_values("predicted_prob_active", ascending=False).head(top_n)

        receptor_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_receptor.pdbqt")
        pocket_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_pocket.txt")
        has_structure = os.path.exists(receptor_path) and os.path.exists(pocket_path)

        dock_scores = {}
        if has_structure:
            progress = st.progress(0, text="Docking top candidates...")
            for i, (_, row) in enumerate(top.iterrows()):
                dock_scores[row["name"]] = dock_smiles(row["smiles"], safe(row["name"])[:30])
                progress.progress((i + 1) / len(top), text=f"Docking {row['name'].title()}...")
            progress.empty()

        with st.container(border=True):
            st.markdown(f"**Top {top_n} candidates for {target_name}**")
            st.caption("Score = predicted probability of activity, 0 to 1 (model confidence, not a physical unit)")
            if has_structure:
                st.caption("Click a candidate to see its docking score")
            for _, row in top.iterrows():
                header = f"{row['name'].title()}   —   {row['predicted_prob_active']:.3f}"
                with st.expander(header):
                    st.markdown(f"**Screening score:** {row['predicted_prob_active']:.3f} (predicted probability of activity, 0-1)")
                    if has_structure:
                        st.markdown(f"**Docking score:** {dock_scores[row['name']]} kcal/mol")
                        st.caption("More negative = stronger predicted binding")
                    else:
                        st.caption(f"No prepared docking structure for {target_name} yet.")

    elif mode == "Screen a specific molecule" and run_screen and target_name and molecule_name:
        with st.spinner("Resolving molecule..."):
            smiles, source = name_to_smiles(molecule_name)
        if smiles is None:
            st.error(f"Could not resolve '{molecule_name}' to a structure.")
        else:
            st.info(f"Resolved via {source}: `{smiles}`")
            with st.spinner(f"Screening against {target_name}..."):
                model = ensure_model(target_name)
                fp = smiles_to_fp(smiles)
                prob = model.predict_proba(fp.reshape(1, -1))[0][1]
            st.metric("Predicted probability of activity (0-1 scale)", f"{prob:.3f}")

            receptor_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_receptor.pdbqt")
            pocket_path = os.path.join(DATA_DIR, "structures", f"{safe(target_name)}_pocket.txt")
            if os.path.exists(receptor_path) and os.path.exists(pocket_path):
                with st.spinner("Docking..."):
                    score = dock_smiles(smiles, safe(molecule_name)[:30])
                st.metric("Predicted binding affinity", f"{score} kcal/mol")
                st.caption("More negative = stronger predicted binding")
            else:
                st.warning(f"No prepared docking structure for {target_name} yet.")

with tab_lit:
    with st.container(border=True):
        st.markdown("**Mine literature for a target**")
        st.caption("Pulls recent PubMed abstracts and uses Gemini to extract known drugs and biological ligands")

        use_same_target = st.checkbox("Use same target as Sift tab", value=True, key="lit_use_same")

        if use_same_target:
            lit_target = st.session_state.get("target_sift", "")
            if lit_target:
                st.caption(f"Target: **{lit_target}** (from Sift tab)")
            else:
                st.caption("Enter a target in the Sift tab first, or uncheck this to type one here.")
        else:
            lit_target = st.text_input("Target protein", label_visibility="collapsed",
                                        placeholder="EGFR, BRAF, JAK2, mTOR...", key="target_lit_manual")

        lit_n = st.slider("How many abstracts to mine?", 5, 20, 10, key="lit_n")
        run_lit = st.button("Mine literature")

    if run_lit and lit_target:
        with st.spinner(f"Fetching and analyzing PubMed abstracts for {lit_target}..."):
            st.session_state["lit_result"] = get_literature(lit_target, max_abstracts=lit_n)

    lit_result = st.session_state.get("lit_result")
    if lit_result:
        with st.container(border=True):
            st.markdown(f"**{lit_result['target']}** — {lit_result['num_abstracts']} abstracts mined")

            st.markdown("**Known drugs / inhibitors**")
            if lit_result["drugs"]:
                for d in lit_result["drugs"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"<div style='padding:6px 0; color:#BC6C25; font-weight:600; font-size:14px;'>• {d}</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        screen_key = f"screen_btn_{d}"
                        if st.button("Screen & dock", key=screen_key):
                            with st.spinner(f"Resolving {d}..."):
                                smiles, source = name_to_smiles(d)
                            if smiles is None:
                                st.session_state[f"result_{d}"] = {"error": f"Could not resolve '{d}' to a structure."}
                            else:
                                with st.spinner(f"Screening {d} against {lit_result['target']}..."):
                                    model = ensure_model(lit_result["target"])
                                    fp = smiles_to_fp(smiles)
                                    prob = model.predict_proba(fp.reshape(1, -1))[0][1]

                                result = {"smiles": smiles, "source": source, "prob": prob}

                                receptor_path = os.path.join(DATA_DIR, "structures", f"{safe(lit_result['target'])}_receptor.pdbqt")
                                pocket_path = os.path.join(DATA_DIR, "structures", f"{safe(lit_result['target'])}_pocket.txt")
                                if os.path.exists(receptor_path) and os.path.exists(pocket_path):
                                    with st.spinner(f"Docking {d}..."):
                                        score = dock_smiles(smiles, safe(d)[:30])
                                    result["dock_score"] = score

                                st.session_state[f"result_{d}"] = result

                    stored = st.session_state.get(f"result_{d}")
                    if stored:
                        if "error" in stored:
                            st.warning(stored["error"])
                        else:
                            msg = f"Screening score: {stored['prob']:.3f}"
                            if "dock_score" in stored:
                                msg += f"  |  Docking: {stored['dock_score']} kcal/mol"
                            st.caption(msg)
            else:
                st.caption("No specific drugs found in these abstracts.")

            st.markdown("**Natural (endogenous) ligands**")
            if lit_result["endogenous_ligands"]:
                for e in lit_result["endogenous_ligands"]:
                    st.markdown(
                        f"<div style='padding:4px 0; color:#606C38; font-size:14px;'>• {e}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No endogenous ligands found in these abstracts.")

            st.markdown("**Context summary**")
            for c in lit_result["contexts"]:
                st.caption(f"- {c}")

with tab_translate:
    st.markdown("**Translate a molecule name to its chemical structure (SMILES)**")
    st.caption("Works with common drug names or IUPAC systematic names")
    lookup_name = st.text_input("Molecule name", placeholder="e.g. aspirin, or 4-methylpent-1-ene")
    if st.button("Translate"):
        with st.spinner("Looking up..."):
            smiles, source = name_to_smiles(lookup_name)
        if smiles is None:
            st.error(f"Could not resolve '{lookup_name}' via PubChem or OPSIN.")
        else:
            st.success(f"Resolved via {source}")
            st.code(smiles, language=None)

with tab_commands:
    st.markdown("**Screen a target for top candidates:**")
    st.code('python src/sift_target.py "TARGET_NAME"', language="bash")

    st.markdown("**Screen (and dock, if prepared) a specific molecule against a target:**")
    st.code('python src/sift_target.py "TARGET_NAME" "molecule name"', language="bash")

    st.markdown("**Prepare a target for docking (needs a PDB ID + ligand code from rcsb.org):**")
    st.code('python src/prepare_target_structure.py "TARGET_NAME" "PDB_ID" "LIGAND_CODE"', language="bash")

    st.markdown("**Translate a molecule name to its chemical structure (SMILES):**")
    st.code('python src/name_to_smiles.py "molecule name"', language="bash")

    st.markdown("**Train a screening model for a target directly:**")
    st.code('python src/train_target.py "TARGET_NAME"', language="bash")

    st.markdown("**Mine PubMed literature for known ligands (cached after first run):**")
    st.code('python src/literature_mining.py "TARGET_NAME"', language="bash")

    st.markdown("**Launch this app:**")
    st.code('streamlit run src/app.py', language="bash")
