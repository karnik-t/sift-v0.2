import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_target_structure import read_structure_meta
from sift_target import ensure_model, smiles_to_fp, safe, DATA_DIR, get_target_structure
from name_to_smiles import name_to_smiles
from pipeline import dock_smiles, read_pocket, get_pose_path, extract_top_pose, tanimoto_similarity, get_molecule_image, find_close_contacts_detailed
from literature_mining import get_literature
from streamlit_ketcher import st_ketcher
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(page_title="SIFT: A Drug Discovery Pipeline", page_icon="🧬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600&display=swap');

.stApp { background-color: #F2F0EF; font-family: 'Manrope', sans-serif; }
h2, h3, h4, p, label, .stMarkdown { color: #606C38 !important; font-family: 'Manrope', sans-serif; }

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

NAV_PAGES = ["Sift", "Literature", "3D View"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Sift"

if st.session_state.get("pending_nav"):
    st.session_state["nav_page"] = st.session_state.pop("pending_nav")


def make_status_reporter(text_slot, bar):
    counter = {"n": 0}

    def _report(msg):
        text_slot.caption(msg)
        counter["n"] += 1
        bar.progress(min(0.9, counter["n"] * 0.12))

    return _report


def compute_pose_centroid(pose_text):
    coords = []
    for line in pose_text.splitlines():
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


def pdbqt_text_to_pdb(pdbqt_text, label="structure"):
    """
    Convert PDBQT text to real PDB text via obabel, so py3Dmol gets proper
    bond-order info to render connected sticks. PDBQT (AutoDock's format)
    lacks the bond records py3Dmol's PDBQT parser relies on, so ligands and
    receptors loaded directly as "pdbqt" often render as disconnected atoms.
    Falls back to the original PDBQT text if obabel fails for any reason --
    surfaced via st.warning so a conversion failure is visible instead of
    silently degrading (which previously made debugging impossible).
    """
    import subprocess
    import tempfile
    import os as _os

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f_in:
            f_in.write(pdbqt_text)
            in_path = f_in.name
        out_path = in_path.replace(".pdbqt", ".pdb")

        result = subprocess.run(["obabel", in_path, "-O", out_path], capture_output=True, text=True)
        if result.returncode != 0:
            st.warning(f"obabel PDBQT\u2192PDB conversion failed for {label}: {result.stderr.strip()[:300]}")
            _os.remove(in_path)
            return pdbqt_text, False

        with open(out_path) as f:
            pdb_text = f.read()

        _os.remove(in_path)
        _os.remove(out_path)

        if not pdb_text.strip():
            st.warning(f"obabel produced an empty PDB file for {label}; falling back to PDBQT.")
            return pdbqt_text, False

        return pdb_text, True
    except Exception as e:
        st.warning(f"PDBQT\u2192PDB conversion error for {label}: {e}")
        return pdbqt_text, False


def mol_to_svg(smiles, width=260, height=180):
    """Render a molecule as a crisp vector image, transparent background.
    Returns an <img> tag with a base64-encoded SVG data URI rather than raw
    <svg> markup -- st.markdown's markdown-parsing pass can silently mangle
    inline <svg>/<text> tags before they reach the browser, but a base64
    string inside an img src attribute is just plain text to that parser,
    so it always survives intact."""
    import base64

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("fill:#FFFFFF;stroke:none", "fill:none;stroke:none")

    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" height="{height}" />'


def get_similarity_to_known_drugs(smiles, target_name, top_n=3):
    lit_result = st.session_state.get("lit_result")
    if not lit_result or lit_result.get("target") != target_name:
        return None

    drugs = lit_result.get("drugs", [])
    if not drugs:
        return []

    cache_key = f"known_drug_fps_{safe(target_name)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    fp_cache = st.session_state[cache_key]

    query_fp = smiles_to_fp(smiles)
    if query_fp is None:
        return []

    results = []
    for drug_name in drugs:
        if drug_name not in fp_cache:
            drug_smiles, _ = name_to_smiles(drug_name)
            fp_cache[drug_name] = smiles_to_fp(drug_smiles) if drug_smiles else None

        drug_fp = fp_cache[drug_name]
        if drug_fp is not None:
            sim = tanimoto_similarity(query_fp, drug_fp)
            if sim is not None:
                results.append((drug_name, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def ensure_literature(target_name, max_abstracts=8):
    """
    Auto-fetch and cache literature mining results for a target so the
    similarity-to-known-drugs feature always has context ready, without
    requiring a manual visit to the Literature tab. Cheap after the first
    run: get_literature() caches to disk per target, so repeat targets
    cost zero additional Gemini calls.
    """
    cached = st.session_state.get("lit_result")
    if cached and cached.get("target") == target_name:
        return cached
    try:
        result = get_literature(target_name, max_abstracts=max_abstracts)
        st.session_state["lit_result"] = result
        return result
    except Exception as e:
        st.caption(f"Literature context unavailable right now ({e}); screening will still run.")
        return None


def jump_to_3d_with_molecule(molecule_name, target_name):
    text_slot = st.empty()
    bar = st.progress(0)
    receptor_path, box_center = get_target_structure(target_name, status=make_status_reporter(text_slot, bar))
    bar.progress(1.0 if receptor_path else 0.9)
    text_slot.caption("Docking structure ready" if receptor_path else "No docking structure found")
    ligand_text = None
    dock_score = None

    smiles, source = name_to_smiles(molecule_name)
    if receptor_path and smiles:
        dock_score = dock_smiles(smiles, safe(molecule_name)[:30],
                                  receptor_path=receptor_path, box_center=box_center)
        pose_path = get_pose_path(safe(molecule_name)[:30], receptor_path)
        if os.path.exists(pose_path):
            ligand_text = extract_top_pose(pose_path)

    if receptor_path:
        with open(receptor_path) as f:
            receptor_text = f.read()
        st.session_state["viz_state"] = {
            "target": target_name,
            "receptor_text": receptor_text,
            "box_center": box_center,
            "ligands": [{"name": molecule_name, "pose_text": ligand_text, "dock_score": dock_score}],
        }
        st.session_state["viz_molecule_name"] = molecule_name

    st.session_state["pending_nav"] = "3D View"
    st.rerun()


def render_sift():
    with st.container(border=True):
        st.markdown("**Target protein**")
        target_name = st.text_input("Target protein", label_visibility="collapsed",
                                     placeholder="EGFR, BRAF, JAK2, mTOR...", key="target_sift")

        st.markdown("**What do you want to do?**")
        mode = st.radio("mode", ["Show top candidates", "Screen a specific molecule"],
                         label_visibility="collapsed", key="sift_mode")

        molecule_name = None
        drawn_smiles = None

        if mode == "Show top candidates":
            top_n = st.slider("How many candidates?", 5, 20, 10, key="sift_top_n")
            run_screen = st.button("Run screening", key="sift_run_screen")
        else:
            input_method = st.radio("How do you want to provide the molecule?",
                                     ["Type a name", "Draw a structure"],
                                     horizontal=True, key="molecule_input_method")

            if input_method == "Type a name":
                molecule_name = st.text_input("Molecule name", placeholder="common or IUPAC name", key="sift_molecule_name")
            else:
                st.caption("Draw a structure below, then click the checkmark in the editor to confirm it.")
                drawn_smiles = st_ketcher(key="ketcher_editor")

            run_screen = st.button("Screen this molecule", key="sift_run_screen_molecule")

    if mode == "Show top candidates" and run_screen and target_name:
        with st.spinner(f"Screening candidates against {target_name}..."):
            model = ensure_model(target_name)
            df = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
            df["fingerprint"] = df["smiles"].apply(smiles_to_fp)
            df = df.dropna(subset=["fingerprint"])
            X = np.stack(df["fingerprint"].values)
            df["predicted_prob_active"] = model.predict_proba(X)[:, 1]
            top = df.sort_values("predicted_prob_active", ascending=False).head(top_n)

        with st.spinner(f"Checking literature context for {target_name}..."):
            ensure_literature(target_name)

        text_slot = st.empty()
        bar = st.progress(0)
        receptor_path, box_center = get_target_structure(target_name, status=make_status_reporter(text_slot, bar))
        bar.progress(1.0 if receptor_path else 0.9)
        text_slot.caption("Docking structure ready" if receptor_path else "No docking structure found")
        has_structure = receptor_path is not None

        if has_structure:
            meta = read_structure_meta(target_name)
            if meta.get("source") == "alphafold":
                st.caption(f"⚠️ Docking structure: AlphaFold prediction (mean pLDDT {meta.get('mean_plddt', '?')}) "
                           f"— predicted, not experimentally verified. Treat binding scores as lower-confidence.")
            else:
                pdb_note = f" (PDB {meta['pdb_id']})" if meta.get("pdb_id") else ""
                st.caption(f"Docking structure: real crystal structure{pdb_note}")

        dock_scores = {}
        if has_structure:
            progress = st.progress(0, text="Docking top candidates...")
            for i, (_, row) in enumerate(top.iterrows()):
                dock_scores[row["name"]] = dock_smiles(row["smiles"], safe(row["name"])[:30],
                                                        receptor_path=receptor_path, box_center=box_center)
                progress.progress((i + 1) / len(top), text=f"Docking {row['name'].title()}...")
            progress.empty()

        st.session_state["sift_last_top"] = top
        st.session_state["sift_last_dock_scores"] = dock_scores
        st.session_state["sift_last_has_structure"] = has_structure
        st.session_state["sift_last_target"] = target_name
        st.session_state["sift_last_top_n"] = top_n

    if st.session_state.get("sift_last_target") == target_name and "sift_last_top" in st.session_state:
        top = st.session_state["sift_last_top"]
        dock_scores = st.session_state["sift_last_dock_scores"]
        has_structure = st.session_state["sift_last_has_structure"]
        shown_top_n = st.session_state["sift_last_top_n"]

        with st.container(border=True):
            st.markdown(f"**Top {shown_top_n} candidates for {target_name}**")
            st.caption("Score = predicted probability of activity, 0 to 1 (model confidence, not a physical unit)")
            if has_structure:
                st.caption("Click a candidate to see its docking score, or jump straight to the 3D view")
            for _, row in top.iterrows():
                header = f"{row['name'].title()}   —   {row['predicted_prob_active']:.3f}"
                with st.expander(header):
                    col_struct, col_info = st.columns([1, 1.4])
                    with col_struct:
                        svg = mol_to_svg(row["smiles"])
                        if svg:
                            st.markdown(svg, unsafe_allow_html=True)
                    with col_info:
                        st.markdown(f"**Screening score:** {row['predicted_prob_active']:.3f} (predicted probability of activity, 0-1)")
                        if has_structure:
                            st.markdown(f"**Docking score:** {dock_scores[row['name']]} kcal/mol")
                            st.caption("More negative = stronger predicted binding")
                            if st.button("View in 3D →", key=f"view3d_{row['name']}"):
                                jump_to_3d_with_molecule(row["name"], target_name)
                        else:
                            st.caption(f"No prepared docking structure for {target_name} yet.")

    if mode == "Screen a specific molecule" and run_screen and target_name and (molecule_name or drawn_smiles):
        if drawn_smiles:
            smiles, source = drawn_smiles, "drawn structure"
        else:
            with st.spinner("Resolving molecule..."):
                smiles, source = name_to_smiles(molecule_name)

        if not smiles:
            st.session_state["sift_last_molecule_result"] = None
            st.error(f"Could not resolve '{molecule_name}' to a structure.")
        else:
            with st.spinner(f"Screening against {target_name}..."):
                model = ensure_model(target_name)
                fp = smiles_to_fp(smiles)
                prob = model.predict_proba(fp.reshape(1, -1))[0][1]

            with st.spinner(f"Checking literature context for {target_name}..."):
                ensure_literature(target_name)

            text_slot = st.empty()
            bar = st.progress(0)
            receptor_path, box_center = get_target_structure(target_name, status=make_status_reporter(text_slot, bar))
            bar.progress(1.0 if receptor_path else 0.9)
            text_slot.caption("Docking structure ready" if receptor_path else "No docking structure found")

            score = None
            if receptor_path is not None:
                with st.spinner("Docking..."):
                    label = molecule_name if molecule_name else "drawn_molecule"
                    score = dock_smiles(smiles, safe(label)[:30],
                                        receptor_path=receptor_path, box_center=box_center)

            st.session_state["sift_last_molecule_result"] = {
                "target": target_name,
                "molecule_name": molecule_name,
                "smiles": smiles,
                "source": source,
                "prob": prob,
                "score": score,
                "has_structure": receptor_path is not None,
            }

    stored_single = st.session_state.get("sift_last_molecule_result")
    if mode == "Screen a specific molecule" and stored_single and stored_single["target"] == target_name:
        col_struct, col_info = st.columns([1, 1.4])

        with col_struct:
            svg = mol_to_svg(stored_single["smiles"], width=280, height=220)
            if svg:
                st.markdown(svg, unsafe_allow_html=True)

        with col_info:
            st.metric("Predicted probability of activity (0-1 scale)", f"{stored_single['prob']:.3f}")

            similarities = get_similarity_to_known_drugs(stored_single["smiles"], target_name)
            if similarities is None:
                st.caption("Mine literature for this target (Literature tab) to compare chemical similarity to known drugs.")
            elif similarities:
                st.markdown("**Chemical similarity to known drugs**")
                st.caption("Tanimoto similarity, 0 (no shared structure) to 1 (identical)")
                for drug_name, sim in similarities:
                    st.caption(f"- {drug_name.title()}: {sim:.2f}")

            if stored_single["has_structure"] and stored_single["score"] is not None:
                st.metric("Predicted binding affinity", f"{stored_single['score']} kcal/mol")
                st.caption("More negative = stronger predicted binding")
                if stored_single["molecule_name"] and st.button("View in 3D →", key="view3d_single_molecule"):
                    jump_to_3d_with_molecule(stored_single["molecule_name"], target_name)
            else:
                st.warning(f"No prepared docking structure for {target_name} yet.")


def render_literature():
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
        run_lit = st.button("Mine literature", key="lit_run")

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

                                text_slot = st.empty()
                                bar = st.progress(0)
                                receptor_path, box_center = get_target_structure(lit_result["target"], status=make_status_reporter(text_slot, bar))
                                bar.progress(1.0 if receptor_path else 0.9)
                                text_slot.caption("Docking structure ready" if receptor_path else "No docking structure found")
                                if receptor_path is not None:
                                    with st.spinner(f"Docking {d}..."):
                                        score = dock_smiles(smiles, safe(d)[:30],
                                                            receptor_path=receptor_path, box_center=box_center)
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
                            if "dock_score" in stored and st.button("View in 3D →", key=f"view3d_lit_{d}"):
                                jump_to_3d_with_molecule(d, lit_result["target"])
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


def render_3d():
    with st.container(border=True):
        st.markdown("**3D structure viewer**")
        st.caption("Shows the target's binding pocket, and optionally a docked molecule's pose")

        use_same_target_3d = st.checkbox("Use same target as Sift tab", value=True, key="viz_use_same")

        if use_same_target_3d:
            viz_target = st.session_state.get("target_sift", "")
            if viz_target:
                st.caption(f"Target: **{viz_target}** (from Sift tab)")
            else:
                st.caption("Enter a target in the Sift tab first, or uncheck this to type one here.")
        else:
            viz_target = st.text_input("Target protein", label_visibility="collapsed",
                                        placeholder="EGFR, BRAF, JAK2, mTOR...", key="target_viz_manual")

        st.markdown("**Style controls**")
        col_a, col_b = st.columns(2)
        with col_a:
            protein_style = st.selectbox("Protein style", ["Cartoon", "Stick", "Surface"], key="viz_protein_style")
            protein_color = st.selectbox("Protein color", ["grey", "white", "spectrum", "cyan", "green"], key="viz_protein_color")
        with col_b:
            ligand_color = st.selectbox("Ligand color", ["orange", "magenta", "yellow", "cyan", "green"], key="viz_ligand_color")
            show_pocket = st.checkbox("Highlight binding pocket", value=True, key="viz_show_pocket")

        pocket_color = st.selectbox("Pocket highlight color", ["magenta", "yellow", "cyan", "red"], key="viz_pocket_color") if show_pocket else None

        show_contacts = st.checkbox("Show close contacts (geometric proximity, not validated bond types)", value=True, key="viz_show_contacts")
        auto_rotate = st.checkbox("Auto-rotate on load", value=False, key="viz_auto_rotate")

        st.markdown("**Optional: dock and show one or more molecules' poses**")
        st.caption("Add molecules one at a time to compare their binding poses side by side")

        if "viz_molecule_list" not in st.session_state:
            st.session_state["viz_molecule_list"] = []

        def _add_molecule_callback():
            name_clean = st.session_state.get("viz_new_molecule", "").strip()
            if name_clean and name_clean not in st.session_state["viz_molecule_list"]:
                st.session_state["viz_molecule_list"].append(name_clean)
            st.session_state["viz_new_molecule"] = ""

        add_col, btn_col = st.columns([4, 1])
        with add_col:
            st.text_input("Molecule name", placeholder="e.g. erlotinib",
                           key="viz_new_molecule", label_visibility="collapsed")
        with btn_col:
            st.button("Add", key="viz_add_molecule", on_click=_add_molecule_callback)

        if "viz_molecule_visibility" not in st.session_state:
            st.session_state["viz_molecule_visibility"] = {}

        if st.session_state["viz_molecule_list"]:
            for i, m in enumerate(st.session_state["viz_molecule_list"]):
                vcol, mcol, xcol = st.columns([1, 4, 1])
                with vcol:
                    visible = st.checkbox(
                        "show", value=st.session_state["viz_molecule_visibility"].get(m, True),
                        key=f"viz_visible_{i}_{m}", label_visibility="collapsed"
                    )
                    st.session_state["viz_molecule_visibility"][m] = visible
                with mcol:
                    st.caption(f"• {m}")
                with xcol:
                    if st.button("Remove", key=f"viz_remove_{i}_{m}"):
                        st.session_state["viz_molecule_list"].pop(i)
                        st.session_state["viz_molecule_visibility"].pop(m, None)
                        st.rerun()
        else:
            st.caption("No molecules added yet.")

        run_viz = st.button("Render 3D view", key="viz_run")

    if run_viz and viz_target:
        text_slot = st.empty()
        bar = st.progress(0)
        receptor_path, box_center = get_target_structure(viz_target, status=make_status_reporter(text_slot, bar))
        bar.progress(1.0 if receptor_path else 0.9)
        text_slot.caption("Docking structure ready" if receptor_path else "No docking structure found")

        if receptor_path is not None:
            meta = read_structure_meta(viz_target)
            if meta.get("source") == "alphafold":
                st.caption(f"⚠️ AlphaFold prediction (mean pLDDT {meta.get('mean_plddt', '?')}) "
                           f"— predicted, not experimentally verified.")
            else:
                pdb_note = f" (PDB {meta['pdb_id']})" if meta.get("pdb_id") else ""
                st.caption(f"Real crystal structure{pdb_note}")

        if receptor_path is None:
            st.warning(f"No prepared docking structure for {viz_target} yet. Run prepare_target_structure.py first.")
            st.session_state["viz_state"] = None
        else:
            with open(receptor_path) as f:
                receptor_text = f.read()

            molecule_names = st.session_state.get("viz_molecule_list", [])
            ligands = []

            if molecule_names:
                progress = st.progress(0, text="Resolving and docking...")
                for i, mol_name in enumerate(molecule_names):
                    smiles, source = name_to_smiles(mol_name)
                    if smiles:
                        score = dock_smiles(smiles, safe(mol_name)[:30],
                                             receptor_path=receptor_path, box_center=box_center)
                        pose_path = get_pose_path(safe(mol_name)[:30], receptor_path)
                        pose_text = extract_top_pose(pose_path) if os.path.exists(pose_path) else None
                        ligands.append({
                            "name": mol_name,
                            "pose_text": pose_text,
                            "dock_score": score,
                            "smiles": smiles,
                            "visible": st.session_state["viz_molecule_visibility"].get(mol_name, True),
                        })
                    else:
                        st.warning(f"Could not resolve '{mol_name}' to a structure.")
                    progress.progress((i + 1) / len(molecule_names), text=f"Docking {mol_name}...")
                progress.empty()

            st.session_state["viz_state"] = {
                "target": viz_target,
                "receptor_text": receptor_text,
                "box_center": box_center,
                "ligands": ligands,
            }

    LIGAND_PALETTE = ["orange", "magenta", "cyan", "yellow", "lime", "hotpink", "turquoise", "gold"]

    viz_state = st.session_state.get("viz_state")
    if viz_state and viz_state["target"] == viz_target:
        view = py3Dmol.view(width=700, height=500)
        receptor_pdb_text, receptor_ok = pdbqt_text_to_pdb(viz_state["receptor_text"], label="receptor")
        view.addModel(receptor_pdb_text, "pdb" if receptor_ok else "pdbqt")

        if protein_style == "Cartoon":
            view.setStyle({"model": 0}, {"cartoon": {"color": protein_color}})
        elif protein_style == "Stick":
            view.setStyle({"model": 0}, {"stick": {"color": protein_color}})
        else:
            view.setStyle({"model": 0}, {"cartoon": {"color": protein_color}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.6, "color": protein_color}, {"model": 0})

        ligands = viz_state.get("ligands", [])
        model_index = 1
        legend_entries = []
        contacts_by_ligand = {}

        visibility_state = st.session_state.get("viz_molecule_visibility", {})
        for i, lig in enumerate(ligands):
            if not lig["pose_text"] or not visibility_state.get(lig["name"], True):
                continue
            color = LIGAND_PALETTE[i % len(LIGAND_PALETTE)]
            ligand_pdb_text, ligand_ok = pdbqt_text_to_pdb(lig["pose_text"], label=lig["name"])
            view.addModel(ligand_pdb_text, "pdb" if ligand_ok else "pdbqt")
            view.setStyle({"model": model_index}, {"stick": {"color": color}})

            # Ligand efficiency: docking score per heavy atom, a size-normalized
            # metric so a small molecule with a modest score isn't unfairly
            # penalized against a larger one that scores better purely from bulk.
            heavy_atoms = None
            if lig.get("smiles"):
                mol = Chem.MolFromSmiles(lig["smiles"])
                if mol:
                    heavy_atoms = mol.GetNumHeavyAtoms()
            ligand_efficiency = (lig["dock_score"] / heavy_atoms
                                  if lig["dock_score"] is not None and heavy_atoms else None)

            legend_entries.append((lig["name"], color, lig["dock_score"], ligand_efficiency, ligand_pdb_text))
            model_index += 1

        if show_pocket:
            cx, cy, cz = viz_state["box_center"]
            view.addSphere({
                "center": {"x": cx, "y": cy, "z": cz},
                "radius": 8,
                "color": pocket_color,
                "opacity": 0.35,
            })

        if show_contacts:
            for lig in ligands:
                if not lig["pose_text"] or not visibility_state.get(lig["name"], True):
                    continue
                contacts = find_close_contacts_detailed(viz_state["receptor_text"], lig["pose_text"], cutoff=4.0)
                contacts_by_ligand[lig["name"]] = contacts
                for c in contacts:
                    lx, ly, lz = c["ligand_coord"]
                    rx, ry, rz = c["receptor_coord"]
                    line_color = "dodgerblue" if c["contact_type"] == "polar" else "grey"
                    view.addCylinder({
                        "start": {"x": lx, "y": ly, "z": lz},
                        "end": {"x": rx, "y": ry, "z": rz},
                        "radius": 0.05,
                        "color": line_color,
                        "dashed": True,
                    })

        view.zoomTo()
        if auto_rotate:
            view.spin(True)

        with st.container(border=True):
            visible_names = [l["name"] for l in ligands if l["pose_text"] and visibility_state.get(l["name"], True)]
            names = ", ".join(visible_names)
            st.markdown(f"**{viz_state['target']}** structure" + (f" with **{names}** docked" if names else ""))
            if auto_rotate:
                st.caption("Auto-rotating — dragging may briefly conflict with rotation "
                           "(known limitation). Uncheck 'Auto-rotate' for smooth manual control.")
            else:
                st.caption("Click and drag to rotate.")
            showmol(view, height=500, width=700)
            st.caption("Legend: sphere = approximate binding pocket center. "
                       "Blue dashed lines = polar contacts (both atoms N/O, possible H-bond). "
                       "Grey dashed lines = hydrophobic contacts.")

            for name, color, score, le, ligand_pdb_text in legend_entries:
                le_text = f", ligand efficiency {le:.3f} kcal/mol per heavy atom" if le is not None else ""
                st.caption(f"● {name} ({color}) — docking score {score} kcal/mol{le_text}")

                residues = contacts_by_ligand.get(name, [])
                if residues:
                    unique_residues = sorted({f"{c['residue']} (chain {c['chain']})" if c["chain"] else c["residue"]
                                               for c in residues})
                    st.caption(f"   Contact residues: {', '.join(unique_residues)}")

                st.download_button(
                    f"Download {name} pose (.pdb)",
                    data=ligand_pdb_text,
                    file_name=f"{safe(name)}_docked_pose.pdb",
                    mime="chemical/x-pdb",
                    key=f"download_{safe(name)}",
                )

            if show_contacts:
                st.caption("Contacts are a geometric proximity check, not a validated interaction "
                           "analysis (hydrogen bonds vs hydrophobic contacts etc. would need a "
                           "dedicated tool like PLIP).")

                centroids = {}
                for lig in ligands:
                    if lig["pose_text"]:
                        c = compute_pose_centroid(lig["pose_text"])
                        if c:
                            centroids[lig["name"]] = c

                names_list = list(centroids.keys())
                overlap_pairs = []
                for i in range(len(names_list)):
                    for j in range(i + 1, len(names_list)):
                        a, b = centroids[names_list[i]], centroids[names_list[j]]
                        dist = ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5
                        if dist < 3.0:
                            overlap_pairs.append((names_list[i], names_list[j], dist))

                if overlap_pairs:
                    pair_text = ", ".join(f"{a.title()} & {b.title()} ({d:.1f} Å apart)" for a, b, d in overlap_pairs)
                    st.warning(f"Possible overlap: {pair_text} -- these poses sit very close together, "
                               f"suggesting they may compete for the same spot in the pocket. "
                               f"(Rough estimate based on center-of-mass distance, not a full steric clash check.)")


def render_commands():
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


# --- Top-level navigation ---

st.radio("nav", NAV_PAGES, horizontal=True, key="nav_page", label_visibility="collapsed")

page = st.session_state["nav_page"]
try:
    if page == "Sift":
        render_sift()
    elif page == "Literature":
        render_literature()
    elif page == "3D View":
        render_3d()
except Exception as e:
    st.error("Something went wrong while loading this page.")
    with st.expander("Technical details"):
        st.exception(e)
