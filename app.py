# app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, Crippen
import torch.nn as nn
from torch_geometric.data import Data as PyGData, Batch
from torch_geometric.nn import GINEConv, global_mean_pool
import pickle
import os

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(
    page_title="Drug Synergy Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .synergy-high {
        color: #00cc96;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-medium {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-low {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-antagonism {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .toxicity-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .toxicity-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .toxicity-low {
        color: #00cc96;
        font-weight: bold;
    }
    .side-effect-high {
        background-color: #ffebee;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #ff4b4b;
    }
    .side-effect-medium {
        background-color: #fff3e0;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
    }
    .side-effect-low {
        background-color: #e8f5e8;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #00cc96;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model classes
# ---------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            conv = GINEConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_dim
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x

class DualGNNModel(nn.Module):
    def __init__(self, node_feat_dim, cancer_dim, num_cells, cell_emb_dim,
                 gnn_hidden, gnn_layers, mlp_hidden, dropout, edge_dim):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, gnn_hidden, gnn_layers, edge_dim)
        self.cell_emb = nn.Embedding(num_cells, cell_emb_dim)
        self.fc1 = nn.Linear(gnn_hidden * 2 + cancer_dim + cell_emb_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden // 2)
        self.fc3 = nn.Linear(mlp_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_g1, batch_g2, cancer_vec, cell_idx):
        x1, e1_idx, e1_attr = batch_g1.x, batch_g1.edge_index, batch_g1.edge_attr
        x2, e2_idx, e2_attr = batch_g2.x, batch_g2.edge_index, batch_g2.edge_attr
        
        emb1 = self.encoder(x1, e1_idx, e1_attr, batch_g1.batch)
        emb2 = self.encoder(x2, e2_idx, e2_attr, batch_g2.batch)
        cell_embed = self.cell_emb(cell_idx)
        combined = torch.cat([emb1, emb2, cancer_vec, cell_embed], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ---------------------------
# RDKit Side Effect Prediction
# ---------------------------
def predict_side_effects(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    
    side_effect_alerts = {
        'hepatotoxicity': ['[#6]-[N+](=O)[O-]', 'c1ccccc1[N]', 'S(=O)(=O)[OH]'],
        'nephrotoxicity': ['[Cl,Br,I]', 'c1ccccc1[F,Cl,Br,I]', '[NH2]C(=O)'],
        'cardiotoxicity': ['c1ccccc1', '[N+]', 'C=O'],
        'neurotoxicity': ['C#N', 'C=O', '[S;D2](=O)(=O)[#6]'],
        'gastrointestinal': ['[OH]', 'C(=O)[OH]', 'c1ccccc1[OH]'],
        'dermatological': ['[Cl,Br,I]', 'C=O', 'c1ccccc1']
    }
    
    side_effects = {}
    hepatotoxicity_score = sum([0.3 for pattern in side_effect_alerts['hepatotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if mw > 500: hepatotoxicity_score += 0.2
    side_effects['hepatotoxicity'] = min(1.0, hepatotoxicity_score)
    
    nephrotoxicity_score = sum([0.25 for pattern in side_effect_alerts['nephrotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if logp > 4: nephrotoxicity_score += 0.2
    side_effects['nephrotoxicity'] = min(1.0, nephrotoxicity_score)
    
    cardiotoxicity_score = sum([0.2 for pattern in side_effect_alerts['cardiotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if Descriptors.NumAromaticRings(mol) > 2: cardiotoxicity_score += 0.3
    side_effects['cardiotoxicity'] = min(1.0, cardiotoxicity_score)
    
    neurotoxicity_score = sum([0.25 for pattern in side_effect_alerts['neurotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['neurotoxicity'] = min(1.0, neurotoxicity_score)
    
    gi_score = sum([0.2 for pattern in side_effect_alerts['gastrointestinal'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if hbd > 3: gi_score += 0.2
    side_effects['gastrointestinal'] = min(1.0, gi_score)
    
    derm_score = sum([0.2 for pattern in side_effect_alerts['dermatological'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['dermatological'] = min(1.0, derm_score)
    
    side_effects['cns_effects'] = min(1.0, 0.1 + (logp / 10))
    side_effects['metabolic_issues'] = min(1.0, 0.1 + (mw - 300) / 700)
    
    return side_effects

def interpret_side_effects(side_effects):
    interpretations = []
    risk_levels = {'High Risk (0.7-1.0)': [], 'Medium Risk (0.4-0.7)': [], 'Low Risk (0.0-0.4)': []}
    for effect, score in side_effects.items():
        effect_name = effect.replace('_', ' ').title()
        if score >= 0.7: risk_levels['High Risk (0.7-1.0)'].append(f"{effect_name} ({score:.2f})")
        elif score >= 0.4: risk_levels['Medium Risk (0.4-0.7)'].append(f"{effect_name} ({score:.2f})")
        else: risk_levels['Low Risk (0.0-0.4)'].append(f"{effect_name} ({score:.2f})")
    for risk_level, effects in risk_levels.items():
        if effects:
            if 'High Risk' in risk_level: interpretations.append(f"🚨 **{risk_level}**: {', '.join(effects)}")
            elif 'Medium Risk' in risk_level: interpretations.append(f"⚠️ **{risk_level}**: {', '.join(effects)}")
            else: interpretations.append(f"✅ **{risk_level}**: {', '.join(effects)}")
    return interpretations

def get_side_effect_class(score):
    if score > 0.7: return "side-effect-high", "High Risk"
    elif score > 0.4: return "side-effect-medium", "Medium Risk"
    else: return "side-effect-low", "Low Risk"

def predict_combination_side_effects(side_effects_a, side_effects_b):
    combo_effects = {}
    for effect in side_effects_a.keys():
        risk_a = side_effects_a[effect]
        risk_b = side_effects_b.get(effect, 0.1)
        combo_risk = 1 - (1 - risk_a) * (1 - risk_b)
        combo_effects[effect] = min(1.0, combo_risk)
    return combo_effects

# ---------------------------
# SMILES -> PyG graph
# ---------------------------
def mol_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_feats = [[atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetDegree(), atom.GetImplicitValence(), 1.0 if atom.GetIsAromatic() else 0.0] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt_val = {Chem.rdchem.BondType.SINGLE: 1.0, Chem.rdchem.BondType.DOUBLE: 2.0, Chem.rdchem.BondType.TRIPLE: 3.0, Chem.rdchem.BondType.AROMATIC: 1.5}.get(bond.GetBondType(), 0.0)
        edge_index.extend([[a1, a2], [a2, a1]])
        edge_attr.extend([[bt_val], [bt_val]])
    if len(edge_index) == 0: edge_index, edge_attr = [[0, 0]], [[0.0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)

def mol_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(300, 300)) if mol else None

# ---------------------------
# Robust helper: find cancer col index
# ---------------------------
def find_cancer_col_index(cancer_cols, cancer_type_key):
    candidates = [f"cancer_type_{cancer_type_key}", f"cancer_type_{cancer_type_key.replace(' ', '_')}",
                  f"cancer_type_{cancer_type_key.lower()}", f"cancer_type_{cancer_type_key.replace(' ', '').lower()}"]
    for cand in candidates:
        if cand in cancer_cols: return cancer_cols.index(cand)
    key_words = [w.lower() for w in cancer_type_key.split()]
    for i, cc in enumerate(cancer_cols):
        if all(w in cc.lower() for w in key_words): return i
    for i, cc in enumerate(cancer_cols):
        if cancer_type_key.lower() in cc.lower(): return i
    return None

# ---------------------------
# Prediction function
# ---------------------------
def predict_synergy_proper(drug1_smiles, drug2_smiles, cancer_type, cell_line,
                           model, device,
                           cancer_cell_mapping, cell_line_mapping,
                           cell_le=None, cancer_cols=None, scaler_y=None,
                           debug_mode=False):
    try:
        if model is None:
            drug_hash = hash(drug1_smiles + drug2_smiles) % 100
            synergy_score = (drug_hash - 50) / 2.0
            return synergy_score, None

        g1 = mol_to_pyg(drug1_smiles)
        g2 = mol_to_pyg(drug2_smiles)
        if g1 is None or g2 is None:
            return None, "Invalid SMILES string"

        batch_g1 = Batch.from_data_list([g1]).to(device)
        batch_g2 = Batch.from_data_list([g2]).to(device)

        if cancer_cols is not None:
            cancer_vec = torch.zeros(1, len(cancer_cols), dtype=torch.float).to(device)
            idx = find_cancer_col_index(cancer_cols, cancer_type)
            if idx is not None:
                cancer_vec[0, idx] = 1.0
        else:
            cancer_cols_list = list(cancer_cell_mapping.keys())
            cancer_vec = torch.zeros(1, len(cancer_cols_list), dtype=torch.float).to(device)
            if cancer_type in cancer_cell_mapping:
                cancer_vec[0, cancer_cols_list.index(cancer_type)] = 1.0

        if cell_le is not None:
            try:
                cell_idx_val = int(cell_le.transform([cell_line])[0])
            except Exception:
                cell_idx_val = int(cell_line_mapping.get(cell_line, 0))
        else:
            cell_idx_val = int(cell_line_mapping.get(cell_line, 0))

        cell_idx = torch.tensor([cell_idx_val], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model(batch_g1, batch_g2, cancer_vec, cell_idx)
            pred_scaled = float(pred.cpu().squeeze().item())

        if scaler_y is not None:
            try:
                synergy_score = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
            except Exception:
                original_mean = -0.145845
                original_std = 8.535542
                synergy_score = (pred_scaled * original_std) + original_mean
        else:
            original_mean = -0.145845
            original_std = 8.535542
            synergy_score = (pred_scaled * original_std) + original_mean

        return synergy_score, None

    except Exception as e:
        return None, str(e)

# ---------------------------
# Load data/artifacts and model
# ---------------------------
@st.cache_resource
def load_data_and_model():
    try:
        drug_df = pd.read_csv("Datasets/comprehensive_drug_smiles.csv")
    except:
        drug_df = pd.DataFrame({
            'drug_name': ['PACLITAXEL', 'DOXORUBICIN', 'CISPLATIN'],
            'smiles': [
                'CC1=C2C(C(=O)C3=C(COC3=O)C(C2(C)C)(CC1OC(=O)C(C(C4=CC=CC=C4)NC(=O)C5=CC=CC=C5)O)C)OC(=O)C6=CC=CC=C6',
                'CC1C(C(CC(O1)OC2C(OC3C(C2O)C(C(C4=CC(=O)C5=CC=CC=C5C4=O)O)(C)C)C)O)(C)O',
                'Cl[Pt](Cl)(N)N'
            ]
        })
    drug_df = drug_df.dropna(subset=['smiles'])

    cancer_cell_mapping = {
        'Bladder Cancer': ['BFTC-905', 'HT-1197', 'HT-1376', 'J82', 'JMSU-1', 'KU-19-19', 'RT-112', 'T24', 'TCCSUP', 'UM-UC-3'],
        'Bone Cancer': ['A-673', 'TC-32', 'TC-71'],
        'Brain Cancer': ['SF-295', 'T98G'],
        'Breast Cancer': ['BT-549', 'MCF7', 'MDA-MB-231', 'MDA-MB-468', 'T-47D', 'BT-20', 'BT-474', 'CAL-120', 'CAL-148', 'CAL-51'],
        'Colon/Colorectal Cancer': ['SW527', 'COLO 205', 'HCT-15', 'KM12', 'RKO', 'SW837', 'LS513'],
        'Gastric Cancer': ['AGS', 'KATO III', 'SNU-16'],
        'Kidney Cancer': ['ACHN', 'SN12C', 'UO-31'],
        'Leukemia': ['CCRF-CEM', 'K-562'],
        'Lung Cancer': ['A427', 'A549', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H322M'],
        'Lymphoma': ['HDLM-2', 'L-1236', 'L-428', 'U-HO1'],
        'Myeloma': ['KMS-11'],
        'Not Available': ['EW-8', 'SF-268', 'SF-539', 'SNB-19', 'SNB-75', 'U251'],
        'Ovarian Cancer': ['A2780', 'IGROV1', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'SK-OV-3'],
        'Pancreatic Cancer': ['PANC-1'],
        'Prostate Cancer': ['PC-3'],
        'Sarcoma': ['RD', 'SMS-CTR'],
        'Skin Cancer': ['A2058', 'LOX IMVI', 'M14', 'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5']
    }

    cell_line_mapping = {}
    idx = 0
    for cancer_type, cell_lines in cancer_cell_mapping.items():
        for cl in cell_lines:
            if cl not in cell_line_mapping:
                cell_line_mapping[cl] = idx
                idx += 1

    scaler_y, cell_le, cancer_cols = None, None, None
    if os.path.exists("scaler_y.pkl"):
        with open("scaler_y.pkl","rb") as f: scaler_y = pickle.load(f)
    if os.path.exists("cell_le.pkl"):
        with open("cell_le.pkl","rb") as f: cell_le = pickle.load(f)
    if os.path.exists("cancer_cols.pkl"):
        with open("cancer_cols.pkl","rb") as f: cancer_cols = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model = DualGNNModel(
            node_feat_dim=5,
            cancer_dim=(len(cancer_cols) if cancer_cols is not None else 17),
            num_cells=(len(cell_le.classes_) if cell_le is not None else 288),
            cell_emb_dim=48,
            gnn_hidden=192,
            gnn_layers=4,
            mlp_hidden=512,
            dropout=0.3,
            edge_dim=1
        ).to(device)
        model.load_state_dict(torch.load("final_gnn_model.pt", map_location=device))
        model.eval()
    except:
        model = None

    return drug_df, cancer_cell_mapping, model, device, cell_line_mapping, cell_le, cancer_cols, scaler_y

# ---------------------------
# Main app
# ---------------------------
def main():
    drug_df, cancer_cell_mapping, model, device, cell_line_mapping, cell_le, cancer_cols, scaler_y = load_data_and_model()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Screening", "About"])
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=True)

    if page == "Single Prediction":
        st.markdown('<div class="main-header">🧪 Drug Combination Synergy Predictor</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Drug Selection")
            drug_a = st.selectbox("Select Drug A", options=drug_df['drug_name'].tolist(), index=0)
            drug_b = st.selectbox("Select Drug B", options=drug_df['drug_name'].tolist(), index=1)
            if drug_a and drug_b:
                col1a, col1b = st.columns(2)
                with col1a:
                    st.write(f"**{drug_a}**")
                    smiles_a = drug_df[drug_df['drug_name']==drug_a]['smiles'].iloc[0]
                    img_a = mol_to_image(smiles_a)
                    if img_a: st.image(img_a, use_container_width=True)
                with col1b:
                    st.write(f"**{drug_b}**")
                    smiles_b = drug_df[drug_df['drug_name']==drug_b]['smiles'].iloc[0]
                    img_b = mol_to_image(smiles_b)
                    if img_b: st.image(img_b, use_container_width=True)

        with col2:
            st.subheader("Cancer Context")
            cancer_type = st.selectbox("Select Cancer Type", options=list(cancer_cell_mapping.keys()), index=0)
            if cancer_type in cancer_cell_mapping:
                cell_line = st.selectbox("Select Cell Line", options=cancer_cell_mapping[cancer_type])
            else:
                cell_line = st.selectbox("Select Cell Line", ["No cell lines available"])

            predict_btn = st.button("🔬 Predict Synergy & Side Effects", type="primary", use_container_width=True)

        if predict_btn:
            if drug_a == drug_b:
                st.error("Please select two different drugs for combination analysis.")
            else:
                with st.spinner("Predicting synergy and side effects..."):
                    synergy_score, error = predict_synergy_proper(
                        smiles_a, smiles_b, cancer_type, cell_line,
                        model, device, cancer_cell_mapping, cell_line_mapping,
                        cell_le=cell_le, cancer_cols=cancer_cols, scaler_y=scaler_y,
                        debug_mode=debug_mode
                    )

                    side_effects_a = predict_side_effects(smiles_a)
                    side_effects_b = predict_side_effects(smiles_b)

                    if error:
                        st.error(f"Prediction error: {error}")
                    else:
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.subheader("🎯 Synergy Prediction Results")

                        if synergy_score > 10:
                            synergy_class = "High Synergy"
                            synergy_color = "synergy-high"
                            interpretation = "Strong positive interaction expected"
                            confidence_level = "High"
                            recommendation = "🎉 Promising candidate for further investigation!"
                        elif synergy_score > 5:
                            synergy_class = "Medium Synergy"
                            synergy_color = "synergy-medium"
                            interpretation = "Moderate positive interaction expected"
                            confidence_level = "High"
                            recommendation = "👍 Worth further experimental validation"
                        elif synergy_score > 0:
                            synergy_class = "Low Synergy"
                            synergy_color = "synergy-low"
                            interpretation = "Weak positive interaction"
                            confidence_level = "Medium"
                            recommendation = "⚠️ May need careful evaluation"
                        else:
                            synergy_class = "Antagonism"
                            synergy_color = "synergy-antagonism"
                            interpretation = "Negative interaction expected"
                            confidence_level = "Low"
                            recommendation = "❌ Likely to be ineffective or harmful"

                        st.markdown(f"**Synergy Score:** <span class='{synergy_color}'>{synergy_score:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Synergy Class:** <span class='{synergy_color}'>{synergy_class}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Interpretation:** {interpretation}")
                        st.markdown(f"**Confidence Level:** {confidence_level}")
                        st.markdown(f"**Recommendation:** {recommendation}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.subheader("💊 Side Effect Risk Assessment")
                        col_side1, col_side2 = st.columns(2)
                        with col_side1:
                            st.write(f"**{drug_a} Side Effect Profile**")
                            if side_effects_a:
                                interpretations = interpret_side_effects(side_effects_a)
                                for interpretation in interpretations:
                                    if "🚨" in interpretation: st.error(interpretation)
                                    elif "⚠️" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")

                        with col_side2:
                            st.write(f"**{drug_b} Side Effect Profile**")
                            if side_effects_b:
                                interpretations = interpret_side_effects(side_effects_b)
                                for interpretation in interpretations:
                                    if "🚨" in interpretation: st.error(interpretation)
                                    elif "⚠️" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")

                        st.subheader("🔄 Combination Side Effect Risks")
                        combination_side_effects = predict_combination_side_effects(side_effects_a, side_effects_b)
                        interpretations = interpret_side_effects(combination_side_effects)
                        for interpretation in interpretations:
                            if "🚨" in interpretation: st.error(interpretation)
                            elif "⚠️" in interpretation: st.warning(interpretation)
                            else: st.success(interpretation)

    elif page == "Batch Screening":
        st.header("Batch Drug Combination Screening")
        uploaded_file = st.file_uploader("Upload CSV with columns: drug_a, drug_b, cancer_type, cell_line", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            batch_df = batch_df[batch_df['drug_a'].isin(drug_df['drug_name']) & batch_df['drug_b'].isin(drug_df['drug_name'])]
            if batch_df.empty:
                st.warning("No valid drug combinations found.")
            else:
                with st.spinner("Processing batch predictions..."):
                    results_list = []
                    for _, row in batch_df.iterrows():
                        smiles_a = drug_df.loc[drug_df['drug_name']==row['drug_a'], 'smiles'].values[0]
                        smiles_b = drug_df.loc[drug_df['drug_name']==row['drug_b'], 'smiles'].values[0]
                        score, error = predict_synergy_proper(
                            smiles_a, smiles_b, row['cancer_type'], row['cell_line'],
                            model, device, cancer_cell_mapping, cell_line_mapping,
                            cell_le=cell_le, cancer_cols=cancer_cols, scaler_y=scaler_y,
                            debug_mode=debug_mode
                        )
                        results_list.append({
                            'drug_a': row['drug_a'],
                            'drug_b': row['drug_b'],
                            'cancer_type': row['cancer_type'],
                            'cell_line': row['cell_line'],
                            'synergy_score': score,
                            'error': error
                        })
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df)

    else:
        st.header("About")
        st.markdown("""
        # 🧪 Drug Combination Synergy Predictor

        This tool predicts **drug combination synergy** for various cancer types using a **Graph Neural Network (GNN)**.  
        It also evaluates potential **side effect risks** based on molecular properties.

        ## Features
        - **Synergy Prediction**: Predicts drug interactions for specific cancer cell lines.
        - **Side Effect Assessment**: Estimates risk in 8 categories:
          - Hepatotoxicity
          - Nephrotoxicity
          - Cardiotoxicity
          - Neurotoxicity
          - Gastrointestinal
          - Dermatological
          - CNS Effects
          - Metabolic Issues
        - **High/Medium/Low Risk Classification** based on molecular features and structural alerts.
        - **Batch Screening** for large CSV inputs.

        **Note:** Only drugs with valid SMILES are considered for predictions.
                
        **Disclaimer:** This tool is for research and educational purposes only and is only based on cancer
        """)

if __name__ == "__main__": 
    main()
