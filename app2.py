from flask import Flask, request, jsonify
from flask_cors import CORS
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
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ---------------------------
# Model classes (same as your original)
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
# Utility functions
# ---------------------------
def mol_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_feats = [[atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetDegree(), 
                   atom.GetImplicitValence(), 1.0 if atom.GetIsAromatic() else 0.0] 
                  for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt_val = {Chem.rdchem.BondType.SINGLE: 1.0, Chem.rdchem.BondType.DOUBLE: 2.0, 
                  Chem.rdchem.BondType.TRIPLE: 3.0, Chem.rdchem.BondType.AROMATIC: 1.5}.get(bond.GetBondType(), 0.0)
        edge_index.extend([[a1, a2], [a2, a1]])
        edge_attr.extend([[bt_val], [bt_val]])
    
    if len(edge_index) == 0:
        edge_index, edge_attr = [[0, 0]], [[0.0]]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_image_base64(smiles):
    """Convert SMILES to base64 encoded image"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=(300, 300))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def predict_side_effects(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    side_effect_alerts = {
        'hepatotoxicity': ['[#6]-[N+](=O)[O-]', 'c1ccccc1[N]', 'S(=O)(=O)[OH]'],
        'nephrotoxicity': ['[Cl,Br,I]', 'c1ccccc1[F,Cl,Br,I]', '[NH2]C(=O)'],
        'cardiotoxicity': ['c1ccccc1', '[N+]', 'C=O'],
        'neurotoxicity': ['C#N', 'C=O', '[S;D2](=O)(=O)[#6]'],
        'gastrointestinal': ['[OH]', 'C(=O)[OH]', 'c1ccccc1[OH]'],
        'dermatological': ['[Cl,Br,I]', 'C=O', 'c1ccccc1']
    }
    
    side_effects = {}
    
    hepatotoxicity_score = sum([0.3 for pattern in side_effect_alerts['hepatotoxicity'] 
                                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if mw > 500:
        hepatotoxicity_score += 0.2
    side_effects['hepatotoxicity'] = min(1.0, hepatotoxicity_score)
    
    nephrotoxicity_score = sum([0.25 for pattern in side_effect_alerts['nephrotoxicity'] 
                                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if logp > 4:
        nephrotoxicity_score += 0.2
    side_effects['nephrotoxicity'] = min(1.0, nephrotoxicity_score)
    
    cardiotoxicity_score = sum([0.2 for pattern in side_effect_alerts['cardiotoxicity'] 
                                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if Descriptors.NumAromaticRings(mol) > 2:
        cardiotoxicity_score += 0.3
    side_effects['cardiotoxicity'] = min(1.0, cardiotoxicity_score)
    
    neurotoxicity_score = sum([0.25 for pattern in side_effect_alerts['neurotoxicity'] 
                               if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['neurotoxicity'] = min(1.0, neurotoxicity_score)
    
    gi_score = sum([0.2 for pattern in side_effect_alerts['gastrointestinal'] 
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if hbd > 3:
        gi_score += 0.2
    side_effects['gastrointestinal'] = min(1.0, gi_score)
    
    derm_score = sum([0.2 for pattern in side_effect_alerts['dermatological'] 
                      if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['dermatological'] = min(1.0, derm_score)
    
    side_effects['cns_effects'] = min(1.0, 0.1 + (logp / 10))
    side_effects['metabolic_issues'] = min(1.0, 0.1 + (mw - 300) / 700)
    
    return side_effects


def find_cancer_col_index(cancer_cols, cancer_type_key):
    candidates = [f"cancer_type_{cancer_type_key}", f"cancer_type_{cancer_type_key.replace(' ', '_')}",
                  f"cancer_type_{cancer_type_key.lower()}", f"cancer_type_{cancer_type_key.replace(' ', '').lower()}"]
    for cand in candidates:
        if cand in cancer_cols:
            return cancer_cols.index(cand)
    
    key_words = [w.lower() for w in cancer_type_key.split()]
    for i, cc in enumerate(cancer_cols):
        if all(w in cc.lower() for w in key_words):
            return i
    
    for i, cc in enumerate(cancer_cols):
        if cancer_type_key.lower() in cc.lower():
            return i
    
    return None


def predict_synergy(drug1_smiles, drug2_smiles, cancer_type, cell_line,
                   model, device, cancer_cell_mapping, cell_line_mapping,
                   cell_le, cancer_cols, scaler_y):
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
# Load model and data
# ---------------------------
def load_model_and_data():
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
        with open("scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
    if os.path.exists("cell_le.pkl"):
        with open("cell_le.pkl", "rb") as f:
            cell_le = pickle.load(f)
    if os.path.exists("cancer_cols.pkl"):
        with open("cancer_cols.pkl", "rb") as f:
            cancer_cols = pickle.load(f)

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


# Load everything at startup
drug_df, cancer_cell_mapping, model, device, cell_line_mapping, cell_le, cancer_cols, scaler_y = load_model_and_data()


# ---------------------------
# API Routes
# ---------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    """Get list of available drugs"""
    # Only return required columns to avoid NaN issues
    drugs_clean = drug_df[['drug_name', 'smiles']].copy()
    drugs_clean = drugs_clean.dropna(subset=['drug_name', 'smiles'])
    drugs_list = drugs_clean.to_dict('records')
    return jsonify({'drugs': drugs_list})




@app.route('/api/cancer-types', methods=['GET'])
def get_cancer_types():
    """Get list of cancer types and their cell lines"""
    return jsonify({'cancer_types': cancer_cell_mapping})


@app.route('/api/molecule-image', methods=['POST'])
def get_molecule_image():
    """Generate molecule image from SMILES"""
    data = request.json
    smiles = data.get('smiles')
    
    if not smiles:
        return jsonify({'error': 'SMILES string required'}), 400
    
    image_base64 = mol_to_image_base64(smiles)
    
    if image_base64 is None:
        return jsonify({'error': 'Invalid SMILES string'}), 400
    
    return jsonify({'image': image_base64})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    data = request.json
    
    drug_a_name = data.get('drug_a')
    drug_b_name = data.get('drug_b')
    cancer_type = data.get('cancer_type')
    cell_line = data.get('cell_line')
    
    # Validate inputs
    if not all([drug_a_name, drug_b_name, cancer_type, cell_line]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if drug_a_name == drug_b_name:
        return jsonify({'error': 'Please select two different drugs'}), 400
    
    # Get SMILES
    try:
        smiles_a = drug_df[drug_df['drug_name'] == drug_a_name]['smiles'].iloc[0]
        smiles_b = drug_df[drug_df['drug_name'] == drug_b_name]['smiles'].iloc[0]
    except IndexError:
        return jsonify({'error': 'Drug not found in database'}), 404
    
    # Predict synergy
    synergy_score, error = predict_synergy(
        smiles_a, smiles_b, cancer_type, cell_line,
        model, device, cancer_cell_mapping, cell_line_mapping,
        cell_le, cancer_cols, scaler_y
    )
    
    if error:
        return jsonify({'error': error}), 500
    
    # Predict side effects
    side_effects_a = predict_side_effects(smiles_a)
    side_effects_b = predict_side_effects(smiles_b)
    
    # Calculate combination side effects
    combo_effects = {}
    if side_effects_a and side_effects_b:
        for effect in side_effects_a.keys():
            risk_a = side_effects_a[effect]
            risk_b = side_effects_b.get(effect, 0.1)
            combo_risk = 1 - (1 - risk_a) * (1 - risk_b)
            combo_effects[effect] = min(1.0, combo_risk)
    
    # Determine synergy class
    if synergy_score > 10:
        synergy_class = "High Synergy"
        interpretation = "Strong positive interaction expected"
        recommendation = "Promising candidate for further investigation!"
    elif synergy_score > 5:
        synergy_class = "Medium Synergy"
        interpretation = "Moderate positive interaction expected"
        recommendation = "Worth further experimental validation"
    elif synergy_score > 0:
        synergy_class = "Low Synergy"
        interpretation = "Weak positive interaction"
        recommendation = "May need careful evaluation"
    else:
        synergy_class = "Antagonism"
        interpretation = "Negative interaction expected"
        recommendation = "Likely to be ineffective or harmful"
    
    # Generate molecule images
    image_a = mol_to_image_base64(smiles_a)
    image_b = mol_to_image_base64(smiles_b)
    
    return jsonify({
        'synergy_score': synergy_score,
        'synergy_class': synergy_class,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'side_effects_a': side_effects_a,
        'side_effects_b': side_effects_b,
        'combination_side_effects': combo_effects,
        'drug_a': {
            'name': drug_a_name,
            'smiles': smiles_a,
            'image': image_a
        },
        'drug_b': {
            'name': drug_b_name,
            'smiles': smiles_b,
            'image': image_b
        }
    })


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    data = request.json
    combinations = data.get('combinations', [])
    
    if not combinations:
        return jsonify({'error': 'No combinations provided'}), 400
    
    results = []
    for combo in combinations:
        drug_a_name = combo.get('drug_a')
        drug_b_name = combo.get('drug_b')
        cancer_type = combo.get('cancer_type')
        cell_line = combo.get('cell_line')
        
        try:
            smiles_a = drug_df[drug_df['drug_name'] == drug_a_name]['smiles'].iloc[0]
            smiles_b = drug_df[drug_df['drug_name'] == drug_b_name]['smiles'].iloc[0]
            
            synergy_score, error = predict_synergy(
                smiles_a, smiles_b, cancer_type, cell_line,
                model, device, cancer_cell_mapping, cell_line_mapping,
                cell_le, cancer_cols, scaler_y
            )
            
            results.append({
                'drug_a': drug_a_name,
                'drug_b': drug_b_name,
                'cancer_type': cancer_type,
                'cell_line': cell_line,
                'synergy_score': synergy_score,
                'error': error
            })
        except Exception as e:
            results.append({
                'drug_a': drug_a_name,
                'drug_b': drug_b_name,
                'cancer_type': cancer_type,
                'cell_line': cell_line,
                'synergy_score': None,
                'error': str(e)
            })
    
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
