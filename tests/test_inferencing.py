import pytest
import joblib
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
from rdkit.Chem.GraphDescriptors import BertzCT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test data
TEST_SMILES = [
    'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F',  # Example compound
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CC(=O)NC1=CC=C(O)C=C1'      # Paracetamol
]

# Copy the essential functions here to avoid Streamlit initialization
def compute_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")
    
    # Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fingerprint_array = np.array(fingerprint, dtype=int)
    
    # Molecular properties
    features = {
        "num_atoms": mol.GetNumAtoms(),
        "logP": Descriptors.MolLogP(mol),
        "molecular_weight": Descriptors.ExactMolWt(mol),
        "polar_surface_area": Descriptors.TPSA(mol),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "ring_count": Descriptors.RingCount(mol),
        "aromatic_rings": sum(
            1 for ring in Chem.GetSymmSSSR(mol) 
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        ),
        "logS": -Descriptors.MolLogP(mol),
        "electrostatic_potential": sum(
            a.GetDoubleProp("_GasteigerCharge")
            for a in AllChem.ComputeGasteigerCharges(mol) or mol.GetAtoms()
        ),
        "shape_index": BertzCT(mol),
        "surface_area": CalcLabuteASA(mol)
    }
    
    return np.hstack((fingerprint_array, list(features.values())))

# Load the model
MODEL_PATH = "Artifacts/model.pkl"

loaded_model = joblib.load(MODEL_PATH)

def predict_pIC50_test(smiles: str):
    """Test version of predict_pIC50 without Streamlit dependencies"""
    try:
        features = compute_features(smiles)
        features = features.reshape(1, -1)
        predicted_pIC50 = loaded_model.predict(features)
        return float(predicted_pIC50[0])
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None

def run_predictions():
    """Function to run predictions without Streamlit"""
    print("Running predictions...")
    
    for smiles in TEST_SMILES:
        # Get prediction
        result = predict_pIC50_test(smiles)
        
        if result is not None:
            # Basic validation
            assert isinstance(result, float)
            assert 0 <= result <= 12  # Typical pIC50 range
            

# For pytest usage
def test_predictions():
    """Pytest function for running predictions"""
    for smiles in TEST_SMILES:
        result = predict_pIC50_test(smiles)
        assert result is not None
        assert isinstance(result, float)
        assert 0 <= result <= 12

