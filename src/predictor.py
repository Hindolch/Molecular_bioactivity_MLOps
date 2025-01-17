from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem import AllChem
import numpy as np
import joblib
import py3Dmol


loaded_model = joblib.load("/home/kenzi/molecular prediction mlops/Artifacts/model.pkl")
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
        "surface_area": CalcLabuteASA(mol),
    }
    
    return np.hstack((fingerprint_array, list(features.values())))
def predict_pIC50(smiles: str):
    try:
        # Compute features
        features = compute_features(smiles)
        # Reshape for prediction (single sample)
        features = features.reshape(1, -1)
        # Predict using the loaded model
        predicted_pIC50 = loaded_model.predict(features)
        return predicted_pIC50[0]
    except ValueError as e:
        return str(e)
    
def generate_3d_structure(smiles: str):
    """Generate a 3D structure for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)  # Add hydrogens
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates
        return mol
    except Exception as e:
        raise ValueError(f"Error generating 3D structure: {e}")

def visualize_3d_structure(smiles: str):
    """Visualize a 3D structure for a given SMILES string."""
    try:
        mol = generate_3d_structure(smiles)
        mol_block = Chem.MolToMolBlock(mol)  # Convert to MolBlock format
        viewer = py3Dmol.view(width=500, height=500)
        viewer.addModel(mol_block, "mol")  # Add the molecule to the viewer
        viewer.setStyle({"stick": {}})  # Set stick style for visualization
        viewer.setBackgroundColor("black")  # Set background color
        viewer.zoomTo()  # Adjust the zoom
        return viewer
    except Exception as e:
        return str(e)

smiles_input = input("Enter a SMILES string")


try:
# Show prediction
    result = predict_pIC50(smiles_input)
    if result > 6:
        print("This molecule has high affinity, ideal for binding")
    elif 4 <= result <= 6:
        print("This molecule has moderate affinity, acceptable for binding but not ideal!")
    else:
        print("This molecule has low affinity")
    print(f"Predicted pIC50: {result}")

            # Visualize the molecule
    print("Rendering 3D structure...")
    viewer = visualize_3d_structure(smiles_input)
    viewer.show()  # Display in a web browser
except Exception as e:
    print(f"Error: {e}")