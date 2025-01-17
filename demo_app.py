import streamlit as st
import py3Dmol
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
from rdkit.Chem.GraphDescriptors import BertzCT
import numpy as np
import joblib
import google.generativeai as genai
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

GEMINI_API_KEY = "AIzaSyC4b_PXAIu62ePfdivmOUx5MUCdL0eNcFw"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load pre-trained model
loaded_model = joblib.load("/home/kenzi/molecular prediction mlops/Artifacts/model.pkl")

# Function to convert molecule name to SMILES using Gemini API
def molecule_to_smiles(name: str):
    prompt = f"Convert the molecular name '{name}' into its SMILES representation. And just give me SMILES string output no need to write anything extra!"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to compute features from SMILES string
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
        features = compute_features(smiles)
        features = features.reshape(1, -1)
        predicted_pIC50 = loaded_model.predict(features)
        return predicted_pIC50[0]
    except ValueError as e:
        return str(e)

# Function to draw 2D structure of the molecule
def draw_2d_structure(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        return img
    else:
        return None

# Streamlit interface
st.set_page_config(page_title="Molecular Visualization", layout="centered")

st.title("Drug/bioactive Molecular Effectiveness Prediction")

# Add description and example
st.markdown("""
Enter a SMILES string or molecule name (preferably bioactive molecules) to predict its binding affinity (pIC50).

**Importance of Binding Affinity**  
Binding affinity is crucial in drug discovery as it determines how effectively a drug binds to its target. High binding affinity often correlates with greater therapeutic efficacy and fewer side effects, making it a key factor in designing effective and safe medications.  
(Note: For best results, input bioactive molecules that are known to interact with biological targets.)
""")


# Add example with copy button
example_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F"
st.code(example_smiles, language="text")
if st.button("Use Example SMILES"):
    st.session_state.smiles_input = example_smiles

# SMILES or molecule name input
smiles_input = st.text_input("Enter a SMILES string or molecule name:", 
                            key="smiles_input",
                            help="Enter a valid SMILES string or molecule name to analyze")

if smiles_input:
    try:
        # First, check if the input is a molecule name or a SMILES string
        if not Chem.MolFromSmiles(smiles_input):  # If input isn't a valid SMILES string
            st.info("Attempting to convert molecule name to SMILES...")
            smiles_input = molecule_to_smiles(smiles_input)
            if "Error" in smiles_input:
                st.error(f"Failed to convert name to SMILES: {smiles_input}")
                raise ValueError("Invalid molecule name")
            else:
                st.info(f"Converted molecule name to SMILES: {smiles_input}")
        
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Results")
            
            # Calculate and display prediction
            result = predict_pIC50(smiles_input)
            st.metric("Predicted pIC50", f"{result:.2f}")
            
            # Display binding interpretation
            if result > 6:
                st.success("🎯 High affinity - Ideal for binding")
            elif 4 <= result <= 6:
                st.warning("👌 Moderate affinity - Acceptable for binding")
            else:
                st.error("⚠️ Low affinity - Poor binding potential")
        
        with col2:
            st.subheader(f"Molecule 2D Structure of the given molecule")
            img = draw_2d_structure(smiles_input)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.error("Failed to generate 2D structure.")


        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check if the SMILES string or molecule name is valid and try again.")
