import pandas as pd
import logging
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
import numpy as np

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DFMolecularPropertyStrategy(ABC):
    @abstractmethod
    def extract_properties(self, df1: pd.DataFrame) -> pd.DataFrame:
        pass



def create_mol_from_smiles(smiles):
    """Safe molecule creation from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception as e:
        logging.warning(f"Error creating molecule from SMILES {smiles}: {str(e)}")
        return None

def safe_mol_to_smiles(mol):
    """Safe SMILES generation from molecule"""
    try:
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logging.warning(f"Error converting molecule to SMILES: {str(e)}")
        return None

class DFProperties(DFMolecularPropertyStrategy):
    def extract_properties(self, df1: pd.DataFrame):
        logging.info("Extracting properties for DF1")
        df1 = df1.copy()
            
        # Create molecules and filter invalid ones
        df1['Mol'] = df1['SMILES'].apply(create_mol_from_smiles)
        df1['smiles'] = df1['Mol'].apply(safe_mol_to_smiles)
        
        # Remove rows with invalid molecules
        invalid_mols = df1['Mol'].isna()
        if invalid_mols.any():
            logging.warning(f"Removing {invalid_mols.sum()} rows with invalid molecules from DF1")
            df1 = df1[~invalid_mols].reset_index(drop=True)
        
        df1 = df1.drop(columns=['SMILES'])

        def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
            """Compute Morgan fingerprint as a NumPy array"""
            if mol is None:
                return np.zeros(n_bits, dtype=int)  # Return a zero vector if molecule is None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp) 
               
        # Safe property calculation function
        def safe_calc_property(mol, calc_function, default_value=None):
            try:
                if mol is None:
                    return default_value
                return calc_function(mol)
            except Exception as e:
                logging.warning(f"Error calculating property: {str(e)}")
                return default_value
        
        
        # Calculate properties safely
        df1['molecular_weight'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.ExactMolWt))
        df1['polar_surface_area'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.TPSA))
        df1['h_bond_donors'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.NumHDonors))
        df1['h_bond_acceptors'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.NumHAcceptors))
        df1['rotatable_bonds'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.NumRotatableBonds))
        df1['ring_count'] = df1['Mol'].apply(lambda x: safe_calc_property(x, Descriptors.RingCount))
        df1['aromatic_rings'] = df1['Mol'].apply(lambda mol: safe_calc_property(mol, lambda x: sum(1 for ring in Chem.GetSymmSSSR(x) if all(x.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))))
        df1['logS'] = df1['Mol'].apply(lambda x: safe_calc_property(x, lambda m: -Descriptors.MolLogP(m)))
        df1['electrostatic_potential'] = df1['Mol'].apply(lambda x: sum([a.GetDoubleProp('_GasteigerCharge')for a in AllChem.ComputeGasteigerCharges(x) or x.GetAtoms()]))
        df1['shape_index'] = df1['Mol'].apply(lambda x: safe_calc_property(x, BertzCT))
        df1['surface_area'] = df1['Mol'].apply(lambda x: safe_calc_property(x, CalcLabuteASA))
        df1['morgan_fingerprint'] = df1['Mol'].apply(lambda x: compute_morgan_fingerprint(x))
        df1 = df1.drop(columns=['Unnamed: 0'])
        return df1



class MolecularFeatureEngineer:
    def __init__(self, df_strategy: DFMolecularPropertyStrategy = None):
        self._df_strategy = df_strategy

    def set_df1_strategy(self, strategy: DFMolecularPropertyStrategy):
        self._df_strategy = strategy
    
    def execute_strategy(self, df:pd.DataFrame)->pd.DataFrame:
        if self._df_strategy is None:
            raise ValueError("No strategy set for molecular property extraction.")
        return self._df_strategy.extract_properties(df)

