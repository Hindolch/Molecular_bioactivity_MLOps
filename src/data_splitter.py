import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df:pd.DataFrame, target_column:str):
        pass

# concrete strategy for train-test split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df:pd.DataFrame, target_column:str):
        # Stack fingerprints into a matrix and combine with other features
        fingerprint_matrix = np.vstack(df['morgan_fingerprint'])
        other_features = df[['num_atoms', 'logP', 'molecular_weight', 
                               'polar_surface_area', 'h_bond_donors', 
                               'h_bond_acceptors', 'rotatable_bonds', 
                               'ring_count', 'aromatic_rings', 'logS', 
                               'electrostatic_potential', 'shape_index', 
                               'surface_area']].values

        # Final feature matrix
        X = np.hstack((fingerprint_matrix, other_features))
        y = df[target_column].values
        X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=self.test_size, random_state=self.random_state)
        return X_train,X_test, y_train, y_test
    

#Context class for data splitting
class DataSplitter:
    def __init__(self, strategy:DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy:DataSplittingStrategy):
        self._strategy = strategy

    def split(self, df:pd.DataFrame, target_column:str):
        return self._strategy.split_data(df,target_column)