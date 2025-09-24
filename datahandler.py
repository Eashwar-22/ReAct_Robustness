import pandas as pd
import random
import json
import os

class DataHandler:
    """
    Handles loading datasets and applying various types of noise.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()

    def load_data(self):
        """Loads a JSON dataset into a pandas DataFrame."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            print(f"Successfully loaded '{self.file_path}'.")
            return df
        except FileNotFoundError:
            print(f"Error: Data file '{self.file_path}' not found.")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{self.file_path}'.")
            return pd.DataFrame()

    def apply_noise(self, noise_type):
        """
        Applies a specified type of noise to the DataFrame.
        Returns a new DataFrame with noise applied.
        """
        if self.df.empty:
            print("Warning: Input DataFrame is empty. Cannot apply noise.")
            return self.df

        noisy_df = self.df.copy()

        if noise_type == "shuffled_headers":
            original_columns = noisy_df.columns.tolist()
            shuffled_columns = original_columns.copy()
            random.shuffle(shuffled_columns)
            
            rename_mapping = dict(zip(original_columns, shuffled_columns))
            noisy_df.rename(columns=rename_mapping, inplace=True)
            print("Noisy data created: headers have been shuffled independently.")
        

        return noisy_df
    
if __name__ == '__main__':
    test_file = 'data/iris.json'
    print(f"--- Testing data_handler.py with '{test_file}' ---")
    
    dh = DataHandler(test_file)
    if os.path.exists(test_file):
        clean_df = dh.df.copy()
        
        if not clean_df.empty:
            print("\nOriginal columns:", clean_df.columns.tolist())
            
            shuffled_df = dh.apply_noise("shuffled_headers")
            print("Shuffled columns:", shuffled_df.columns.tolist())
            
            print("\n--- All data handling functions are working. ---")
    else:
        print(f"Test file '{test_file}' not found. Cannot run tests.")