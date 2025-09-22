import pandas as pd
import numpy as np
import random
import json
import os
import string

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a JSON file and normalizes it into a pandas DataFrame.
    
    Args:
        file_path: The path to the JSON file.
        
    Returns:
        A pandas DataFrame with the loaded data, or an empty DataFrame on error.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        print(f"Successfully loaded '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        return pd.DataFrame()

def make_noisy_data(df: pd.DataFrame, noise_type: str, column_name: str = None) -> pd.DataFrame:
    """
    Applies a specified type of noise to a pandas DataFrame.
    
    Args:
        df: The input pandas DataFrame.
        noise_type: The type of noise to apply (e.g., 'shuffled_headers').
        column_name: The target column for noise types that require it.
        
    Returns:
        A new DataFrame with the applied noise.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Cannot apply noise.")
        return df

    noisy_df = df.copy()

    if noise_type == "shuffled_headers":
        original_columns = noisy_df.columns.tolist()
        shuffled_columns = original_columns.copy()
        random.shuffle(shuffled_columns)
        
        rename_mapping = dict(zip(original_columns, shuffled_columns))
        noisy_df.rename(columns=rename_mapping, inplace=True)
        print("Noisy data created: headers have been shuffled independently.")
    
    elif noise_type == "misspelled_headers":
        if not column_name or column_name not in noisy_df.columns:
            # If no valid column is provided, pick a random one
            column_name = random.choice(noisy_df.columns.tolist())
        
        # Create a simple misspelling
        new_name = column_name + random.choice(string.ascii_lowercase)
        noisy_df.rename(columns={column_name: new_name}, inplace=True)
        print(f"Noisy data created: header '{column_name}' misspelled to '{new_name}'.")
    
    elif noise_type == "typos_in_content":
        if not column_name or column_name not in noisy_df.columns:
            column_name = random.choice(noisy_df.columns.tolist())

        # Introduce typos in 10% of a column's string values
        def add_typo(s):
            if isinstance(s, str) and len(s) > 1 and random.random() < 0.1:
                pos = random.randint(0, len(s) - 1)
                return s[:pos] + random.choice(string.ascii_lowercase) + s[pos+1:]
            return s
        noisy_df[column_name] = noisy_df[column_name].apply(add_typo)
        print(f"Noisy data created: typos introduced in '{column_name}' column.")

    elif noise_type == "mixed_data_types":
        # Target a numeric column for introducing a string
        numeric_cols = noisy_df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            print("Warning: No numeric columns found for 'mixed_data_types' noise.")
            return noisy_df
        
        column_to_disrupt = random.choice(numeric_cols)
        # Introduce a non-numeric value in a random row
        rand_idx = random.choice(noisy_df.index)
        noisy_df.loc[rand_idx, column_to_disrupt] = "error"
        print(f"Noisy data created: string value introduced in numeric column '{column_to_disrupt}'.")
            
    elif noise_type == "missing_values":
        if not column_name or column_name not in noisy_df.columns:
            column_name = random.choice(noisy_df.columns.tolist())
        
        # Introduce missing values in 10% of a column
        indices_to_null = noisy_df.sample(frac=0.1).index
        noisy_df.loc[indices_to_null, column_name] = np.nan
        print(f"Noisy data created: 10% missing values in '{column_name}'.")
    
    else:
        print(f"Warning: Noise type '{noise_type}' not recognized. Returning original DataFrame.")

    return noisy_df

# This block allows for independent testing of the script's functions.
if __name__ == '__main__':
    test_file = 'data/iris.json'
    print(f"--- Testing data_handler.py with '{test_file}' ---")
    
    if os.path.exists(test_file):
        clean_df = load_data(test_file)
        
        if not clean_df.empty:
            print("\nOriginal columns:", clean_df.columns.tolist())
            
            # Test each noise type
            shuffled_df = make_noisy_data(clean_df, "shuffled_headers")
            print("Shuffled columns:", shuffled_df.columns.tolist())
            
            misspelled_df = make_noisy_data(clean_df, "misspelled_headers")
            print("Misspelled columns:", misspelled_df.columns.tolist())

            print("\n--- All data handling functions are working. ---")
    else:
        print(f"Test file '{test_file}' not found. Cannot run tests.")
