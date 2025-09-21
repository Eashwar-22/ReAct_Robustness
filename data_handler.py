import pandas as pd
import random
import json
import os
import string

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a JSON file and normalizes it into a pandas DataFrame."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        print(f"Successfully loaded '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()

def make_noisy_data(df: pd.DataFrame, noise_type: str = 'shuffled_headers', column_name: str = '', value: any = None) -> pd.DataFrame:
    """Simulates different noise types on the DataFrame."""
    noisy_df = df.copy()

    if noise_type == "shuffled_headers":
        columns = noisy_df.columns.tolist()
        random.shuffle(columns)
        noisy_df = noisy_df[columns]
        print("Noisy data created: headers shuffled.")
    
    elif noise_type == "misspelled_headers":
        if column_name and column_name in noisy_df.columns:
            # Create a simple misspelling by adding or removing a letter
            new_name = column_name + 'e' if random.random() > 0.5 else column_name[:-1]
            noisy_df.rename(columns={column_name: new_name}, inplace=True)
            print(f"Noisy data created: '{column_name}' column misspelled to '{new_name}'.")
    
    elif noise_type == "typos_in_content":
        if column_name and column_name in noisy_df.columns and noisy_df[column_name].dtype == 'object':
            # Introduce typos in a categorical column
            noisy_df[column_name] = noisy_df[column_name].apply(
                lambda x: str(x) + 'a' if random.random() > 0.5 and isinstance(x, str) else x
            )
            print(f"Noisy data created: typos introduced in '{column_name}' column.")

    elif noise_type == "mixed_data_types":
        if column_name and column_name in noisy_df.columns and pd.api.types.is_numeric_dtype(noisy_df[column_name]):
            # Introduce a non-numeric value in a numerical column
            noisy_df.loc[0, column_name] = 'N/A'
            print(f"Noisy data created: '{column_name}' column now contains a string.")
            
    elif noise_type == "missing_values":
        if column_name and column_name in noisy_df.columns:
            # Introduce missing values in a specific column
            noisy_df.loc[noisy_df.sample(frac=0.1).index, column_name] = pd.NA
            print(f"Noisy data created: 10% missing values introduced in '{column_name}'.")

    return noisy_df

# This block can be run to test the functions independently.
if __name__ == '__main__':
    for data in ['data/electric_vehicles_dataset.json', 'data/heart_disease_dataset.json', 'data/iris.json']:
        clean_df = load_data(data)
        print("Original columns : ", clean_df.columns.tolist())
        random_column_name = random.choice(clean_df.columns.tolist()) if not clean_df.empty else ''
        if not clean_df.empty:
            # example
            noisy_df_misspelled = make_noisy_data(clean_df.copy(), "misspelled_headers", column_name=random_column_name)
            noisy_df_typos = make_noisy_data(clean_df.copy(), "typos_in_content", column_name=random_column_name)
            noisy_df_shuffled = make_noisy_data(clean_df.copy(), "shuffled_headers", column_name=random_column_name)
            print("\nShuffled columns: ", noisy_df_shuffled.columns.tolist())
    
            noisy_df_mixed = make_noisy_data(clean_df.copy(), "mixed_data_types", column_name=random_column_name)
            noisy_df_missing = make_noisy_data(clean_df.copy(), "missing_values", column_name=random_column_name)
            print("\nData handling functions are working.")
