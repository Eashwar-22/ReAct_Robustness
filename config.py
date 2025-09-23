import os

# --- Configuration for Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
# LLM_MODEL = os.getenv("LLM_MODEL", "qwen")

# --- Data and Output Paths ---
DATA_DIR = 'data'
ANSWERKEY_DIR = 'answer_key'
OUTPUTS_FOLDER = 'outputs'

DATA_FILE_PATH1 = os.path.join(DATA_DIR, 'electric_vehicles_dataset.json')
DATA_FILE_PATH2 = os.path.join(DATA_DIR, 'heart_disease_dataset.json')
DATA_FILE_PATH3 = os.path.join(DATA_DIR, 'iris.json')

ANSWER_KEY_PATH = os.path.join(DATA_DIR,ANSWERKEY_DIR, 'answer_key.json')

TASKS_BY_DATASET = {
    'electric_vehicles': [
        "Calculate the mean and standard deviation of the 'Battery_Capacity_kWh' column.",
        "Find the number of vehicles for each unique 'Country_of_Manufacture'.",
        "What is the correlation between the 'Range_km' and 'Charge_Time_hr'?",
        "What is the average of the column that contains the vehicles' prices?"
    ],
    'heart_disease': [
        "What is the median `age` and `chol` (cholesterol) of the patients)?",
        "Count how many patients are marked with `heart_disease` = 1 versus `heart_disease` = 0.",
        "Find the Pearson correlation between `age` and `trestbps` (resting blood pressure).",
        "What is the mean of the column representing the maximum heart rate achieved?"
    ],
    'iris': [
        "Calculate the mean of `sepalLength` and `sepalWidth`.",
        "List the unique `species` present in the dataset.",
        "What is the correlation between `petalLength` and `petalWidth`?",
        "Find the average of the column that contains the length of the sepals."
    ]
}
