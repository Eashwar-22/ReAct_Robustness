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
