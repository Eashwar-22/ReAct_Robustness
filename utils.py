import pandas as pd
import numpy as np
import os
import csv
import re
from typing import Annotated, TypedDict
from datetime import datetime
import time
import ast
import json

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities.python import PythonREPL

# Ollama-specific imports
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Import your configuration and data handling modules
from config import OLLAMA_BASE_URL, LLM_MODEL, DATA_FILE_PATH1, DATA_FILE_PATH3, ANSWER_KEY_PATH, OUTPUTS_FOLDER
from data_handler import load_data, make_noisy_data




def log_to_csv(log_file_path: str,
               model_name: str, 
               dataset_name: str, 
               task: str, 
               noise_type: str, 
               result: str, 
               final_answer: str = '', 
               agent_produced_output: str = '', 
               ground_truth: str = '', 
               error_type: str = '', 
               failure_mode: str = '', 
               run_duration: float = 0.0, 
               final_code: str = ''):
    """Logs the result of an experiment to a CSV file."""
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset_name', 'model', 'task', 'noise_type', 'result', 'final_answer', 'agent_produced_output', 'ground_truth', 'error_type', 'failure_mode', 'run_duration', 'final_code']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'dataset_name': dataset_name, 
            'model': model_name,
            'task': task, 
            'noise_type': noise_type,
            'result': result, 
            'final_answer': final_answer, 
            'agent_produced_output': agent_produced_output,
            'ground_truth': ground_truth, 
            'error_type': error_type, 
            'failure_mode': failure_mode,
            'run_duration': run_duration, 
            'final_code': final_code
        })


def extract_data_from_response(text: str):
    """Extracts numerical values, lists, or dictionaries from the agent's final text response."""
    string_list_match = re.findall(r"['\"]([^'\"]+)['\"]", text)
    if len(string_list_match) > 1:
        return sorted(string_list_match)

    dict_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if dict_match:
        try:
            return ast.literal_eval(dict_match.group(0))
        except:
            pass

    num_match = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if num_match:
        numbers = [float(n) if '.' in n else int(n) for n in num_match]
        if len(numbers) > 1:
            return sorted(numbers)
        return numbers[0]

    return None
