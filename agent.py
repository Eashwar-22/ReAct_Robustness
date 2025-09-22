# --- Main Agent Code (Corrected) ---
import pandas as pd
import numpy as np
import random
import os
import csv
import re
from typing import Annotated, TypedDict
from datetime import datetime
import time
import ast

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
from config import OLLAMA_BASE_URL, LLM_MODEL, DATA_FILE_PATH3, TASKS_BY_DATASET, OUTPUTS_FOLDER
from data_handler import load_data, make_noisy_data

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def execute_python_code(code: str, file_path: str) -> str:
    """
    Executes Python code to analyze a pandas DataFrame.
    The DataFrame is automatically loaded from the provided file_path
    and is available as the variable `df`.

    Args:
        code (str): The Python code to execute on the `df` DataFrame.
        file_path (str): The path to the JSON data file.
    """
    repl = PythonREPL()
    try:
        # --- CORRECTED LOGIC: Build the full script as a single string ---
        full_script = (
            f"import pandas as pd\n"
            f"import numpy as np\n"
            f"df = pd.read_json('{file_path}')\n"
            f"{code}"
        )
        # Execute the complete script string
        output = repl.run(full_script)
        return output
    except FileNotFoundError:
        return f"Execution Error: The file was not found at the path: {file_path}"
    except Exception as e:
        return f"Execution Error: {type(e).__name__} - {e}"


def log_to_csv(log_file_path: str, dataset_name: str, task: str, noise_type: str, result: str, final_answer: str = '', error_type: str = '', failure_mode: str = '', run_duration: float = 0.0, final_code: str = ''):
    """Logs the result of an experiment to a CSV file."""
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset_name', 'task', 'noise_type', 'result', 'final_answer', 'error_type', 'failure_mode', 'run_duration', 'final_code']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'dataset_name': dataset_name, 'task': task, 'noise_type': noise_type,
            'result': result, 'final_answer': final_answer, 'error_type': error_type,
            'failure_mode': failure_mode, 'run_duration': run_duration, 'final_code': final_code
        })

def extract_data_from_response(text: str):
    """Extracts numerical values or lists from the agent's final text response."""
    # Find lists (e.g., ['setosa', 'versicolor'])
    list_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if list_match:
        try:
            # Use ast.literal_eval for safe evaluation of the list string
            return sorted(ast.literal_eval(f"[{list_match.group(1)}]"))
        except:
            return None # Return None if parsing fails

    # Find numerical values (including decimals and negatives)
    num_match = re.findall(r'-?\d+\.\d+', text)
    if num_match:
        return [float(n) for n in num_match]

    return None

tools = [execute_python_code]
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a data analysis agent. Your goal is to provide a correct final answer. First, reason about the user's request, then use the execute_python_code tool. The DataFrame is already loaded for you as `df`. ALWAYS inspect the data with `df.head()` before performing calculations. If you encounter an error, you must try to self-correct. Conclude with a final, human-readable answer summarizing the result."),
        ("placeholder", "{messages}"),
    ]
)

agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
print("Agent created successfully.")

def run_and_log_task(agent, inputs, dataset_name, task, noise_type, log_file_path, ground_truth_answer=None):
    final_answer_text = ""
    error_type = ""
    failure_mode = ""
    final_code = ""
    result_status = "Failure"
    start_time = datetime.now()

    try:
        final_state = agent.invoke(inputs)
        messages = final_state.get('messages', [])
        
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.tool_calls:
                final_code = message.tool_calls[0]['args']['code']
                break
        if not final_code:
            final_code = "No Code Executed"

        has_execution_error = False
        error_keywords = ["KeyError", "TypeError", "NameError", "ValueError", "Exception", "Error"]
        for msg in messages:
            if isinstance(msg, ToolMessage) and any(keyword in msg.content for keyword in error_keywords):
                has_execution_error = True
                error_type = msg.content.strip()
                failure_mode = "Recognition Failure" if "KeyError" in error_type else "Execution Failure"
                break 

        final_message = messages[-1] if messages else None
        
        if has_execution_error:
            result_status = "Failure"
            final_answer_text = final_message.content.strip() if isinstance(final_message, AIMessage) else "Agent stopped after error."
        elif isinstance(final_message, AIMessage) and final_message.content.strip():
            final_answer_text = final_message.content.strip()
            # --- CORRECTNESS CHECKING LOGIC ---
            if ground_truth_answer is not None:
                extracted_data = extract_data_from_response(final_answer_text)
                if extracted_data == ground_truth_answer:
                    result_status = "Success"
                    failure_mode = "N/A"
                    error_type = "N/A"
                else:
                    result_status = "Failure"
                    failure_mode = "Correctness Failure"
                    error_type = f"Output '{extracted_data}' did not match ground truth '{ground_truth_answer}'."
            else: # This is a clean run, used to establish ground truth
                result_status = "Success"
                failure_mode = "N/A"
                error_type = "N/A"
        else:
            result_status = "Failure"
            failure_mode = "Planning Failure"
            final_answer_text = "Agent did not produce a final answer."

    except Exception as e:
        error_type = f"Critical Error: {type(e).__name__}"
        failure_mode = "Critical Failure"
        result_status = "Failure"

    end_time = datetime.now()
    run_duration = (end_time - start_time).total_seconds()
    
    log_to_csv(log_file_path, dataset_name, task, noise_type, result_status, final_answer_text, error_type, failure_mode, run_duration, final_code)
    print(f"Task '{task}' on '{dataset_name}' with '{noise_type}' -> {result_status} ({failure_mode})")
    
    # Return the extracted data for caching if it was a successful clean run
    if result_status == "Success" and ground_truth_answer is None:
        return extract_data_from_response(final_answer_text)
    return None


if __name__ == '__main__':
    load_dotenv()
    
    datasets = {'iris': {'path': DATA_FILE_PATH3, 'tasks': TASKS_BY_DATASET['iris']}}
    noise_types = ['shuffled_headers', 
                   #'misspelled_headers', 'typos_in_content', 'mixed_data_types', 'missing_values'
                   ]
    
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(OUTPUTS_FOLDER, f"testrun_{timestamp}.csv")
    
    total_iterations = sum(len(info['tasks']) * (1 + len(noise_types)) for info in datasets.values())
    print(f"Total iterations to run: {total_iterations}")
    
    ground_truth_cache = {}
    run_count = 0

    for dataset_name, dataset_info in datasets.items():
        # --- PHASE 1: ESTABLISH GROUND TRUTH ---
        print("\n" + "=" * 70 + f"\nPHASE 1: Establishing Ground Truth for: {dataset_name.upper()}\n" + "=" * 70)
        for task in dataset_info['tasks']:
            run_count += 1
            print(f"\n--- Running iteration {run_count}/{total_iterations} (Clean) ---")
            task_prompt = (f"Your task: '{task}'. The data is at: '{dataset_info['path']}'. You MUST use this path in the 'file_path' argument for 'execute_python_code'.")
            ground_truth = run_and_log_task(agent_executor, {"messages": [HumanMessage(content=task_prompt)]}, dataset_name, task, "clean", log_file_path)
            if ground_truth is not None:
                ground_truth_cache[task] = ground_truth
                print(f"Ground truth for '{task}': {ground_truth}")

        # --- PHASE 2: EVALUATE ON NOISY DATA ---
        print("\n" + "=" * 70 + f"\nPHASE 2: Evaluating Robustness on Noisy Data\n" + "=" * 70)
        for noise_type in noise_types:
            clean_df = load_data(dataset_info['path'])
            if clean_df.empty:
                print(f"Skipping noise type {noise_type} due to data loading failure.")
                continue

            for task in dataset_info['tasks']:
                run_count += 1
                print(f"\n--- Running iteration {run_count}/{total_iterations} ({noise_type}) ---")
                
                noisy_df = make_noisy_data(clean_df, noise_type)
                noisy_file_path = os.path.join("data", f"noisy_{dataset_name}_{noise_type}.json")
                noisy_df.to_json(noisy_file_path, orient='records', indent=4)

                task_prompt = (f"Your task: '{task}'. The noisy data is at: '{noisy_file_path}'. You MUST use this path for the 'file_path' argument.")
                # Retrieve the ground truth for comparison
                truth_for_task = ground_truth_cache.get(task)
                run_and_log_task(agent_executor, {"messages": [HumanMessage(content=task_prompt)]}, dataset_name, task, noise_type, log_file_path, ground_truth_answer=truth_for_task)

    print("\n" + "=" * 70 + f"\nAll experiments complete. Check '{log_file_path}' for results.\n" + "=" * 70)

