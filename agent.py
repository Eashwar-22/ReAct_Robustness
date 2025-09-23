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
        full_script = (
            f"import pandas as pd\n"
            f"import numpy as np\n"
            f"df = pd.read_json('{file_path}')\n"
            f"{code}"
        )
        output = repl.run(full_script)
        return output
    except FileNotFoundError:
        return f"Execution Error: The file was not found at the path: {file_path}"
    except Exception as e:
        return f"Execution Error: {type(e).__name__} - {e}"

def log_to_csv(log_file_path: str, dataset_name: str, task: str, noise_type: str, result: str, final_answer: str = '', agent_produced_output: str = '', ground_truth: str = '', error_type: str = '', failure_mode: str = '', run_duration: float = 0.0, final_code: str = ''):
    """Logs the result of an experiment to a CSV file."""
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset_name', 'task', 'noise_type', 'result', 'final_answer', 'agent_produced_output', 'ground_truth', 'error_type', 'failure_mode', 'run_duration', 'final_code']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'dataset_name': dataset_name, 'task': task, 'noise_type': noise_type,
            'result': result, 'final_answer': final_answer, 'agent_produced_output': agent_produced_output,
            'ground_truth': ground_truth, 'error_type': error_type, 'failure_mode': failure_mode,
            'run_duration': run_duration, 'final_code': final_code
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

tools = [execute_python_code]
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)

# --- NEW, STRICTER SYSTEM PROMPT ---
system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a data analysis agent. Your task is to write a single, executable line of Python code to answer a question.

**RULES:**
1. A pandas DataFrame is already loaded for you as a variable named `df`.
2. **DO NOT** write `import pandas as pd`.
3. **DO NOT** define functions.
4. **DO NOT** write `df = pd.read_json(...)`.
5. Your code should be a single expression that can be printed, for example: `print(df['column_name'].mean())`.
6. If you get a `KeyError`, your **ONLY** next step is to print the columns with `print(df.columns)`. After you see the correct column names, use them in your next attempt.

Conclude with a final, human-readable answer summarizing the result of your code."""),
        ("placeholder", "{messages}"),
    ]
)

agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
print("Agent created successfully.")

def run_and_log_task(agent, inputs, dataset_name, task, noise_type, log_file_path, ground_truth_answer):
    final_answer_text = ""
    error_type = ""
    failure_mode = ""
    final_code = ""
    result_status = "Failure"
    extracted_data = None
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
            extracted_data = extract_data_from_response(final_answer_text)

            if isinstance(ground_truth_answer, list):
                ground_truth_answer_sorted = sorted(ground_truth_answer)
            else:
                ground_truth_answer_sorted = ground_truth_answer

            is_correct = False
            if isinstance(extracted_data, float) and isinstance(ground_truth_answer_sorted, float):
                is_correct = abs(extracted_data - ground_truth_answer_sorted) < 0.01
            elif isinstance(extracted_data, list) and isinstance(ground_truth_answer_sorted, list):
                 is_correct = extracted_data == ground_truth_answer_sorted
            else:
                is_correct = (extracted_data == ground_truth_answer_sorted)

            if is_correct:
                result_status = "Success"
                failure_mode = "N/A"
                error_type = "N/A"
            else:
                result_status = "Failure"
                failure_mode = "Correctness Failure"
                error_type = f"Output '{extracted_data}' did not match ground truth '{ground_truth_answer_sorted}'."
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

    log_to_csv(log_file_path, dataset_name, task, noise_type, result_status,
               final_answer=final_answer_text,
               agent_produced_output=str(extracted_data),
               ground_truth=str(ground_truth_answer),
               error_type=error_type,
               failure_mode=failure_mode,
               run_duration=run_duration,
               final_code=final_code)

    print(f"Task '{task}' on '{dataset_name}' with '{noise_type}' -> {result_status} ({failure_mode})")


if __name__ == '__main__':
    load_dotenv()

    datasets = {
        'electric_vehicles': {'path': DATA_FILE_PATH1},
        'iris': {'path': DATA_FILE_PATH3}
    }
    noise_types = ['shuffled_headers']

    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(OUTPUTS_FOLDER, f"testrun_{timestamp}.csv")

    with open(ANSWER_KEY_PATH, 'r') as f:
        answer_key_data = json.load(f)

    total_iterations = 0
    for dataset_name in datasets:
        if dataset_name in answer_key_data:
            total_iterations += len(answer_key_data[dataset_name]) * (1 + len(noise_types))
            
    print(f"Total iterations to run: {total_iterations}")

    run_count = 0

    for dataset_name, dataset_info in datasets.items():
        if dataset_name not in answer_key_data:
            print(f"\nSkipping dataset '{dataset_name}' - no questions found in answer key.")
            continue

        print("\n" + "=" * 30 + f" PROCESSING DATASET: {dataset_name.upper()} " + "=" * 30)
        
        tasks_for_dataset = answer_key_data[dataset_name]
        
        for task_info in tasks_for_dataset:
            task = task_info['question']
            ground_truth = task_info['answer']
            
            # --- Run on Clean Data ---
            run_count += 1
            print(f"\n--- Running iteration {run_count}/{total_iterations} (Clean: {dataset_name}) ---")
            
            file_path = dataset_info['path']
            task_prompt = (f"First, inspect the dataframe at '{file_path}' to understand its columns. "
                           f"Then, perform the following task: '{task}'. "
                           f"You MUST use the exact file path '{file_path}' in the 'file_path' argument for 'execute_python_code'.")
            
            run_and_log_task(agent_executor, {"messages": [HumanMessage(content=task_prompt)]}, dataset_name, task, "clean", log_file_path, ground_truth_answer=ground_truth)

            # --- Run on Noisy Data ---
            for noise_type in noise_types:
                clean_df = load_data(file_path)
                if clean_df.empty:
                    print(f"Skipping noise type {noise_type} due to data loading failure for {dataset_name}.")
                    continue

                run_count += 1
                print(f"\n--- Running iteration {run_count}/{total_iterations} ({noise_type}: {dataset_name}) ---")

                noisy_df = make_noisy_data(clean_df, noise_type)
                noisy_file_path = os.path.join("data", f"noisy_{dataset_name}_{noise_type}.json")
                noisy_df.to_json(noisy_file_path, orient='records', indent=4)

                task_prompt = (f"First, inspect the dataframe at '{noisy_file_path}' to understand its columns. "
                               f"Then, perform the following task: '{task}'. "
                               f"You MUST use the exact file path '{noisy_file_path}' for the 'file_path' argument.")
                
                run_and_log_task(agent_executor, {"messages": [HumanMessage(content=task_prompt)]}, dataset_name, task, noise_type, log_file_path, ground_truth_answer=ground_truth)

    print("\n" + "=" * 70 + f"\nAll experiments complete. Check '{log_file_path}' for results.\n" + "=" * 70)