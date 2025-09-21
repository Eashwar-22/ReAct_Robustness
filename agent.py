# --- Main Agent Code ---
import pandas as pd
import numpy as np
import random
import os
import csv
import re
from typing import Annotated, TypedDict, List, Union
from datetime import datetime
import time

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_experimental.utilities.python import PythonREPL

# Ollama-specific imports
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Import your configuration and data handling modules
from config import OLLAMA_BASE_URL, LLM_MODEL, DATA_FILE_PATH1, DATA_FILE_PATH2, DATA_FILE_PATH3, TASKS_BY_DATASET, OUTPUTS_FOLDER
from data_handler import load_data, make_noisy_data

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
@tool
def execute_python_code(code: str, file_path: str) -> str:
    """Execute a Python script in a sandboxed environment.
    The pandas library is already imported as 'pd', numpy as 'np',
    and the DataFrame is available as 'df'.
    
    The output of the code should be printed. For example: `print(df.head())`.
    
    Args:
        code: The Python code to execute, as a single string.
        file_path: The path to the data file to be loaded into the DataFrame.
    """
    repl = PythonREPL()
    try:
        exec_code = f"import pandas as pd\nimport numpy as np\nimport json\nwith open('{file_path}', 'r') as f:\n    data = json.load(f)\ndf = pd.json_normalize(data)\n{code}"
        output = repl.run(exec_code)
        return output
    except Exception as e:
        return f"Error executing code: {e}"

def log_to_csv(log_file_path: str, dataset_name: str, task: str, noise_type: str, result: str, final_answer: str = '', error_type: str = '', failure_mode: str = '', run_duration: float = 0.0, final_code: str = ''):
    """
    Logs the result of an experiment to a CSV file with more detailed columns.
    
    Args:
        log_file_path: The full path to the log file.
        dataset_name: The name of the dataset used.
        task: The task that was performed.
        noise_type: The type of noise introduced.
        result: The final output of the agent (Success or Failure).
        final_answer: The final natural language response from the agent.
        error_type: The type of error encountered (e.g., KeyError).
        failure_mode: The classified failure mode (e.g., 'Recognition Failure').
        run_duration: The total time the run took.
        final_code: The last piece of code the agent attempted to run.
    """
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset_name', 'task', 'noise_type', 'result', 'final_answer', 'error_type', 'failure_mode', 'run_duration', 'final_code']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow({
            'dataset_name': dataset_name,
            'task': task,
            'noise_type': noise_type,
            'result': result,
            'final_answer': final_answer,
            'error_type': error_type,
            'failure_mode': failure_mode,
            'run_duration': run_duration,
            'final_code': final_code
        })
    return "Log entry created."

tools = [execute_python_code]
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)

system_prompt = f"""
You are a highly skilled data analysis agent. Your primary goal is to perform in-depth data analysis on a pandas DataFrame.

Your workflow should follow these steps:
1.  **Initial Plan:** Start by thinking about the user's request and the best way to approach it.
2.  **Data Inspection:** You MUST begin by inspecting the DataFrame. Use `df.head()` or `df.info()` to understand its columns, data types, and any potential issues before taking action.
3.  **Action:** Write and execute Python code using the provided `execute_python_code` tool to perform the requested analysis.
4.  **Error Handling:** Carefully read the output of your code. If an error occurs, you must re-evaluate your plan and correct your code. For example, if you get a `KeyError`, you must re-inspect the DataFrame's columns to find the correct name.
5.  **Final Answer:** Once the analysis is complete and you have a result, provide a clear and concise final answer.

The DataFrame is available as the variable 'df'. Do not make assumptions about column names or data types without first inspecting the data.
"""

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
)
print("Agent created with a self-correcting pre-analysis prompt.")

def run_and_log_task(agent, inputs, dataset_name, task, noise_type, log_file_path):
    full_output = ""
    final_answer = ""
    error_type = ""
    failure_mode = ""
    final_code = ""
    result_status = "Failure"
    start_time = datetime.now()
    
    try:
        stream_results = []
        for step in agent.stream(inputs):
            stream_results.append(step)

        # Re-construct full output from stream_results
        for step in stream_results:
            full_output += str(step) + "\n"

        # Check the last message for a final answer
        if stream_results and 'messages' in stream_results[-1]:
            last_message = stream_results[-1]['messages'][-1]
            if isinstance(last_message, AIMessage) and last_message.content.strip():
                # A final answer is usually the last AIMessage content
                final_answer = last_message.content.strip()

        # Check for tool-call errors throughout the whole output
        error_match = re.search(r"Error executing code: (.*)", full_output)
        keyerror_match = re.search(r"KeyError\((\'.*?\')\)", full_output)
        
        if keyerror_match:
            result_status = "Failure"
            error_type = keyerror_match.group(1).strip()
            failure_mode = "Recognition Failure"
        elif error_match:
            result_status = "Failure"
            error_type = error_match.group(1).strip()
            failure_mode = "Execution Failure"
        elif final_answer:
            result_status = "Success"
        else:
            result_status = "Failure"
            failure_mode = "Planning Failure"
        
        # Capture the final code
        code_match = re.findall(r"'code': \"(.*?)\"", full_output, re.DOTALL)
        final_code = code_match[-1] if code_match else "No Code"

    except Exception as e:
        # A crash in the stream itself implies a planning failure
        result_status = "Failure"
        error_type = type(e).__name__
        failure_mode = "Planning Failure"
    
    end_time = datetime.now()
    run_duration = (end_time - start_time).total_seconds()
    
    # Simple heuristic to clean up the final answer from extra text
    final_answer = re.sub(r"Final Answer:\s*", "", final_answer, flags=re.DOTALL)
    
    log_to_csv(log_file_path, dataset_name, task, noise_type, result_status, final_answer, error_type, failure_mode, run_duration, final_code)
    print(f"Task '{task}' on '{dataset_name}' with '{noise_type}' -> {result_status}")

if __name__ == '__main__':
    load_dotenv()
    
    datasets = {
      #  'electric_vehicles': {'path': DATA_FILE_PATH1, 'tasks': TASKS_BY_DATASET['electric_vehicles']},
      #  'heart_disease': {'path': DATA_FILE_PATH2, 'tasks': TASKS_BY_DATASET['heart_disease']},
        'iris': {'path': DATA_FILE_PATH3, 'tasks': TASKS_BY_DATASET['iris']}
    }
    
    noise_types = ['shuffled_headers', ]
    # 'misspelled_headers', 'typos_in_content', 'mixed_data_types', 'missing_values']
    
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"testrun_{timestamp}.csv"
    log_file_path = os.path.join(OUTPUTS_FOLDER, log_file_name)
    
    total_iterations = 0
    for dataset_name in datasets:
        num_tasks = len(datasets[dataset_name]['tasks'])
        total_iterations += num_tasks
        total_iterations += num_tasks * len(noise_types)

    print(f"Total iterations to run: {total_iterations}")
    
    run_count = 0
    
    for dataset_name, dataset_info in datasets.items():
        print("\n" + "=" * 70)
        print(f"Running experiments for dataset: {dataset_name.upper()}")
        print("=" * 70)

        for task in dataset_info['tasks']:
            run_count += 1
            print(f"\n--- Running iteration {run_count}/{total_iterations} ---")
            inputs_clean = {"messages": [HumanMessage(content=f"Use the data from {dataset_info['path']} to answer: {task}")]}
            run_and_log_task(agent_executor, inputs_clean, dataset_name, task, "clean", log_file_path)
        
        for noise_type in noise_types:
            for task in dataset_info['tasks']:
                run_count += 1
                print(f"\n--- Running iteration {run_count}/{total_iterations} ---")
                inputs_noisy = {"messages": [HumanMessage(content=f"Use the noisy data from {dataset_info['path']} with '{noise_type}' noise to answer: {task}")]}
                run_and_log_task(agent_executor, inputs_noisy, dataset_name, task, noise_type, log_file_path)

    print("\n" + "=" * 70)
    print(f"All experiments complete. Check '{log_file_path}' for results.")
