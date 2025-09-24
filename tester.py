import time
import json
import re
import pandas as pd
import os
import csv
from datetime import datetime
import ast
import math

# Import the classes from your other files
from datahandler import DataHandler
from agent import AgentSetup 
import config

class ResultLogger:
    """
    Handles the creation and management of a CSV log file for test results.
    """
    def __init__(self, output_dir: str):
        """
        Initializes the logger and sets up the log file.

        Args:
            output_dir (str): The directory where the log file will be saved.
        """
        self.output_dir = output_dir
        self.log_file_path = self._setup_log_file()
        self.fieldnames = [
            "dataset_name", "task", "noise_type", "result", "final_answer",
            "agent_produced_output", "ground_truth", "agent_generated_code", # <-- ADDED
            "error_type", "failure_mode", "run_duration"
        ]
        self._write_header()

    def _setup_log_file(self) -> str:
        """Creates the output directory and defines the log file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"testrun_{timestamp}.csv")

    def _write_header(self):
        """Writes the header row to the CSV file."""
        with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_result(self, result_data: dict):
        """
        Logs a single test result to the CSV file.

        Args:
            result_data (dict): A dictionary containing the data for one test run.
        """
        with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Ensure all fields are present to avoid errors
            full_row = {key: result_data.get(key, "") for key in self.fieldnames}
            writer.writerow(full_row)
        print(f"  Result logged to '{self.log_file_path}'")

class RobustnessTester:
    """
    Orchestrates the entire process of testing the agent's robustness
    by coordinating the DataHandler, AgentSetup, and ResultLogger.
    """
    def __init__(self, datasets: dict, noise_types: list, answer_key_path: str):
        """
        Initializes the tester.

        Args:
            datasets (dict): A dictionary mapping dataset names to their file paths.
            noise_types (list): A list of noise type strings to apply.
            answer_key_path (str): The file path to the JSON answer key.
        """
        self.datasets = datasets
        self.noise_types = noise_types
        self.answer_key = self._load_answer_key(answer_key_path)
        
        # Initialize the components
        self.agent_setup = AgentSetup(base_url=config.OLLAMA_BASE_URL, model_name=config.LLM_MODEL)
        self.logger = ResultLogger(output_dir=config.OUTPUTS_FOLDER)
        
        self.temp_dir = "temp_data" # Directory for temporary noisy files
        self.noisy_file_paths = {} # To store paths of pre-generated noisy files
        self._prepare_noisy_datasets()


    def _prepare_noisy_datasets(self):
        """
        Generates all noisy versions of datasets before running experiments.
        """
        print("\n--- Preparing all noisy datasets upfront... ---")
        os.makedirs(self.temp_dir, exist_ok=True)

        for ds_name, ds_path in self.datasets.items():
            for noise_type in self.noise_types:
                if noise_type == "clean":
                    continue 
                
                print(f"  Generating noisy data for: Dataset='{ds_name}', Noise='{noise_type}'")
                data_handler = DataHandler(ds_path)
                noisy_df = data_handler.apply_noise(noise_type)

                if not noisy_df.empty:
                    temp_file_path = os.path.join(self.temp_dir, f"{noise_type}_{ds_name}.json")
                    noisy_df.to_json(temp_file_path, orient='records', indent=2)
                    self.noisy_file_paths[(ds_name, noise_type)] = temp_file_path
        print("--- All noisy datasets prepared. ---")

    def _load_answer_key(self, path: str) -> dict:
        """Loads the JSON answer key from a file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading answer key: {e}")
            return {}
    
    def _parse_answer_string(self, answer_str: str) -> any:
        """
        Attempts to parse a string into a Python object, fixing common errors first.
        """
        try:
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError, TypeError):
            try:
                fixed_str = re.sub(r"(?<=')\s+(?=')", "', '", answer_str)
                fixed_str = re.sub(r'(?<=")\s+(?=")', '", "', fixed_str)
                return ast.literal_eval(fixed_str)
            except (ValueError, SyntaxError, TypeError):
                return answer_str.strip()

    def _are_answers_equivalent(self, agent_answer_str: str, ground_truth_str: str) -> bool:
        """
        Compares two answers by first parsing them into Python objects.
        """
        agent_obj = self._parse_answer_string(agent_answer_str)
        gt_obj = self._parse_answer_string(ground_truth_str)

        if type(agent_obj) != type(gt_obj):
            if isinstance(agent_obj, (int, float)) and isinstance(gt_obj, (int, float)):
                pass 
            else:
                return False

        if isinstance(agent_obj, (int, float)):
            return math.isclose(agent_obj, gt_obj, rel_tol=1e-5)

        if isinstance(agent_obj, list):
            try:
                agent_list_sorted = sorted(map(str, agent_obj))
                gt_list_sorted = sorted(map(str, gt_obj))
                return agent_list_sorted == gt_list_sorted
            except TypeError:
                return False 

        if isinstance(agent_obj, dict):
            return agent_obj == gt_obj

        return str(agent_obj) == str(gt_obj)

    def run_experiments(self):
        """
        Main loop to run tests across all datasets, tasks, and noise types.
        """
        print("\n--- Starting Robustness Testing Experiments ---")
        for ds_name, ds_path in self.datasets.items():
            tasks = self.answer_key.get(ds_name, [])
            for task_info in tasks:
                question = task_info["question"]
                ground_truth = str(task_info["answer"])

                for noise_type in self.noise_types:
                    self._run_single_test(ds_name, ds_path, question, ground_truth, noise_type)
        print("\n--- All Experiments Complete ---")

    def _run_single_test(self, ds_name: str, ds_path: str, question: str, ground_truth: str, noise_type: str):
        """Executes and logs a single test run using pre-generated data."""
        print("-"*10)
        start_time = time.time()
        print(f"\nRunning test: Dataset='{ds_name}', Noise='{noise_type}', Task='{question}'")

        log_data = {
            "dataset_name": ds_name, "task": question,
            "noise_type": noise_type, "ground_truth": ground_truth
        }
        
        if noise_type == "clean":
            effective_path = ds_path
        else:
            effective_path = self.noisy_file_paths.get((ds_name, noise_type))
            if not effective_path:
                print(f"  Warning: Noisy file for '{ds_name}' with noise '{noise_type}' not found. Skipping test.")
                return

        try:
            df = pd.read_json(effective_path)
            columns = df.columns.tolist()
            
            full_question = (
                f"You are working with a file located at '{effective_path}'. "
                f"The DataFrame `df` has the following columns: {columns}. "
                f"Based on this, answer the following question: {question}"
            )

            result_state = self.agent_setup.executor.invoke({
                "messages": [("human", full_question)]
            })
            
            final_response = result_state['messages'][-1].content
            
            # --- NEW: Extract the generated code ---
            agent_code = "N/A" # Default if not found
            if len(result_state['messages']) > 1 and result_state['messages'][1].tool_calls:
                tool_calls = result_state['messages'][1].tool_calls
                if tool_calls and isinstance(tool_calls, list) and tool_calls[0].get('args'):
                    agent_code = tool_calls[0]['args'].get('code', "N/A")
            # --- END NEW ---

            if len(result_state['messages']) < 3:
                raise ValueError("Agent did not call the Python tool. Final response: " + final_response)
            
            agent_output = result_state['messages'][2].content
            # if "TypeError(" in agent_output:
            #    agent_output = "Type Error"
            print("Agent output: ",agent_output)
            print("Ground truth: ", ground_truth)

            log_data.update({
                "final_answer": final_response,
                "agent_produced_output": agent_output,
                "agent_generated_code": agent_code, # <-- ADDED
            })
            
            if self._are_answers_equivalent(agent_output, ground_truth):
                log_data["result"] = "Success"
            else:
                log_data["result"] = "Failure"
                log_data["failure_mode"] = "Correctness Failure"

        except Exception as e:
            log_data["result"] = "Failure"
            log_data["error_type"] = type(e).__name__
            log_data["failure_mode"] = "Execution Failure"
            print(f"  ERROR during test run: {e}")

        finally:
            run_duration = time.time() - start_time
            log_data["run_duration"] = round(run_duration, 2)
            self.logger.log_result(log_data)
            print(f"  Test completed in {log_data['run_duration']}s. Result: {log_data.get('result', 'Unknown')}")


def main():
    """Main function to initialize and run the tester."""
    datasets = {
        # "iris": config.DATA_FILE_PATH3,
        "electric_vehicles": config.DATA_FILE_PATH1
    }
    noise_types = ["clean", "shuffled_headers"]

    tester = RobustnessTester(
        datasets=datasets,
        noise_types=noise_types,
        answer_key_path=config.ANSWER_KEY_PATH
    )
    tester.run_experiments()

if __name__ == '__main__':
    main()

