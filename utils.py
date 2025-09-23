import csv
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import re
import ast


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
    # Find lists of quoted strings (e.g., 'setosa', 'versicolor')
    string_list_match = re.findall(r"['\"]([^'\"]+)['\"]", text)
    if len(string_list_match) > 1: # More than one quoted item found
        return sorted(string_list_match)

    # Find dictionaries (e.g., {'setosa': 50, ...})
    dict_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if dict_match:
        try:
            return ast.literal_eval(dict_match.group(0))
        except:
            pass

    # Find numerical values (including decimals and negatives)
    num_match = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if num_match:
        numbers = [float(n) if '.' in n else int(n) for n in num_match]
        # If the ground truth is a list of numbers, return the sorted list of extracted numbers
        if len(numbers) > 1:
            return sorted(numbers)
        # Otherwise, return the first number found
        return numbers[0]

    return None

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
            # --- CORRECTNESS CHECKING LOGIC ---
            extracted_data = extract_data_from_response(final_answer_text)

            # Prepare ground truth for comparison (sort lists)
            if isinstance(ground_truth_answer, list):
                ground_truth_answer_sorted = sorted(ground_truth_answer)
            else:
                ground_truth_answer_sorted = ground_truth_answer

            # Comparison logic
            is_correct = False
            if isinstance(extracted_data, float) and isinstance(ground_truth_answer_sorted, float):
                # Use tolerance for float comparison
                is_correct = abs(extracted_data - ground_truth_answer_sorted) < 0.01
            elif isinstance(extracted_data, list) and isinstance(ground_truth_answer_sorted, list):
                 is_correct = extracted_data == ground_truth_answer_sorted
            else:
                # Direct comparison for other types
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


