Of course. Here is a detailed README file for your project.

-----

# AI Agent Robustness Testing Framework

This project provides a framework for systematically evaluating the robustness of a ReAct-style AI agent against various types of data noise. It automates the process of running data analysis tasks on both clean and noisy datasets, logging the results, and comparing the agent's performance against a pre-defined answer key.

## 📋 Table of Contents

  - [Overview](https://www.google.com/search?q=%23-overview)
  - [Features](https://www.google.com/search?q=%23-features)
  - [Project-Structure](https://www.google.com/search?q=%23-project-structure)
  - [Setup-and-Installation](https://www.google.com/search?q=%23-setup-and-installation)
  - [How-to-Run-the-Experiments](https://www.google.com/search?q=%23-how-to-run-the-experiments)
  - [Configuration](https://www.google.com/search?q=%23-configuration)
  - [Understanding-the-Output](https://www.google.com/search?q=%23-understanding-the-output)
  - [Extending-the-Framework](https://www.google.com/search?q=%23-extending-the-framework)

-----

## 🧐 Overview

The core idea is to test how well a data analysis agent can perform when the data it's working with is imperfect. Real-world data is often messy, with shuffled columns, typos, or missing values. This framework measures the agent's ability to self-correct and provide accurate answers under these adverse conditions.

The workflow is as follows:

1.  The agent is given a data analysis task for a specific dataset (e.g., "What is the average sepal length?").
2.  It first runs the task on the original, **clean** dataset.
3.  The agent's answer is compared to a ground-truth answer from an `answer_key.json` file.
4.  Next, **noise** is programmatically injected into the dataset (e.g., column headers are shuffled).
5.  The agent runs the exact same task on the new, **noisy** dataset.
6.  The results from both runs (clean and noisy) are logged to a CSV file for analysis.

-----

## ✨ Features

  * **Automated Evaluation:** Run a suite of tests across multiple datasets with a single command.
  * **Noise Injection:** Easily introduce different types of noise into datasets to simulate real-world data quality issues.
  * **Modular Design:** The agent, configuration, and data handling logic are separated into different files, making it easy to modify or extend.
  * **Detailed Logging:** Every experiment run is logged to a timestamped CSV file with rich details, including the task, noise type, result, agent's final answer, run duration, and the exact code the agent executed.
  * **Ground-Truth Comparison:** Uses a JSON-based answer key for objective and reliable correctness checking.

-----

## 📁 Project Structure

```
ReAct_Robustness/
│
├── 📂 data/
│   ├── electric_vehicles_dataset.json # Example dataset
│   ├── iris.json                      # Example dataset
│   └── answer_key.json                # Ground truth answers for tasks
│
├── 📂 outputs/
│   └── testrun_YYYYMMDD_HHMMSS.csv    # Log files from experiment runs
│
├── 📜 agent.py                         # Main script: defines the agent and runs the experiments
├── 📜 config.py                       # Configuration file for paths and model settings
├── 📜 data_handler.py                  # Handles data loading and noise injection
├── 📜 .env                             # Environment variables (for Ollama URL, etc.)
└── 📜 requirements.txt                 # Python dependencies
```

  * **`agent.py`**: The orchestrator. It defines the LangGraph agent, sets up the experiment loop, and calls the logging functions.
  * **`config.py`**: A central hub for all file paths and model configuration.
  * **`data_handler.py`**: Contains the logic for loading JSON data into pandas DataFrames and applying various types of noise.
  * **`data/answer_key.json`**: The source of truth. It's structured by dataset, containing a list of questions and their correct answers.
  * **`outputs/`**: All experiment results are saved here in CSV files.

-----

## 🛠️ Setup and Installation

Follow these steps to get the project running.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd ReAct_Robustness
```

**2. Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
Create a `requirements.txt` file with the following content:

```
pandas
numpy
python-dotenv
langchain
langgraph
langchain-ollama
langchain-experimental
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**
Create a file named `.env` in the root of the project directory and add the configuration for your Ollama instance:

```
OLLAMA_BASE_URL="http://localhost:11434"
LLM_MODEL="llama3.2" # Or any other model you have pulled with Ollama
```

-----

## 🚀 How to Run the Experiments

Once the setup is complete, you can run the entire evaluation suite with a single command from the root of the project directory:

```bash
python agent.py
```

The script will print its progress to the console and save the detailed results in the `outputs` directory.

-----

## ⚙️ Configuration

You can easily customize the experiments by modifying a few files:

  * **To Add New Datasets:**

    1.  Place your new JSON dataset (e.g., `my_dataset.json`) in the `data/` directory.
    2.  Open `config.py` and add a new path variable for it.
    3.  In `agent.py`, add your new dataset to the `datasets` dictionary in the `if __name__ == '__main__':` block.

  * **To Add New Tasks:**

    1.  Open `data/answer_key.json`.
    2.  Find the key for the relevant dataset (e.g., `"iris"`).
    3.  Add a new object to the list with a `"question"` and its corresponding `"answer"`.

  * **To Add New Noise Types:**

    1.  Open `data_handler.py`.
    2.  In the `make_noisy_data` function, add a new `elif noise_type == "my_new_noise":` block.
    3.  Implement the logic to apply your new type of noise to the DataFrame.
    4.  In `agent.py`, add your `"my_new_noise"` to the `noise_types` list.

-----

## 📊 Understanding the Output

The results of each run are saved in a CSV file in the `outputs/` folder. Here’s a description of each column:

  * **`dataset_name`**: The dataset used for the task.
  * **`task`**: The question that was asked to the agent.
  * **`noise_type`**: The type of noise applied (`clean` for the original dataset).
  * **`result`**: The final status of the run (`Success` or `Failure`).
  * **`final_answer`**: The full, human-readable text response from the agent.
  * **`agent_produced_output`**: The specific data that was extracted from the agent's final answer for comparison.
  * **`ground_truth`**: The correct answer from the answer key.
  * **`error_type`**: If a failure occurred, this shows the technical reason (e.g., `KeyError`).
  * **`failure_mode`**: A high-level classification of the failure (e.g., `Correctness Failure`, `Recognition Failure`).
  * **`run_duration`**: The time in seconds it took to complete the task.
  * **`final_code`**: The last piece of Python code the agent attempted to execute.

-----

## 🧩 Extending the Framework

This framework is designed to be extensible. Here are a few ideas for taking it further:

  * **Implement More Noise Types**: Add functions to `data_handler.py` for `missing_values`, `typos_in_content`, or `mixed_data_types`.
  * **Test Different Agents**: Modify `agent.py` to use a different agent architecture (e.g., a different model from Ollama, or an agent from another provider like OpenAI or Anthropic).
  * **Add More Complex Tasks**: Expand the `answer_key.json` with more challenging, multi-step analysis questions.
  * **Data Visualization**: Create a script or a Jupyter notebook to read the output CSV files and generate charts to visualize the agent's performance across different noise types.