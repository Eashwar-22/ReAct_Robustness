# AI Agent Robustness Testing Framework

This project provides a framework for systematically evaluating the robustness of a ReAct-style AI agent against various types of data noise. It automates the process of running data analysis tasks on both clean and noisy datasets, logging the results, and comparing the agent's performance against a pre-defined answer key.

## Table of Contents

  - [Motivation]
  - [Overview]
  - [Problem-Statement]
  - [Features]
  - [Project-Structure]
  - [Methodology]
  - [Setup-and-Installation]
  - [How-to-Run-the-Experiments]
  - [Configuration]
  - [Understanding-the-Output]
  - [Overview-of-Noise-Types-and-Failure-Modes]
  - [Extending-the-Framework]

-----

## Motivation

  * [cite\_start]LLMs are increasingly employed as autonomous agents to perform complex data analysis[cite: 4].
  * [cite\_start]Businesses and researchers are actively exploring these agents for tasks ranging from generating summary statistics to complex modeling[cite: 5].
  * [cite\_start]However, their reliability degrades significantly when faced with the noisy, imperfectly structured tabular data common in real-world scenarios[cite: 6].
  * [cite\_start]Data often deviates from the clean, well-structured formats found in benchmarks[cite: 7].

-----

## Overview

The core idea is to test how well a data analysis agent can perform when the data it's working with is imperfect. Real-world data is often messy, with shuffled columns, typos, or missing values. This framework measures the agent's ability to self-correct and provide accurate answers under these adverse conditions.

The workflow is as follows:

1.  The agent is given a data analysis task for a specific dataset (e.g., "What is the average sepal length?").
2.  It first runs the task on the original, **clean** dataset.
3.  The agent's answer is compared to a ground-truth answer from an `answer_key.json` file.
4.  Next, **noise** is programmatically injected into the dataset (e.g., column headers are shuffled).
5.  The agent runs the exact same task on the new, **noisy** dataset.
6.  The results from both runs (clean and noisy) are logged to a CSV file for analysis.

-----

## Problem Statement

Current research on LLM robustness for tabular data often focuses on:

  * [cite\_start]a) General TQA tasks against various perturbations[cite: 22].
  * [cite\_start]b) Evaluating structural understanding based on table representations[cite: 23].
  * [cite\_start]c) Evaluating agent performance on complex tasks, often assuming clean data inputs[cite: 24].

[cite\_start]There remains a gap in systematically studying the reliability of multi-step data analysis agents when executing specific computational tasks on tables containing realistic noise[cite: 25]. [cite\_start]It is unclear how specific noise types impede particular stages of the agent's workflow (planning, code generation, execution, interpretation) for different kinds of analysis tasks[cite: 26].

-----

## Features

  * **Automated Evaluation:** Run a suite of tests across multiple datasets with a single command.
  * **Noise Injection:** Easily introduce different types of noise into datasets to simulate real-world data quality issues.
  * **Modular Design:** The agent, configuration, and data handling logic are separated into different files, making it easy to modify or extend.
  * **Detailed Logging:** Every experiment run is logged to a timestamped CSV file with rich details, including the task, noise type, result, agent's final answer, run duration, and the exact code the agent executed.
  * **Ground-Truth Comparison:** Uses a JSON-based answer key for objective and reliable correctness checking.

-----

## Project Structure

```
ReAct_Robustness/
│
├── 📂 data/
│   ├── electric_vehicles_dataset.json
│   ├── iris.json
│   └── answer_key.json
│
├── 📂 outputs/
│   └── testrun_YYYYMMDD_HHMMSS.csv
│
├── 📜 agent.py
├── 📜 config.py
├── 📜 datahandler.py
├── 📜 tester.py
├── 📜 utils.py
├── 📜 .env
└── 📜 requirements.txt
```

  * **`tester.py`**: The orchestrator. It initializes the agent, prepares the datasets, runs the experiment loop, and logs the results.
  * **`agent.py`**: Defines the LangGraph agent, including the system prompt, tools, and model configuration.
  * **`config.py`**: A central hub for all file paths and model configuration.
  * **`datahandler.py`**: Contains the logic for loading JSON data into pandas DataFrames and applying various types of noise.
  * **`utils.py`**: Provides utility functions for logging and data extraction from the agent's responses.
  * **`data/answer_key.json`**: The source of truth. It's structured by dataset, containing a list of questions and their correct answers.
  * **`outputs/`**: All experiment results are saved here in CSV files.

-----

## Methodology

  * **Agent Implementation:**
      * [cite\_start]**Framework:** LangGraph[cite: 68].
      * [cite\_start]**Engine:** LLM API (e.g., GPT-4, Claude 3)[cite: 69].
      * [cite\_start]**Execution:** Sandboxed Python Environment (Pandas, NumPy, etc.)[cite: 69].
      * [cite\_start]**Workflow:** ReAct-style (Plan -\> Code -\> Execute -\> Observe)[cite: 70].
  * **Experimental Design:**
      * [cite\_start]**Tasks:** Specific computational analyses (e.g., Grouped Aggregation + Outlier Detection)[cite: 72].
      * [cite\_start]**Noise Types:** Realistic variations (e.g., Header Synonyms, Shuffled Columns, Merged Cells)[cite: 72].
      * [cite\_start]**Data:** Public datasets (e.g., Kaggle); generate Clean vs. Noisy pairs[cite: 72, 73].
  * **Evaluation Metrics:**
      * [cite\_start]**Task Success Rate:** Compare agent output vs. ground truth[cite: 77].
      * [cite\_start]**Robustness Drop:** Measure performance decrease (Clean vs. Noisy)[cite: 78].
  * **Core Analysis: Workflow Failures**
      * [cite\_start]Detailed qualitative error analysis of agent logs (reasoning, code, execution results)[cite: 82].
      * [cite\_start]Identify failure modes (Planning, Code Gen, etc.) linked to specific Task-Noise interactions[cite: 84].

-----

## Setup and Installation

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

## How to Run the Experiments

Once the setup is complete, you can run the entire evaluation suite with a single command from the root of the project directory:

```bash
python tester.py
```

The script will print its progress to the console and save the detailed results in the `outputs` directory.

-----

## Configuration

You can easily customize the experiments by modifying a few files:

  * **To Add New Datasets:**

    1.  Place your new JSON dataset (e.g., `my_dataset.json`) in the `data/` directory.
    2.  Open `config.py` and add a new path variable for it.
    3.  In `tester.py`, add your new dataset to the `datasets` dictionary in the `main()` function.

  * **To Add New Tasks:**

    1.  Open `data/answer_key.json`.
    2.  Find the key for the relevant dataset (e.g., `"iris"`).
    3.  Add a new object to the list with a `"question"` and its corresponding `"answer"`.

  * **To Add New Noise Types:**

    1.  Open `datahandler.py`.
    2.  In the `apply_noise` function, add a new `elif noise_type == "my_new_noise":` block.
    3.  Implement the logic to apply your new type of noise to the DataFrame.
    4.  In `tester.py`, add your `"my_new_noise"` to the `noise_types` list in the `main()` function.

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
  * **`agent_generated_code`**: The Python code generated by the agent to answer the question.
  * **`error_type`**: If a failure occurred, this shows the technical reason (e.g., `KeyError`).
  * **`failure_mode`**: A high-level classification of the failure (e.g., `Correctness Failure`, `Execution Failure`).
  * **`run_duration`**: The time in seconds it took to complete the task.

-----

## Overview of Noise Types and Failure Modes

### Overview of Noise Types

The framework is designed to test the agent's robustness against different types of data imperfections. The following noise types are currently implemented:

  * **Shuffled Headers**: This noise type simulates a common real-world scenario where the column headers of a dataset are scrambled. The agent must be able to correctly identify the columns based on their content rather than their position or name.

### Overview of Failure Modes

The framework categorizes failures into the following types to provide a more granular understanding of the agent's weaknesses:

  * **Correctness Failure**: This occurs when the agent successfully executes a command, but the output it produces does not match the ground truth from the `answer_key.json` file. This indicates a failure in the agent's reasoning or data manipulation logic.
  * **Execution Failure**: This is a more critical error where the agent is unable to execute its generated code. This could be due to a syntax error, a `KeyError` from referencing a non-existent column, or other runtime exceptions.

-----

## Extending the Framework

This framework is designed to be extensible. Here are a few ideas for taking it further:

  * **Implement More Noise Types**: Add functions to `datahandler.py` for `missing_values`, `typos_in_content`, or `mixed_data_types`.
  * **Test Different Agents**: Modify `agent.py` to use a different agent architecture (e.g., a different model from Ollama, or an agent from another provider like OpenAI or Anthropic).
  * **Add More Complex Tasks**: Expand the `answer_key.json` with more challenging, multi-step analysis questions.
  * **Data Visualization**: Create a script or a Jupyter notebook to read the output CSV files and generate charts to visualize the agent's performance across different noise types.
  * **Integrate MLOps Practices**:
      * **CI/CD for Testing**: Set up a continuous integration/continuous deployment (CI/CD) pipeline using tools like GitHub Actions or Jenkins. This can automatically run the robustness tests whenever changes are pushed to the repository, ensuring that new code doesn't degrade the agent's performance.
      * **Experiment Tracking**: Use tools like MLflow or Weights & Biases to log the results of each test run. This will allow for more sophisticated querying, comparison, and visualization of results over time, especially when testing different agent versions or models.
      * **Model Registry**: If you experiment with different models, a model registry (often included in tools like MLflow) can help you version control your models and keep track of which model was used for which experiment.
      * **Automated Reporting**: Enhance the data visualization scripts to automatically generate and share reports (e.g., as a PDF or a web dashboard) summarizing the agent's performance after each test run.
  * [cite\_start]**Implement Non-training Based Enhancements**[cite: 33, 56, 64, 86, 87]:
      * [cite\_start]**Pre-analysis prompt step**: Add a prompt that instructs the agent to first examine the headers and clarify their meaning before proceeding if they are not standard names[cite: 35].
      * [cite\_start]**Modify the code generation prompt**: Ensure the agent's code explicitly handles potential inconsistencies in column names[cite: 36].
      * [cite\_start]**Add a check step in the LangGraph workflow**: After code execution, have the LLM check if the output format matches expectations before concluding[cite: 37].