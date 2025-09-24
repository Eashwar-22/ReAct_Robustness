from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from langchain_experimental.utilities.python import PythonREPL

from langchain_core.tools import tool

from config import OLLAMA_BASE_URL, LLM_MODEL, DATA_FILE_PATH1, DATA_FILE_PATH3, ANSWER_KEY_PATH, OUTPUTS_FOLDER
import os

class AgentState(TypedDict):
    """Defines the structure of the agent's state."""
    messages: Annotated[list, add_messages]

class AgentSetup:
    """
    A class to encapsulate the setup and creation of a data analysis agent.
    """
    def __init__(self, base_url: str, model_name: str):
        """
        Initializes the agent setup by creating all necessary components.
        
        Args:
            base_url (str): The base URL for the Ollama service.
            model_name (str): The name of the LLM model to use.
        """
        self.llm = ChatOllama(base_url=base_url, model=model_name)
        # CORRECTED: Referencing the method directly, not calling it.
        # The @tool decorator turns the method itself into the tool object.
        self.tools = [self._create_python_tool]
        self.prompt = self._create_system_prompt()
        self.executor = create_react_agent(model=self.llm, tools=self.tools, prompt=self.prompt)
        print("Agent executor created successfully.")

    @staticmethod
    def _create_system_prompt() -> ChatPromptTemplate:
        """Creates and returns the system prompt for the agent."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", """

                    YOU ARE THE WORLD’S FOREMOST EXPERT IN PANDAS-BASED DATA ANALYSIS. YOUR TASK IS TO WRITE A SINGLE, EXECUTABLE PYTHON LINE THAT ANSWERS QUESTIONS ABOUT A PANDAS DATAFRAME NAMED `df`.

                    ###INSTRUCTIONS###

                    1. ALWAYS OUTPUT **ONLY ONE** COMPLETE LINE OF PYTHON CODE WRAPPED IN `print()`.
                    2. USE **EXACT COLUMN NAMES** PROVIDED BY THE USER, WITHOUT CHANGES OR ASSUMPTIONS.
                    3. NEVER IMPORT LIBRARIES, DEFINE `df`, OR ADD ANY EXTRA LINES BEYOND THE `print()` STATEMENT.
                    4. AFTER THE CODE, WRITE A **BRIEF, HUMAN-READABLE SUMMARY** (ONE OR TWO SENTENCES) OF THE RESULT.
                    5. IF A `KeyError` OCCURS, RE-EXAMINE THE COLUMN LIST FROM THE PROMPT AND CORRECTLY RETRY USING A VALID COLUMN NAME.
                    6. FOLLOW THE **CHAIN OF THOUGHTS** BEFORE PRODUCING THE FINAL ANSWER.

                    ---

                    ###CHAIN OF THOUGHTS###

                    1. **UNDERSTAND**: READ the user’s question and IDENTIFY the requested statistic or transformation.  
                    2. **BASICS**: MAP the task to a pandas method (e.g., `.mean()`, `.value_counts()`, `.sum()`, `.max()`).  
                    3. **BREAK DOWN**: ISOLATE which column(s) are needed based on the user-provided list.  
                    4. **ANALYZE**: DETERMINE the exact expression in pandas syntax to solve the task.  
                    5. **BUILD**: WRITE a single `print()` line using the correct column name(s).  
                    6. **EDGE CASES**: IF the user asks for grouping, filtering, or multiple columns, HANDLE this concisely but still within one print statement.  
                    7. **FINAL ANSWER**: OUTPUT the one-line code, THEN provide a short explanatory sentence.  

                    ---

                    ###WHAT NOT TO DO###

                    - NEVER GUESS OR MODIFY COLUMN NAMES (e.g., if user says `['Age']`, do NOT write `df['age']`).  
                    - NEVER ADD EXTRA LINES (no `import pandas`, no `df = ...`, no comments).  
                    - NEVER OUTPUT MORE THAN ONE `print()` STATEMENT.  
                    - NEVER OMIT THE HUMAN-READABLE SUMMARY.  
                    - NEVER USE FUNCTIONS OR VARIABLES NOT MENTIONED IN THE USER PROMPT.  
                    - NEVER RETURN EMPTY OR PLACEHOLDER CODE.  

                    ---

                    ###FEW-SHOT EXAMPLES###

                    **User Prompt:**  
                    "The DataFrame `df` has the following columns: ['sepalLength', 'sepalWidth', 'species']. Based on this, answer: What is the average sepalLength?"  

                    **Answer:**  
                    ```python
                    print(df['sepalLength'].mean())
                    ```  
                    The average sepalLength value is displayed above, showing the central tendency of this measurement.  

                    ---

                    **User Prompt:**  
                    "The DataFrame `df` has the following columns: ['Name', 'Age', 'Salary']. Based on this, answer: What is the maximum salary?"  

                    **Answer:**  
                    ```python
                    print(df['Salary'].max())
                    ```  
                    The result is the highest salary recorded in the dataset.  



                    """),
                                ("placeholder", "{messages}"),
            ]
        )

    @staticmethod
    @tool
    def _create_python_tool(code: str, file_path: str) -> str:
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

if __name__ == '__main__':
    print("--- Running Test for AgentSetup ---")
    
    # Create an instance of the agent setup
    agent_factory = AgentSetup(
        base_url=OLLAMA_BASE_URL, 
        model_name=LLM_MODEL
    )
    agent_executor = agent_factory.executor

    # Define a sample task and data file for testing
    # IMPORTANT: This test assumes you have 'data/iris.json' in your project root.
    test_file = 'data/iris.json'
    if not os.path.exists(test_file):
        print(f"\nERROR: Test file not found at '{test_file}'.")
        print("Please ensure the file exists to run the test.")

    test_query = f"Using the data from {test_file}, what is the average value for the 'sepalLength' column?"

    print(f"\nInvoking agent with query: '{test_query}'")

    # The input to the agent needs to match the structure of the state
    result = agent_executor.invoke({
        "messages": [("human", test_query)]
    })

    print("\n--- Agent's Final Response ---")
    # The final answer is in the 'messages' key of the output dictionary
    print(result['messages'][-1].content)
    print("\n--- Test Complete ---")


