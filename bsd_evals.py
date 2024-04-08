from eval import Eval
from model import Model
from services.anthropic_class import AnthropicClass
from services.google_ai_studio_class import GoogleAIStudioClass
from services.google_cloud_class import GoogleCloudClass
from services.openai_class import OpenAIClass

import json
import time
import numpy as np
import pandas as pd
import IPython

from bs4 import BeautifulSoup
from decimal import Decimal
from rouge_score import rouge_scorer
from textwrap import dedent
from typing import Optional

def measure_time(func):
  """This decorator is used to measure the execution time of a function. The
  results are stored in an appropriate variable.

  It is hard coded to handle two different use cases.
  1. For measuring time to execute an LLM API call. Results are stored in the
     class instance variable runtime_matrix.
  2. For measuring the total time to execution all evalutions. Results are
     stored in the class instance variable total_runtime.
  """
  def wrapper(self, *args, **kwargs):
    start_time = time.time()
    result = func(self, *args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time

    # Update runtime matrix for specific LLM API calls
    model_index = kwargs.get('model_index', None)
    eval_index = kwargs.get('eval_index', None)

    if eval_index is not None and model_index is not None:
      self.runtime_matrix[eval_index][model_index] = runtime
      print(f"runtime: {runtime}")
    else:
      self.total_runtime = end_time - start_time

    return result
  return wrapper

class BSD_Evals:
  """Evaluate machine learning models."""
  def __init__(
      self,
      models: Optional[list[Model]] = None,
      evals: Optional[list[Eval]] = None,
      test_eval_file: Optional[str] = ""):
    if models:
      self.models = models
    else:
      self.models = []
    if evals:
      self.evals = evals
    else:
      self.evals = []
    if test_eval_file:
      self.load_test_evals(test_eval_file)

    # Results
    self.total_runtime = None
    self.results_matrix = None
    self.runtime_matrix = None

  def add_eval(self, eval: Eval) -> None:
    """Add a single evaluation to perform on the next run."""
    self.evals.append(eval)

  def add_model(self, model: Model) -> None:
    """Add a single model to perform on the next run."""
    self.models.append(model)

  def display_results(self, output_format: Optional[str] = "html") -> None:
    """Calculate and print summary evaluation results."""
    if self.results_matrix is None:
      self._init_results_matrix()
      self._init_runtime_matrix()

    self._display_summary_table(output_format)
    self._display_results_table(output_format)
    self._display_runtime_table(output_format)

  @measure_time
  def execute_prompt(
      self,
      model: list,
      prompt: list,
      model_index: int,
      eval_index: int) -> str:
    """Execute a single prompt for a given model."""
    if model.service == "Anthropic":
      g = AnthropicClass(model)
    elif model.service == "Google AI Studio":
      g = GoogleAIStudioClass(model)
    elif model.service == "Google Cloud":
      g = GoogleCloudClass(model)
    elif model.service == "Open AI":
      g = OpenAIClass(model)
    else:
      print("Model or service not found! Skipping...")
      return

    # Handle requests and retries (as needed)
    max_retries = 1
    retry_delay = 3  # seconds

    for attempt in range(max_retries + 1):
      response = g.generate(prompt)
      if response is not None:
        print(f"Model response: {response}")
        return response
      else:
        if attempt < max_retries:
          print(f"No response or error received. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
          time.sleep(retry_delay)
        else:
          print(f"Max retries reached. Skipping...")
          return None

  def load_test_evals(self, filename):
    # Read the JSON file
    with open(f"{filename}", "r") as file:
      data = json.load(file)

    # Create a list of Eval objects
    self.evals = []
    for eval_data in data:
      eval = Eval(
          description = eval_data['description'],
          prompt = eval_data['prompt'],
          expected_response = eval_data['expected_response'],
          eval_type = eval_data['eval_type'],
          database = eval_data.get('database', None)
      )
      self.evals.append(eval)

  def run_eval(
      self,
      model: Model,
      model_index: int,
      eval: Eval,
      eval_index: int) -> None:
    """Execute a single evaluation for one model and one or more prompts."""
    print("---")
    print(model)

    response = self.execute_prompt(model=model, prompt=eval.prompt, model_index=model_index, eval_index=eval_index)
    if not response:
      return

    if eval.eval_type == "perfect_exact_match":
      self._eval_perfect_match(response, eval, model_index, eval_index)
    elif eval.eval_type == "case_insensitive_match":
      self._eval_case_insensitive_match(response, eval, model_index, eval_index)
    elif eval.eval_type[:5] == "rouge":
      self._eval_rouge_score(response, eval, model_index, eval_index)
    else:
      print(f"Invalid evaluation type. Skipping...")

  @measure_time
  def run(self) -> None:
    """Iteratively step through through all evals and models."""
    self._init_results_matrix()
    self._init_runtime_matrix()

    model_cnt = len(self.models)
    eval_cnt = len(self.evals)
    total_evals_cnt = model_cnt * eval_cnt
    print(f"""Executing {eval_cnt} evals across {model_cnt} models => {total_evals_cnt} total evals.""")

    for eval_index, eval in enumerate(self.evals):
      print(dedent(f"""
        **********************************
        Evaluation #{eval_index+1}
        Evaluation description: {eval.description}
        Evaluation prompt: {eval.prompt[:1000]}
        Expected response: {eval.expected_response[:1000]}
        Evaluation type: {eval.eval_type}"""))
      for model_index, model in enumerate(self.models):
        self.run_eval(model, model_index, eval, eval_index)


    print(dedent("""
      Evaluations complete.
      **********************************"""))

  def _display_results_table(
      self,
      output_format: Optional[str] = "html") -> None:
    """Format and display full evaluation results."""
    # Generate labels for evals and models and append totals labels to each
    eval_labels = [
        f"{i+1}: " + eval.description for i, eval in enumerate(self.evals)
    ]
    eval_labels.append("Totals")
    model_labels = [f"{model.model_version} ({model.service})" for model in self.models]
    model_labels.append("Totals")

    # Add a totals row to the results matrix
    totals_row = np.sum(self.results_matrix, axis=0)
    results_matrix_with_totals = np.vstack((self.results_matrix, totals_row))

    # Add a totals column to the results matrix
    totals_column = np.sum(results_matrix_with_totals, axis=1)
    results_matrix_with_totals = np.hstack((
        results_matrix_with_totals,
        totals_column.reshape(-1, 1)
    ))

    print("\nEvaluation matrix:")

    # Convert to Pandas dataframe and add labels
    df = pd.DataFrame(
      results_matrix_with_totals,
      index=eval_labels,
      columns=model_labels
    )

    if output_format == "html":
      # Convert results matrix to HTML table
      html = df.to_html()

      # Format results matrix HTML table:
      # 1. Add green background color for passing evals
      # 2. Add red background color for failing evals
      soup = BeautifulSoup(html, 'html.parser')
      rows = soup.find_all('tr')
      num_rows = len(rows)
      num_cols = len(rows[0].find_all('td'))

      for i, row in enumerate(rows[:-1]):
          cells = row.find_all('td')
          for j, cell in enumerate(cells[:-1]):
              value = Decimal(cell.get_text().strip())
              if value == 1:
                  cell['style'] = 'background-color: green'
              elif value == 0:
                  cell['style'] = 'background-color: red'

      html = str(soup)

      # Display results matrix HTML table
      IPython.display.display(IPython.display.HTML(html))

    else: # Text Output
      # Display all columns
      pd.set_option('display.max_columns', None)

      # Enable wrapping of columns
      pd.set_option('display.expand_frame_repr', False)

      # Display results matrix console text output
      print(df)

  def _display_runtime_table(
      self,
      output_format: Optional[str] = "html") -> None:
    """Format and display full runtime evaluation results."""
    # Generate labels for evals and models and append totals labels to each
    eval_labels = [
        f"{i+1}: " + eval.description for i, eval in enumerate(self.evals)
    ]
    eval_labels.append("Totals")
    model_labels = [f"{model.model_version} ({model.service})" for model in self.models]
    model_labels.append("Totals")

    # Add a totals row to the runtime matrix
    totals_row = np.sum(self.runtime_matrix, axis=0)
    runtime_matrix_with_totals = np.vstack((self.runtime_matrix, totals_row))

    # Add a totals column to the runtime matrix
    totals_column = np.sum(runtime_matrix_with_totals, axis=1)
    runtime_matrix_with_totals = np.hstack((
        runtime_matrix_with_totals,
        totals_column.reshape(-1, 1)
    ))

    print("\nRuntime matrix:")

    # Convert to Pandas dataframe and add labels
    df = pd.DataFrame(
      runtime_matrix_with_totals,
      index=eval_labels,
      columns=model_labels
    )

    if output_format == "html":
      # Convert runtime matrix to HTML table
      html = df.to_html()

      # Display results matrix HTML table
      IPython.display.display(IPython.display.HTML(html))

    else: # Text Output
      # Display all columns
      pd.set_option('display.max_columns', None)

      # Enable wrapping of columns
      pd.set_option('display.expand_frame_repr', False)

      # Display results matrix console text output
      print(df)

  def _display_summary_table(
      self,
      output_format: Optional[str] = "html") -> None:
    """Calculate and display summary metrics."""
    print("Execution summary:")
    print(f"Total runtime: {self.total_runtime}")

    total_evals = self.results_matrix.size
    passed = np.count_nonzero(self.results_matrix == 1)
    failed = np.count_nonzero(self.results_matrix == 0)
    other = total_evals - passed - failed
    data = {
        "models_count": self.results_matrix.shape[1],
        "evals_count": self.results_matrix.shape[0],
        "total_evals": total_evals,
        "passed": passed,
        "failed": failed,
        "other": other
    }
    print(dedent(f"""
      Models: {data['models_count']}
      Evals: {data['evals_count']}
      Total Evals: {data['total_evals']}
      Passed Evals: {data['passed']}
      Failed Evals: {data['failed']}
      Other Evals: {data['other']}"""))

  def _eval_case_insensitive_match(
      self,
      response: str,
      eval: Eval,
      model_index: int,
      eval_index: int) -> bool:
    """ Evaluation method: Case Insensitive Match.

    This is the same as perfect match except that it is case insensitive.
    """
    if response.lower() == eval.expected_response.lower():
      print(f"Evaluation passed.")
      self.results_matrix[eval_index][model_index] = True
      return True
    else: # Default = False
      print(f"Evaluation failed!")
      return False

  def _eval_perfect_match(
      self,
      response: str,
      eval: Eval,
      model_index: int,
      eval_index: int) -> bool:
    """Evaluation method: Perfect Match.

    This is the most stringent evaluator.
    """
    if response == eval.expected_response:
      print(f"Evaluation passed.")
      self.results_matrix[eval_index][model_index] = True
      return True
    else: # Default = False
      print(f"Evaluation failed!")
      return False

  def _eval_rouge_score(
      self,
      response: str,
      eval: Eval,
      model_index: int,
      eval_index: int) -> float:
    """Evaluation method: Rouge Score.

    Calculates the f-measure based on whatever rouge types passed.
    Acceptable rouge types: "rouge1", "rouge2", "rougeL".
    """
    rouge_type = eval.eval_type
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(
        eval.expected_response,
        response
    )
    print(scores)
    # Retrieves F-Measure
    score = scores[rouge_type][2]
    self.results_matrix[eval_index][model_index] = score
    return score

  def _init_results_matrix(self) -> None:
    """Initialize results matrix shape and set all values to zero."""
    self.results_matrix = np.zeros((len(self.evals), len(self.models)))

  def _init_runtime_matrix(self) -> None:
    """Initialize runtime matrix shape and set all values to zero."""
    self.runtime_matrix = np.zeros((len(self.evals), len(self.models)))