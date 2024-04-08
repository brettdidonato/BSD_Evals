# BSD_Evals
LLM evaluation framework

This project enables the creation of your own LLM evaluation framework against popular LLM providers (Anthropic, Google, OpenAI) and cloud providers (Google Cloud). Run test.py or walk through the notebook BSD_Evals.ipynb to get started. Update config.ini before running to enable APIs and services as needed. See evals/test_evals.json for an example on how to build your own set of evaluations and test.py for execution.

Once your evaluation has completed you will see an evaluation summary:

Total runtime: 65.56242179870605

Models: 8
Evals: 6
Total Evals: 48
Passed Evals: 26
Failed Evals: 14
Other Evals: 8

An evaluation matrix:
![Evaluation Matrix](evaluation_matrix.png)

And a runtime matrix:
![Runtime Matrix](runtime_matrix.png)