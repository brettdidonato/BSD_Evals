from bsd_evals import BSD_Evals
from eval import Eval
from model import Model

models = [
    Model(
        model_family="Gemini",
        model_version="gemini-1.0-pro",
        service="Google AI Studio",
        max_output_tokens=2048),
    Model(
        model_family="Gemini",
        model_version="gemini-1.0-pro-001",
        service="Google Cloud",
        max_output_tokens=2048,
        temperature=0.8,
        top_k=40,
        top_p=1)
]
models = [
    Model(
        model_family="Claude",
        model_version="claude-3-haiku-20240307",
        service="Anthropic",
        max_tokens=4096,
        temperature=1.0),
    Model(
        model_family="Claude",
        model_version="claude-3-sonnet-20240229",
        service="Anthropic",
        max_tokens=4096,
        temperature=1.0),
    Model(
        model_family="Claude",
        model_version="claude-3-opus-20240229",
        service="Anthropic",
        max_tokens=4096,
        temperature=1.0),
    Model(
        model_family="Gemini",
        model_version="gemini-1.0-pro",
        service="Google AI Studio",
        max_output_tokens=2048),
    Model(
        model_family="Gemini",
        model_version="gemini-1.0-pro-001",
        service="Google Cloud",
        max_output_tokens=2048,
        temperature=0.8,
        top_k=40,
        top_p=1),
    Model(
        model_family="GPT",
        model_version="gpt-3.5-turbo",
        service="Open AI",
        temperature=1.0),
    Model(
        model_family="GPT",
        model_version="gpt-4-turbo-preview",
        service="Open AI",
        temperature=1.0),
    Model(
        model_family="GPT",
        model_version="gpt-4",
        service="Open AI",
        temperature=1.0)
]

'''
evals = [
    Eval(
        description="Basic math",
        prompt="1+2=",
        expected_response="3",
        eval_type="perfect_exact_match"
    ),
    Eval(
        description="Basic stats",
        prompt="I have two red balls and two blue balls. You take one blue ball away. How many balls do I have left? Just provide a numeric digit response (no text).",
        expected_response="3",
        eval_type="perfect_exact_match"
    )
]
'''

evals = BSD_Evals(models=models, test_eval_file="./evals/test_evals.json")
evals.run()
evals.display_results("html")