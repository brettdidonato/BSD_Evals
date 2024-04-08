class Eval:
  """A machine learning model evaluation metric."""
  def __init__(
      self,
      description: str,
      prompt: str,
      expected_response: str,
      eval_type: str,
      **kwargs):
    """Initilize Eval objects.

    Args:
      description: The evaluation description e.g. "Basic math problem".
      prompt: The model prompt e.g. "1+2=".
      expected_response: The expected model response e.g. "3".
      eval_type: The evaluation type e.g. "perfect_exact_match".

    Returns:
      Model object.

    """
    self.description = description
    self.prompt = prompt
    self.expected_response = expected_response
    self.eval_type = eval_type
    self.attributes = kwargs

  def __repr__(self):
    attributes_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
    return f"Eval(description='{self.description}', eval_type='{self.eval_type}')"