class Model:
  """A machine learning model."""
  def __init__(
      self,
      model_family: str,
      model_version: str,
      service: str,
      **kwargs):
    """Initilize Model objects.

    Args:
      model_family: The family name e.g. "gpt-4".
      model_version: The model version e.g. "gpt-3.5-turbo".
      service: The hosting service e.g. "Open AI".

    Returns:
      Model object.

    """
    self.model_family = model_family
    self.model_version = model_version
    self.service = service
    self.attributes = kwargs

  def __repr__(self):
    attributes_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
    return f"Model(model_family='{self.model_family}', model_version='{self.model_version}', service='{self.service}' {attributes_str})"