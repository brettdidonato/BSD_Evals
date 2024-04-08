from model import Model

import anthropic
import configparser
import json
import os

config = configparser.ConfigParser()
config.read("config.ini")
ANTHROPIC_API_KEY = config["API Keys"]["ANTHROPIC_API_KEY"]

class AnthropicClass:
  """Prompt Anthropic Claude family of models.

  Documentation: https://github.com/anthropics/anthropic-sdk-python
  """
  def __init__(self, model: Model):
    self.client = anthropic.Anthropic(
      api_key=ANTHROPIC_API_KEY
    )
    self.model = model.model_version
    if 'max_tokens' in model.attributes:
      self.max_tokens = model.attributes['max_tokens']
    else:
      self.max_token = 4096
    if 'temperature' in model.attributes:
      self.temperature = model.attributes['temperature']
    else:
      self.temperature = 1.0

  def generate(self, prompt: str) -> str:
    try:
      message = self.client.messages.create(
          model=self.model,
          max_tokens=self.max_tokens,
          temperature=self.temperature,
          messages=[
              {"role": "user", "content": prompt}
          ]
      )
      return message.content[0].text
    except anthropic.APIConnectionError as e:
      print("The server could not be reached")
      print(e.__cause__)
    except anthropic.RateLimitError as e:
      print("A 429 status code was received; we should back off a bit.")
    except anthropic.APIStatusError as e:
      print("Another non-200-range status code was received")
      print(e.status_code)
      print(e.response)

    return None
