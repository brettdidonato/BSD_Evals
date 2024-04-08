from model import Model

import configparser
import openai
from openai import OpenAI

config = configparser.ConfigParser()
config.read("config.ini")
OPENAI_API_KEY = config["API Keys"]["OPENAI_API_KEY"]

class OpenAIClass:
  """Prompt OpenAI GPT family of models.

  Documentation: https://github.com/openai/openai-python
  """
  def __init__(self, model: Model):
    self.client = OpenAI(api_key=OPENAI_API_KEY)
    self.model = model.model_version
    if 'temperature' in model.attributes:
      self.temperature = model.attributes['temperature']
    else:
      self.temperature = 1.0

  def generate(self, prompt: str) -> str:
    try:
      completion = self.client.chat.completions.create(
        model=self.model,
        temperature=self.temperature,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return completion.choices[0].message.content

    except openai.APIConnectionError as e:
      print("The server could not be reached")
      print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
      print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
      print("Another non-200-range status code was received")
      print(e.status_code)
      print(e.response)

    return None