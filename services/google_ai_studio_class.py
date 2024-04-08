from model import Model

import configparser
import google.generativeai as genai

config = configparser.ConfigParser()
config.read("config.ini")
GOOGLE_AI_STUDIO_API_KEY = config["API Keys"]["GOOGLE_AI_STUDIO_API_KEY"]

class GoogleAIStudioClass:
  """Prompt Google AI Studio Gemini family of models.

  Documentation: https://ai.google.dev/tutorials/quickstart
  """
  def __init__(self, model: Model):
    self.client = genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)

    generation_config = {
      "temperature": 0.9,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048,
    }

    # Documentation: https://ai.google.dev/docs/safety_setting_gemini
    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
      },
    ]

    self.model = genai.GenerativeModel(
        model_name=model.model_version,
        generation_config=generation_config,
        safety_settings=safety_settings)

  def generate(self, prompt: str) -> str:
    try:
      response = self.model.generate_content(prompt)
      #print(f"Safety candidates: {response.candidates}")
      print(response.text)
      return response.text
    except:
      return None