import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-3n-e2b-it")


def _get_model():
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY is not configured")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(DEFAULT_GEMINI_MODEL)


async def get_gemini_suggestion(input_data):
    model = _get_model()
    prompt = f"""
You are an accessibility expert.

Element:
{input_data.element}

Issue:
{input_data.issue}

Help:
{input_data.help}

Provide a practical and helpful suggestion to improve this element.
"""
    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response")
    return text.strip()
