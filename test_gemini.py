import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Check your root .env file."
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents=(
        "Explain the difference between company revenue and net income "
        "in three concise bullet points."
    ),
)

print("\n--- Gemini API response ---\n")
print(response.text)