import os

# Set the API token using following command in PowerShell
# $env:OPENAI_API_KEY="<OPENAI_API_KEY>"

# Fetch the token
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set.")
    exit(1)

# Proceed with OpenAI API setup
import openai

openai.api_key = OPENAI_API_KEY

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, can you confirm if this token works?"}]
    )
    print("API call successful! Response:", response.choices[0].message["content"])
except Exception as e:
    print("Error during API call:", e)