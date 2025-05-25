import requests

def query_ollama(prompt, model="mistral"):
    """Send a prompt to Ollama's Mistral model."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json().get("response", "Error: No response from Ollama.")