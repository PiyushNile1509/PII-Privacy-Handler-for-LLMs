import requests

API_KEY = "AIzaSyDJb2LPaNY5TVAYCpYmAnNnv0vQcLplQyE"

# List available models
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Available models:")
        for model in data.get('models', []):
            print(f"- {model['name']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {e}")