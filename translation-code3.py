import requests  # Importing the necessary library

# Your Azure Translator resource details
key = "10Pr0Ti5bmjOLHKvwDn6dvaMN3YdqH3FQr2nYP24Ze0TXA7o8zKuJQQJ99BDAC4f1cMXJ3w3AAAAACOGpqR3"  # API Key
endpoint = "https://ai-ggaihub550954870170.cognitiveservices.azure.com/"  # Endpoint URL
region = "westus"  # Azure service region

# Text to translate
text = "Hello, how are you today?"
from_language = "en"  # Source language: English
to_language = "de"    # Target language: German

# Construct the request URL
path = '/translator/text/v3.0/translate'
url = endpoint + path

# Request parameters
params = {
    'api-version': '3.0',
    'from': from_language,
    'to': [to_language]
}

# Request headers
headers = {
    'Ocp-Apim-Subscription-Key': key,
    'Ocp-Apim-Subscription-Region': region,
    'Content-Type': 'application/json'
}

# Request body
body = [{
    'text': text
}]

try:
    # Making the request to Azure AI Translator
    response = requests.post(url, params=params, headers=headers, json=body)
    response.raise_for_status()  # Raise an error for unsuccessful requests

    # Parse and print the translation result
    result = response.json()
    translated_text = result[0]['translations'][0]['text']
    print(f"Translated Text: {translated_text}")  # Output translated text
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")  # Print errors if any
