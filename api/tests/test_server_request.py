import requests

url = "http://localhost:8000/v2/check"
data = {
    "language": "en-US",
    "text": "This are a test sentence."
}

response = requests.post(url, json=data)
print("Status code:", response.status_code)
print("Response:", response.json())
