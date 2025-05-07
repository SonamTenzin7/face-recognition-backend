import requests

# Define the API endpoint URL
url = "http://localhost:5000/recognize"

# Path to your image file
file_path = r"D:\Fyp\assets\images\2 (1).jpg"

try:
    # Open the image file and send it in the POST request
    with open(file_path, "rb") as file:
        files = {"image": file}  # Ensure the key matches what the Flask app expects
        response = requests.post(url, files=files)

    # Print the response from the API
    print("Status Code:", response.status_code)
    print("Response Body:", response.json())

except Exception as e:
    print("An error occurred:", str(e))