import requests

# Define the URL of the Flask application
url = 'http://127.0.0.1:5000/process'

# Define the folder path containing keypoints data
folder_path = '/keypoints'

# Make a POST request to the /process endpoint with the folder_path parameter
response = requests.post(url, data={'folder_path': folder_path})

# Print the response from the server
print(response.json())
