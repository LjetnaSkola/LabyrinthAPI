import requests

def get_solution(image, base_url, login_ep="login", solve_ep="solve"):
    # 1. Login to obtain the token
    login_url = f"{base_url}/{login_ep}"
    login_data = {
        "username": "user",
        "password": "password"
    }

    login_headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    login_response = requests.post(login_url, data=login_data, headers=login_headers)

    if login_response.status_code != 200:
        print("Login failed:", login_response.text)
        exit()

    # Extract the access token
    access_token = login_response.json().get("access_token")
    print("Access token:", access_token)

    # 2. Use token to send image to /solve
    solve_url = f"{base_url}/{solve_ep}"
    solve_headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    with open(image, "rb") as f:
        files = {
            "file": (image, f, "image/jpeg")
        }
        solve_response = requests.post(solve_url, headers=solve_headers, files=files)

    print("Status code:", solve_response.status_code)
    return solve_response.json()

solve_response = get_solution("third22.jpg", "http://127.0.0.1:8000")
print("Response:", solve_response["path"])
