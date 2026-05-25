"""Example API client for SmartBite service."""

import requests
import json


def main():
    base_url = "http://localhost:8000"

    health = requests.get(f"{base_url}/health")
    print(f"Health: {health.json()}")

    with open("test_food.jpg", "rb") as f:
        files = {"file": ("test_food.jpg", f, "image/jpeg")}
        resp = requests.post(f"{base_url}/analyze", files=files)

    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {resp.status_code}")


if __name__ == "__main__":
    main()
