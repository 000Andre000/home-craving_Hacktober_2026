import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                 params={"key": os.getenv("LLM_API_KEY"), "pageSize": 200}).json()
for m in r.get("models", []):
    if "gemma" in m["name"] or "flash" in m["name"]:
        print(m["name"])
print(r.get("error", ""))