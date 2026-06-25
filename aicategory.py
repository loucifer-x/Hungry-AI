import requests
import json
import re

try:
    currentmodel = "phi3:mini"
except:
    with open("extraconfig.json", "r") as f:
        data = json.load(f)
        currentmodel = data["model"]

def clean_value(text):
    text = re.sub(r"^(category|subcategory)\s*[:\-]\s*", "", text.strip(), flags=re.IGNORECASE)
    text = text.split(";")[0].split(",")[0].strip()
    text = text.strip("\"'")
    return text.replace(" ", "-").lower()

def classify_text(text):
    truncated = text[:1500] if len(text) > 1500 else text
    prompt = f"""You are a precise text classifier. Analyze the text and return exactly one line:
    category | subcategory

    Rules:
    - Use the most specific and accurate category for the content
    - Do not default to generic categories like "fantasy" when more specific ones apply (e.g. "science-fiction", "sports", "politics")
    - No explanation, no labels, no extra text, just: word | word

    Text: {truncated}
    Reply:""".strip()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": currentmodel,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 20
            }
        }
    )

    output = response.json().get("response", "").strip()

    for line in output.splitlines():
        line = line.strip().strip("`\"'")
        if not line:
            continue
        if "|" in line:
            parts = line.split("|", 1)
            category = clean_value(parts[0])
            subcategory = clean_value(parts[1])
            if category and subcategory:
                print(f"{category}, {subcategory}")
                return f"{category}, {subcategory}"

        break
    return f"unknown, {output[:80]}"

