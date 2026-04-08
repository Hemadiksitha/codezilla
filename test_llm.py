"""Test LLM API directly."""
import httpx
import json

r = httpx.post(
    "https://a1a-52048-dev-cog-rioai-eus2-1.openai.azure.com/openai/deployments/gpt-5_2025-08-07/chat/completions?api-version=2025-01-01-preview",
    json={
        "messages": [
            {"role": "system", "content": "You are a financial analyst. Respond with valid JSON: {\"summary\":\"...\",\"detailed\":\"...\",\"key_takeaways\":[\"...\"],\"tone\":\"neutral\"}"},
            {"role": "user", "content": "Generate a 2-sentence summary for: US unemployment fell from 11% to 8% for young workers (16-24) over 2020-2025."},
        ],
        "max_completion_tokens": 10001,
    },
    headers={
        "api-key": "f57572c4e8db4f8a8ef2878cabe5fce2",
        "Content-Type": "application/json",
    },
    timeout=30,
)

print(f"Status: {r.status_code}")
data = r.json()
if r.status_code != 200:
    print(f"Error: {json.dumps(data, indent=2)}")
else:
    import json as j
    print(f"Full response: {j.dumps(data, indent=2)[:1000]}")

