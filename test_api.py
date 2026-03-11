#!/usr/bin/env python3
"""Test the Z-Image Interior Design API."""
import base64, requests, json, sys, time

with open("test.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"Image: {len(img_b64)} chars")

url = "https://reyanshkumar-08--z-image-interior-design-fastapi-app.modal.run/generate"
payload = {
    "image": img_b64,
    "prompt": "kids room interior design, playful, colorful, bunk bed, toys, soft rug, cartoon wall art, bright and airy",
    "num_inference_steps": 8,
    "strength": 0.7,
}

print("Sending request... (cold start can take 3-5 min)")
sys.stdout.flush()
t0 = time.time()
resp = requests.post(url, json=payload, timeout=600)
dt = time.time() - t0
print(f"Response: {resp.status_code} in {dt:.1f}s")

if resp.status_code == 200:
    data = resp.json()
    if data.get("success"):
        img_str = data["image"]
        if img_str.startswith("data:"):
            img_str = img_str.split(",", 1)[-1]
        out_bytes = base64.b64decode(img_str)
        with open("output_modern_bedroom.png", "wb") as f:
            f.write(out_bytes)
        print(f"Saved: output_modern_bedrooms.png ({data.get('width', 0)}x{data.get('height', 0)})")
        print(f"Seed: {data.get('seed')}")
        print(f"Mode: {data.get('detected_mode', 'N/A')}")
    else:
        print(f"Failure: {json.dumps(data, indent=2)[:500]}")
else:
    try:
        print(f"Error: {json.dumps(resp.json(), indent=2)[:2000]}")
    except Exception:
        print(f"Error: {resp.text[:2000]}")
