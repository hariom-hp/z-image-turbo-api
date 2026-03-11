import json, base64

with open('response.json') as f:
    data = json.load(f)

for k, v in data.items():
    if k != 'image':
        print(f'  {k}: {v}')

if 'image' in data:
    img = base64.b64decode(data['image'])
    with open('test_modern_result.png', 'wb') as f:
        f.write(img)
    print(f'Result saved: test_modern_result.png ({len(img)} bytes)')
elif 'detail' in data:
    print(f'Error: {data["detail"]}')
else:
    print(json.dumps(data, indent=2)[:2000])
