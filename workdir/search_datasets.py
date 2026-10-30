import urllib.request
import json

queries = ['hardhat-vest', 'safety-vest', 'construction-site-safety', 'ppe']
for q in queries:
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={q}&sortBy=relevance&page=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            print(f"=== Query: {q} ===")
            for item in data[:4]:
                ref = item.get("ref")
                title = item.get("title")
                mb = item.get("totalBytes", 0) / (1024 * 1024)
                print(f"  Ref: {ref} | Size: {mb:.1f} MB | Title: {title}")
    except Exception as e:
        print(f"Query {q} error: {e}")
