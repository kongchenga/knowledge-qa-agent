"""Test live agent with DeepSeek API."""
import urllib.request, json, os

BASE = "http://localhost:8020"

def api(method, path, data=None, form=None):
    url = f"{BASE}{path}"
    headers = {}
    body = None
    if form:
        body = urllib.parse.urlencode(form).encode()
    elif data:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# 1. Stats
status, data = api("GET", "/api/stats")
print(f"Stats: {data}")

# 2. Upload text
status, data = api("POST", "/api/documents/text", form={
    "title": "Python 编程语言",
    "content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它强调代码可读性和简洁性。"
               "Python在Web开发（Django, Flask）、数据科学（NumPy, Pandas）、人工智能（TensorFlow, PyTorch）等领域广泛应用。"
               "Python的设计哲学是\"优雅、明确、简单\"。",
    "tags": "python,编程,AI",
})
print(f"Upload: {data}")

# 3. Query
status, data = api("POST", "/api/query", data={"question": "Python有哪些应用领域?", "top_k": 3})
print(f"\n=== Answer ===")
print(data["answer"])
print(f"\n=== Sources ({len(data['sources'])}) ===")
for s in data["sources"]:
    print(f"  [{s['score']}] {s['title']} ({s['source']})")
