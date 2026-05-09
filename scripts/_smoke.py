import requests, time
cid = f"smoke_{int(time.time()*1000)}"
r = requests.post("http://localhost:8001/chat", json={"customer_id": cid, "message": "ola", "domain": "tizerdral"}, timeout=30)
d = r.json()
print(f"status={r.status_code} intent={d.get('intent')} route={d.get('route')}")
print(d.get("response","")[:120])
