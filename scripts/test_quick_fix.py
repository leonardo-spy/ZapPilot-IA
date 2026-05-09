#!/usr/bin/env python3
"""Quick test for the 'sim por favor' fix."""
import requests
import time

CID = f"qfix_{int(time.time()*1000)}"
DOMAIN = "tizerdral"
URL = "http://localhost:8001/chat"


def chat(msg):
    r = requests.post(URL, json={"customer_id": CID, "message": msg, "domain": DOMAIN}, timeout=120)
    d = r.json()
    print(f"USER: {msg}")
    print(f"  INTENT: {d['intent']}  ROUTE: {d['route']}  CONF: {d['confidence']}")
    print(f"  BOT: {d['response'][:300]}")
    print()
    return d


# Step 1: Greeting
r1 = chat("ola")
time.sleep(2)

# Step 2: Not experienced
r2 = chat("nao utilizo")
time.sleep(2)

# Step 3: Agreement — THE FIX
r3 = chat("sim por favor")

print("=== VERIFICATION ===")
# Check step 2 was INICIANTE not EXPERIENTE
if "conhece o efeito" in r2["response"]:
    print("FAIL: Step 2 should be INICIANTE, got EXPERIENTE")
elif "tratamentos mais fortes" in r2["response"] or "tirzepatida" in r2["response"].lower():
    print("OK: Step 2 is INICIANTE")

# Check step 3 is PLAYBOOK not RAG
if r3["route"] == "playbook":
    print("OK: Step 3 route is playbook (direct flow)")
else:
    print(f"FAIL: Step 3 route is {r3['route']} (expected playbook)")

# Check step 3 has photos/explanation
resp3 = r3["response"].lower()
if "concentra" in resp3 or "seringa" in resp3 or "15mg" in resp3 or "imagem" in resp3:
    print("OK: Step 3 has explanation sequence")
else:
    print(f"WARN: Step 3 might not have explanation: {resp3[:100]}")

# Check step 3 does NOT mention TG nonsense
if "similares ao tg" in resp3:
    print("FAIL: Step 3 has nonsensical TG reference")
else:
    print("OK: Step 3 has no TG nonsense")

print("\n=== DONE ===")
