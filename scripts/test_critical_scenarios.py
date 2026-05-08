#!/usr/bin/env python3
"""
Testes dos cenários CRÍTICOS que validam os bugs corrigidos nesta sessão.
Usa delays maiores para evitar rate limiting do Groq.

Cenários:
1. Flow completo sem reiniciar (bug principal)
2. Horário de entrega (06:00-22:00, sem dizer explícito)
3. Farewell não reinicia flow
4. goto_flow transiciona corretamente
5. Horário fora do range (03:00) → sugere alternativa
"""
import requests
import sys
import time

BASE_URL = "http://localhost:8001"
DOMAIN = "tizerdral"
DELAY = 20  # seconds entre mensagens para evitar rate limit

RESULTS: list[tuple[str, bool, list[str]]] = []


def chat(customer_id: str, message: str) -> dict:
    """Envia mensagem e retorna resposta."""
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"customer_id": customer_id, "message": message, "domain": DOMAIN},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def chat_with_delay(customer_id: str, message: str, delay: int = DELAY) -> dict:
    time.sleep(delay)
    return chat(customer_id, message)


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time()*1000)}"


# ============================================================================
# TEST 1: Flow completo sem reiniciar
# ============================================================================
def test_flow_completo_sem_reiniciar():
    """
    ola → ja utilizei → sim por favor → qual horario? → 20:30 → podemos marcar?
    Nenhuma resposta dos steps finais deve conter "Você já utiliza" (abertura).
    """
    name = "Flow completo sem reiniciar"
    errors = []
    cid = uid("flow")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    # Step 1
    r1 = chat(cid, "ola")
    resp1 = r1.get("response", "")
    print(f"  1. ola → [{r1.get('intent')}] {resp1[:80]}")
    if "você já utiliza" not in resp1.lower():
        errors.append("Step 1: esperava abertura com 'Você já utiliza'")

    # Step 2
    r2 = chat_with_delay(cid, "ja utilizei")
    resp2 = r2.get("response", "")
    print(f"  2. ja utilizei → [{r2.get('intent')}] {resp2[:80]}")
    if "500" not in resp2 and "conhece" not in resp2.lower():
        errors.append("Step 2: esperava resposta_experiente")

    # Step 3
    r3 = chat_with_delay(cid, "sim por favor")
    resp3 = r3.get("response", "")
    print(f"  3. sim por favor → [{r3.get('intent')}] {resp3[:80]}")
    if "seringa" not in resp3.lower() and "imagem" not in resp3.lower() and "foto" not in resp3.lower():
        errors.append("Step 3: esperava sequência de fotos")

    # Step 4
    r4 = chat_with_delay(cid, "qual horario voces entregam?")
    resp4 = r4.get("response", "")
    print(f"  4. qual horario? → [{r4.get('intent')}] {resp4[:80]}")
    if "você já utiliza" in resp4.lower():
        errors.append("Step 4: REINICIOU FLOW!")

    # Step 5
    r5 = chat_with_delay(cid, "as 20:30 de amanha")
    resp5 = r5.get("response", "")
    print(f"  5. 20:30 amanha → [{r5.get('intent')}] {resp5[:80]}")
    if "você já utiliza" in resp5.lower():
        errors.append("Step 5: REINICIOU FLOW!")

    # Step 6
    r6 = chat_with_delay(cid, "podemos marcar?")
    resp6 = r6.get("response", "")
    print(f"  6. podemos marcar? → [{r6.get('intent')}] {resp6[:80]}")
    if "você já utiliza" in resp6.lower():
        errors.append("Step 6: REINICIOU FLOW!")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 2: Horário de entrega (não explicitar range, não dizer 24h)
# ============================================================================
def test_horario_entrega():
    """
    Pergunta horário → não deve dizer "24 horas" nem "06-22h" explicitamente.
    Deve perguntar preferência do cliente ou aceitar horário proposto.
    """
    name = "Horário entrega (sem 24h, sem range explícito)"
    errors = []
    cid = uid("hora")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim, quero")

    # Pergunta horário
    r = chat_with_delay(cid, "qual horario voces fazem a entrega?")
    resp = r.get("response", "")
    print(f"  Pergunta: 'qual horario voces fazem a entrega?'")
    print(f"  Resposta: {resp[:200]}")

    if "24" in resp and "hora" in resp.lower():
        errors.append("Disse '24 horas' — PROIBIDO")
    if "06" in resp and "22" in resp:
        errors.append("Explicitou range 06-22h — não deveria")
    if "6h" in resp.lower() and "22h" in resp.lower():
        errors.append("Explicitou range 6h-22h — não deveria")

    # Test: propor horário dentro do range → aceitar
    r2 = chat_with_delay(cid, "as 19:00 pode ser?")
    resp2 = r2.get("response", "")
    print(f"\n  Pergunta: 'as 19:00 pode ser?'")
    print(f"  Resposta: {resp2[:200]}")

    # Não deve recusar horário dentro do range
    recusa_words = ["não consigo", "fora do horário", "não atendemos", "indisponível"]
    if any(w in resp2.lower() for w in recusa_words):
        errors.append("Recusou 19:00 — deveria aceitar (está dentro do range)")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 3: Farewell não reinicia flow
# ============================================================================
def test_farewell():
    """
    Após iniciar flow, farewell deve dar despedida sem reiniciar.
    """
    name = "Farewell não reinicia flow"
    errors = []
    cid = uid("bye")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")

    # Farewell
    r = chat_with_delay(cid, "valeu, era isso mesmo, tchau!")
    resp = r.get("response", "")
    intent = r.get("intent", "")
    print(f"  MSG: 'valeu, era isso mesmo, tchau!'")
    print(f"  Intent: {intent}")
    print(f"  Resposta: {resp[:200]}")

    if "você já utiliza" in resp.lower():
        errors.append("REINICIOU FLOW com abertura!")
    if intent != "farewell":
        errors.append(f"Intent={intent}, esperava farewell")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 4: goto_flow → fechamento_venda
# ============================================================================
def test_goto_flow():
    """
    Após step 7 (wait_response), step 8 é goto_flow fechamento_venda.
    A resposta deve ser do flow fechamento_venda.
    """
    name = "goto_flow transiciona para fechamento_venda"
    errors = []
    cid = uid("goto")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    # Avança rápido pelo flow
    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim")
    chat_with_delay(cid, "quando voces entregam?")
    chat_with_delay(cid, "as 20:00")

    # Agora em step 7 (wait_response) → próximo é goto_flow fechamento_venda
    r = chat_with_delay(cid, "vamos fechar então!")
    resp = r.get("response", "")
    route = r.get("route", "")
    print(f"  MSG: 'vamos fechar então!'")
    print(f"  Route: {route}")
    print(f"  Resposta: {resp[:300]}")

    # Deve ter route=playbook (direct_flow do fechamento)
    if route != "playbook":
        errors.append(f"Route={route}, esperava playbook (goto_flow)")

    # NÃO deve reiniciar
    if "você já utiliza" in resp.lower():
        errors.append("REINICIOU FLOW!")

    # Deve ter conteúdo de fechamento (endereço, separar, confirmar)
    fechamento_words = ["endereço", "separar", "caixa", "ampola", "confirmar", "agendar", "pegar", "maioria"]
    if not any(w in resp.lower() for w in fechamento_words):
        errors.append(f"Resposta não parece ser de fechamento_venda: '{resp[:100]}'")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 5: Horário FORA do range → sugere alternativa
# ============================================================================
def test_horario_fora_range():
    """
    Cliente propõe 03:00 da manhã → bot deve sugerir outro horário.
    """
    name = "Horário fora do range (03:00) → sugere alternativa"
    errors = []
    cid = uid("h3am")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim quero comprar")
    chat_with_delay(cid, "como funciona a entrega?")

    # Propõe horário absurdo
    r = chat_with_delay(cid, "pode ser as 3 da manha?")
    resp = r.get("response", "")
    print(f"  MSG: 'pode ser as 3 da manha?'")
    print(f"  Resposta: {resp[:300]}")

    # Não deve aceitar cegamente
    aceitou_cego = any(w in resp.lower() for w in ["perfeito, 3", "pode sim, 3", "combinado para as 3", "tá marcado"])
    if aceitou_cego:
        errors.append("Aceitou 3h da manhã sem questionar!")

    # Deve sugerir alternativa ou questionar
    sugere = any(w in resp.lower() for w in [
        "outro horário", "manhã", "tarde", "sugerir", "preferência",
        "disponível", "melhor", "não consigo", "complicado", "difícil"
    ])
    if not sugere:
        errors.append(f"Não sugeriu alternativa para 3h da manhã: '{resp[:100]}'")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("🧪 Teste dos Cenários CRÍTICOS — Bugs Corrigidos")
    print(f"🌐 Servidor: {BASE_URL}")
    print(f"⏱️  Delay entre msgs: {DELAY}s (rate limit)")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print("✅ Servidor online\n")
    except Exception:
        print(f"❌ Servidor não está rodando em {BASE_URL}")
        sys.exit(1)

    tests = [
        test_flow_completo_sem_reiniciar,
        test_horario_entrega,
        test_farewell,
        test_goto_flow,
        test_horario_fora_range,
    ]

    for i, test in enumerate(tests):
        test()
        if i < len(tests) - 1:
            print(f"\n  ⏳ Aguardando {DELAY}s (rate limit)...")
            time.sleep(DELAY)

    # Relatório final
    print(f"\n{'='*60}")
    print("📊 RESULTADO FINAL")
    print("=" * 60)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)

    for name, ok, errs in RESULTS:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} cenários passaram")

    if passed == total:
        print("\n  🎉 TODOS OS BUGS CRÍTICOS CORRIGIDOS!")
    else:
        print("\n  ⚠️  Alguns cenários falharam — verificar.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
