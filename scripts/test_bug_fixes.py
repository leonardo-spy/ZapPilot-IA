#!/usr/bin/env python3
"""
Testes dos bugs corrigidos + edge cases identificados na análise.

Cenários:
1. Cliente menciona Tirzec na primeira resposta → nao_tenho_tirzec + ja_usei_todas + resposta_experiente
2. Cliente menciona TG na primeira resposta → nao_tenho_tg + ja_usei_todas + resposta_experiente
3. "ah ok" após farewell → NÃO deve reiniciar flow
4. "?" (mensagem curta) → NÃO deve ser classificado como feedback_negative
5. "que problema?!" → NÃO deve disparar loop de feedback
6. Edge: "ok" durante flow ativo → não deve reiniciar
7. Edge: "quanto custa?" → deve mostrar preços
8. Edge: hesitação "vou pensar" → deve mencionar Anvisa/escassez
9. Edge: pergunta sobre orientação médica → não deve dar orientação direta
"""
import os
import requests
import sys
import time

BASE_URL = "http://localhost:8001"
DOMAIN = "tizerdral"
DELAY = int(os.environ.get("TEST_DELAY", "6"))  # seconds entre mensagens

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
# TEST 1: Cliente menciona Tirzec na PRIMEIRA resposta
# ============================================================================
def test_tirzec_first_response():
    """
    Cliente: "ola"
    Bot: abertura ("Você já utiliza...?")
    Cliente: "Vi que vc tem tizerc, quanto custa?"
    → Deve responder com nao_tenho_tirzec + ja_usei_todas + resposta_experiente
    → NÃO deve esperar até step 7 para mencionar Tirzec
    """
    name = "Tirzec na primeira resposta → respostas imediatas"
    errors = []
    cid = uid("tirzec")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    r1 = chat(cid, "ola")
    resp1 = r1.get("response", "")
    print(f"  1. ola → [{r1.get('intent')}] {resp1[:80]}")

    r2 = chat_with_delay(cid, "Vi que vc tem tizerc, quanto custa?")
    resp2 = r2.get("response", "")
    intent2 = r2.get("intent", "")
    print(f"  2. 'Vi que vc tem tizerc, quanto custa?' → [{intent2}] {resp2[:200]}")

    # Deve mencionar que Tirzec acabou
    if "tirzec" not in resp2.lower() and "acabou" not in resp2.lower():
        errors.append("Não mencionou que Tirzec acabou")
    # Deve mencionar Tizerdral como alternativa
    if "tizerdral" not in resp2.lower():
        errors.append("Não mencionou Tizerdral como alternativa")
    # Deve ter preço (500 ou 1800)
    if "500" not in resp2 and "1.800" not in resp2 and "1800" not in resp2:
        errors.append("Não mostrou preços na resposta")
    # NÃO deve ser apenas a resposta_experiente genérica (deve ter conteúdo Tirzec-específico)
    if "mesmo princípio ativo" not in resp2.lower():
        errors.append("Não mencionou 'mesmo princípio ativo' — esperado em resposta Tirzec")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 2: Cliente menciona TG na primeira resposta
# ============================================================================
def test_tg_first_response():
    """
    Cliente: "ola"
    Bot: abertura
    Cliente: "tem tg?"
    → Deve responder com nao_tenho_tg + ja_usei_todas + resposta_experiente
    """
    name = "TG na primeira resposta → respostas imediatas"
    errors = []
    cid = uid("tgfirst")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    r1 = chat(cid, "ola")
    print(f"  1. ola → [{r1.get('intent')}] {r1.get('response', '')[:80]}")

    r2 = chat_with_delay(cid, "tem tg?")
    resp2 = r2.get("response", "")
    intent2 = r2.get("intent", "")
    print(f"  2. 'tem tg?' → [{intent2}] {resp2[:200]}")

    # Deve mencionar que não tem TG
    if "não" not in resp2.lower() or "tg" not in resp2.lower():
        errors.append("Não mencionou que não tem TG")
    # Deve mencionar Tizerdral
    if "tizerdral" not in resp2.lower():
        errors.append("Não mencionou Tizerdral")
    # Deve ter info de preço
    if "500" not in resp2 and "1.800" not in resp2 and "1800" not in resp2:
        errors.append("Não mostrou preços na resposta")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 3: "ah ok" após farewell NÃO reinicia flow
# ============================================================================
def test_ah_ok_after_farewell():
    """
    Após farewell bem-sucedido, "ah ok" ou "ok então" NÃO deve reiniciar
    o flow de vendas. Deve ser tratado como out_of_scope/noise.
    """
    name = "'ah ok' após farewell não reinicia flow"
    errors = []
    cid = uid("ahok")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    # Conversa normal até farewell
    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim")

    # Farewell
    r = chat_with_delay(cid, "valeu, obrigado, tchau!")
    intent_farewell = r.get("intent", "")
    print(f"  Farewell: 'valeu, obrigado, tchau!' → intent={intent_farewell}")

    # Mensagem pós-farewell
    r2 = chat_with_delay(cid, "ah ok")
    resp2 = r2.get("response", "")
    intent2 = r2.get("intent", "")
    route2 = r2.get("route", "")
    print(f"  Pós-farewell: 'ah ok' → intent={intent2}, route={route2}")
    print(f"  Resposta: {resp2[:200]}")

    # NÃO deve reiniciar flow (sem abertura, sem greeting)
    opening_patterns = [
        "tudo bem?", "me fala uma coisa", "vou explicar melhor",
        "anvisa está barrando", "formas de pagamento",
    ]
    for pat in opening_patterns:
        if pat in resp2.lower():
            errors.append(f"REINICIOU FLOW após farewell! (detected: '{pat}')")
            break
    # Não deve ser greeting
    if intent2 == "greeting":
        errors.append("Classificou 'ah ok' como greeting após farewell")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 4: "?" NÃO deve ser feedback_negative
# ============================================================================
def test_interrogacao_nao_e_feedback():
    """
    Mensagem "?" sozinha NÃO deve ser classificada como feedback_negative.
    Deve ser out_of_scope ou tratada como ruído.
    """
    name = "'?' não é feedback_negative"
    errors = []
    cid = uid("interr")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    # Inicia conversa
    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")

    # Envia "?"
    r = chat_with_delay(cid, "?")
    intent = r.get("intent", "")
    resp = r.get("response", "")
    print(f"  MSG: '?' → intent={intent}")
    print(f"  Resposta: {resp[:200]}")

    if intent == "feedback_negative":
        errors.append("Classificou '?' como feedback_negative!")
    if "tentando" in resp.lower() and "resolver" in resp.lower():
        errors.append("Resposta de feedback handler para '?' — não deveria")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 5: "que problema?!" NÃO deve disparar feedback loop
# ============================================================================
def test_que_problema_nao_e_feedback():
    """
    "que problema?!" é uma pergunta, não feedback negativo.
    Não deve disparar o fluxo de feedback.
    """
    name = "'que problema?!' não é feedback_negative"
    errors = []
    cid = uid("qprob")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim")

    r = chat_with_delay(cid, "que problema?!")
    intent = r.get("intent", "")
    resp = r.get("response", "")
    route = r.get("route", "")
    print(f"  MSG: 'que problema?!' → intent={intent}, route={route}")
    print(f"  Resposta: {resp[:200]}")

    # Não deve ser feedback_negative
    if intent == "feedback_negative":
        errors.append("Classificou 'que problema?!' como feedback_negative!")
    # Não deve ter resposta genérica de feedback
    feedback_patterns = ["tentando resolver", "verificar seu caso", "encaminhar para"]
    if any(p in resp.lower() for p in feedback_patterns):
        errors.append("Resposta de feedback handler para pergunta — não deveria")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 6: Edge — "ok" durante flow ativo não reinicia
# ============================================================================
def test_ok_during_flow():
    """
    Mensagens curtas como "ok" durante um flow ativo não devem
    reiniciar o flow nem disparar feedback.
    """
    name = "Edge: 'ok' durante flow ativo"
    errors = []
    cid = uid("okflow")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "sou iniciante")
    chat_with_delay(cid, "ok")

    # Envia "ok" durante o flow
    r = chat_with_delay(cid, "ok")
    intent = r.get("intent", "")
    resp = r.get("response", "")
    print(f"  MSG: 'ok' → intent={intent}")
    print(f"  Resposta: {resp[:200]}")

    # Não deve reiniciar flow
    if "você já utiliza" in resp.lower():
        errors.append("REINICIOU FLOW com 'ok'!")
    # Não deve ser feedback
    if intent in ("feedback_positive", "feedback_negative"):
        errors.append(f"Classificou 'ok' como {intent}!")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 7: Edge — "quanto custa?" deve mostrar preços
# ============================================================================
def test_quanto_custa_mostra_precos():
    """
    Pergunta sobre preço deve disparar asks_payment_method
    e mostrar formas_pagamento ou pelo menos os valores.
    """
    name = "Edge: 'quanto custa?' mostra preços"
    errors = []
    cid = uid("preco")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja usei ozempic")

    r = chat_with_delay(cid, "quanto custa?")
    resp = r.get("response", "")
    print(f"  MSG: 'quanto custa?'")
    print(f"  Resposta: {resp[:300]}")

    # Deve conter valores
    if "500" not in resp:
        errors.append("Não mencionou preço da ampola (R$500)")
    if "1.800" not in resp and "1800" not in resp:
        errors.append("Não mencionou preço da caixa (R$1.800)")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 8: Edge — hesitação "vou pensar" → Anvisa/escassez
# ============================================================================
def test_hesitacao_anvisa():
    """
    Cliente hesita ("vou pensar") → deve mencionar Anvisa e escassez
    conforme playbook.
    """
    name = "Edge: hesitação → Anvisa/escassez"
    errors = []
    cid = uid("hesit")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")
    chat_with_delay(cid, "sim")
    chat_with_delay(cid, "quanto custa?")

    # Hesitação
    r = chat_with_delay(cid, "vou pensar, depois te falo")
    resp = r.get("response", "")
    print(f"  MSG: 'vou pensar, depois te falo'")
    print(f"  Resposta: {resp[:300]}")

    # Deve mencionar Anvisa ou escassez
    if "anvisa" not in resp.lower() and "escasso" not in resp.lower():
        errors.append("Não mencionou Anvisa/escassez após hesitação")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 9: Edge — orientação médica não deve ser dada
# ============================================================================
def test_sem_orientacao_medica():
    """
    Perguntas sobre dosagem, efeitos colaterais etc. não devem
    receber orientação médica direta.
    """
    name = "Edge: sem orientação médica direta"
    errors = []
    cid = uid("medic")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "sou iniciante, nunca usei nada")

    r = chat_with_delay(cid, "quantas doses devo tomar por semana?")
    resp = r.get("response", "")
    print(f"  MSG: 'quantas doses devo tomar por semana?'")
    print(f"  Resposta: {resp[:300]}")

    # Não deve dar instrução médica direta (ex: "tome X doses")
    orientacao_direta = ["tome 1", "tome uma", "deve tomar", "aplicar a cada"]
    if any(p in resp.lower() for p in orientacao_direta):
        errors.append("Deu orientação médica direta — não deveria")

    passed = len(errors) == 0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status}")
    for e in errors:
        print(f"    ⚠️  {e}")

    RESULTS.append((name, passed, errors))
    return passed


# ============================================================================
# TEST 10: Edge — mensagem muito curta "a" ou "s" durante flow
# ============================================================================
def test_mensagem_ultra_curta():
    """
    Mensagens de 1-2 caracteres durante flow ativo não devem
    ser classificadas como feedback nem reiniciar o flow.
    """
    name = "Edge: mensagem ultra-curta durante flow"
    errors = []
    cid = uid("short")

    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")

    chat(cid, "ola")
    chat_with_delay(cid, "ja utilizei")

    r = chat_with_delay(cid, "s")
    intent = r.get("intent", "")
    resp = r.get("response", "")
    print(f"  MSG: 's' → intent={intent}")
    print(f"  Resposta: {resp[:200]}")

    if intent in ("feedback_positive", "feedback_negative"):
        errors.append(f"Classificou 's' como {intent}!")

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
    print("🧪 Teste dos Bugs Corrigidos + Edge Cases")
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
        test_tirzec_first_response,
        test_tg_first_response,
        test_ah_ok_after_farewell,
        test_interrogacao_nao_e_feedback,
        test_que_problema_nao_e_feedback,
        test_ok_during_flow,
        test_quanto_custa_mostra_precos,
        test_hesitacao_anvisa,
        test_sem_orientacao_medica,
        test_mensagem_ultra_curta,
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
        print("\n  🎉 TODOS OS CENÁRIOS PASSARAM!")
    else:
        print("\n  ⚠️  Alguns cenários falharam — verificar.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()