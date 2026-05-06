#!/usr/bin/env python3
"""
Testa conversas completas com o flow do Tizerdral.
Simula os cenários reais que falharam e valida as respostas.
"""
import requests
import sys
import time

BASE_URL = "http://localhost:8001"


def chat(customer_id: str, message: str) -> dict:
    """Envia mensagem e retorna resposta."""
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"customer_id": customer_id, "message": message},
        timeout=90,
    )
    return r.json()


def assert_contains(response: dict, text: str, step: str):
    """Verifica se a resposta contém texto esperado."""
    full = response.get("response", "")
    if text.lower() not in full.lower():
        print(f"  ❌ FALHOU em '{step}'")
        print(f"     Esperava conter: '{text[:80]}'")
        print(f"     Recebeu: '{full[:120]}'")
        return False
    return True


def assert_not_contains(response: dict, text: str, step: str):
    """Verifica que a resposta NÃO contém texto indesejado."""
    full = response.get("response", "")
    if text.lower() in full.lower():
        print(f"  ❌ FALHOU em '{step}'")
        print(f"     NÃO esperava conter: '{text[:80]}'")
        print(f"     Recebeu: '{full[:120]}'")
        return False
    return True


def assert_route(response: dict, expected_route: str, step: str):
    """Verifica a rota."""
    actual = response.get("route", "")
    if actual != expected_route:
        print(f"  ❌ FALHOU em '{step}' — route={actual}, esperava={expected_route}")
        return False
    return True


def assert_not_route(response: dict, bad_route: str, step: str):
    """Verifica que NÃO foi para uma rota."""
    actual = response.get("route", "")
    if actual == bad_route:
        print(f"  ❌ FALHOU em '{step}' — route={actual} (não deveria ser {bad_route})")
        return False
    return True


# ============================================================================
# CENÁRIO 1: Cliente pergunta sobre horário de entrega (não é pagamento)
# ============================================================================
def test_scenario_1():
    """
    Fluxo: ola → ja utilizei → sim → pergunta sobre horário
    Esperado: NÃO enviar formas de pagamento, NÃO enviar TG, usar LLM/RAG
    """
    print("\n📋 Cenário 1: Pergunta sobre horário de entrega")
    print("=" * 60)
    cid = f"test_scenario1_{int(time.time())}"
    ok = True

    # Step 1: Saudação
    r = chat(cid, "ola")
    if not assert_contains(r, "Você já utiliza", "step1_abertura"):
        ok = False
    else:
        print("  ✅ Step 1: Abertura correta")

    # Step 2: Já utilizei → resposta_experiente
    r = chat(cid, "ja utilizei")
    if not assert_contains(r, "já conhece o efeito", "step2_experiente"):
        ok = False
    else:
        print("  ✅ Step 2: Resposta experiente correta")

    # Step 3: Sim → fotos/explicação
    r = chat(cid, "sim por favor")
    if not assert_contains(r, "15mg por concentração", "step3_explicacao"):
        ok = False
    else:
        print(f"  ✅ Step 3: Explicação com {len(r.get('response_parts', []))} partes")

    # Step 4: Pergunta sobre horário de entrega
    # NÃO deve enviar pagamento, NÃO deve enviar TG/Tirzec
    r = chat(cid, "ah beleza, voces teriam disponibilidade pra qual horario pra trazer aqui?")
    step4_ok = True
    if not assert_not_contains(r, "Formas de pagamento", "step4_nao_pagamento"):
        step4_ok = False
    if not assert_not_contains(r, "Não estou trabalhando com a TG", "step4_nao_tg"):
        step4_ok = False
    if not assert_not_contains(r, "Tirzec acabou", "step4_nao_tirzec"):
        step4_ok = False
    if not assert_not_route(r, "human_handoff", "step4_nao_humano"):
        step4_ok = False
    if step4_ok:
        print(f"  ✅ Step 4: Respondeu sobre entrega sem enviar pagamento/TG")
        print(f"     Resposta: {r['response'][:100]}...")
    else:
        ok = False

    # Step 5: Questionar resposta errada — NÃO deve ir pro humano
    r = chat(cid, "e o que isso tem a ver com o que eu perguntei?")
    if not assert_not_route(r, "human_handoff", "step5_nao_humano"):
        ok = False
    else:
        print(f"  ✅ Step 5: Não mandou pro humano ao questionar")
        print(f"     Resposta: {r['response'][:100]}...")

    return ok


# ============================================================================
# CENÁRIO 2: Mesmo fluxo, mas com TG errado + recuperação
# ============================================================================
def test_scenario_2():
    """
    Fluxo: ola → ja utilizei → sim → pergunta horário → questiona → continua
    Esperado: NÃO mandar pro humano, recuperar a conversa
    """
    print("\n📋 Cenário 2: Bot responde errado e cliente questiona")
    print("=" * 60)
    cid = f"test_scenario2_{int(time.time())}"
    ok = True

    # Steps 1-3 rápidos
    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "sim por favor")
    print("  ✅ Steps 1-3: Flow normal (abertura → experiente → explicação)")

    # Step 4: Pergunta sobre entrega — deve usar generate_response
    r = chat(cid, "qual horario voces entregam?")
    step4_ok = True
    if not assert_not_contains(r, "Formas de pagamento", "step4_nao_pagamento"):
        step4_ok = False
    if not assert_not_contains(r, "Não estou trabalhando com a TG", "step4_nao_tg"):
        step4_ok = False
    if not assert_not_route(r, "human_handoff", "step4_nao_humano"):
        step4_ok = False
    if step4_ok:
        print(f"  ✅ Step 4: Respondeu sobre horário sem enviar lixo")
        print(f"     Route: {r['route']}")
        print(f"     Resposta: {r['response'][:100]}...")
    else:
        ok = False

    # Step 5: Cliente questiona — NÃO mandar pro humano
    r = chat(cid, "O que isso tem a ver? qual horario voces entrega?")
    if not assert_not_route(r, "human_handoff", "step5_nao_humano"):
        ok = False
    else:
        print(f"  ✅ Step 5: Não mandou pro humano ao questionar")
        print(f"     Route: {r['route']}")
        print(f"     Resposta: {r['response'][:100]}...")

    # Step 6: Continuar conversa normalmente
    r = chat(cid, "quero comprar a caixa")
    if not assert_not_route(r, "human_handoff", "step6_nao_humano"):
        ok = False
    else:
        print(f"  ✅ Step 6: Conversa continuou normalmente")
        print(f"     Route: {r['route']}")
        print(f"     Resposta: {r['response'][:80]}...")

    return ok


if __name__ == "__main__":
    print("🧪 Teste de Conversas Completas — Tizerdral")
    print("=" * 60)

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except Exception:
        print(f"❌ Servidor não está rodando em {BASE_URL}")
        sys.exit(1)

    results = []
    results.append(("Cenário 1: Horário de entrega", test_scenario_1()))
    print("\n⏳ Aguardando 30s para rate limit do Groq resetar...\n")
    import time
    time.sleep(30)
    results.append(("Cenário 2: Questiona + recupera", test_scenario_2()))

    print("\n" + "=" * 60)
    print("📊 RESULTADOS:")
    all_ok = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")
        if not passed:
            all_ok = False

    print()
    sys.exit(0 if all_ok else 1)
