#!/usr/bin/env python3
"""
Script de exploração interativa do bot.
Simula conversas reais e mostra respostas completas para diagnóstico.
Foca em cenários de agendamento de horário e continuidade de fluxo.
"""
import requests
import sys
import time
import json

BASE_URL = "http://localhost:8001"
DOMAIN = "tizerdral"


def chat(customer_id: str, message: str) -> dict:
    """Envia mensagem e retorna resposta completa."""
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"customer_id": customer_id, "message": message, "domain": DOMAIN},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def show(label: str, response: dict):
    """Mostra resposta formatada."""
    intent = response.get("intent", "?")
    route = response.get("route", "?")
    conf = response.get("confidence", 0)
    text = response.get("response", "")

    print(f"\n  [{label}] {intent} | {route} | conf: {conf*100:.0f}%")
    print(f"  {'─'*50}")
    # Mostrar primeiras 300 chars
    for line in text[:300].split("\n"):
        print(f"  │ {line}")
    if len(text) > 300:
        print(f"  │ ... ({len(text)} chars total)")
    print()


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time()*1000)}"


def scenario(name: str):
    print(f"\n{'═'*60}")
    print(f"  🧪 {name}")
    print(f"{'═'*60}")


# ============================================================================
# CENÁRIO A: Fluxo completo até agendamento (horário dentro do range)
# ============================================================================
def test_agendamento_dentro_range():
    scenario("A: Agendamento dentro do range (20:30)")
    cid = uid("sch_a")

    r = chat(cid, "ola")
    show("eu: ola", r)

    r = chat(cid, "ja utilizei, quero a caixa")
    show("eu: ja utilizei, quero a caixa", r)

    r = chat(cid, "beleza, qual horario voces entregam?")
    show("eu: qual horario voces entregam?", r)

    r = chat(cid, "as 20:30 de amanha pode ser?")
    show("eu: as 20:30 de amanha pode ser?", r)

    r = chat(cid, "perfeito, como eu pago?")
    show("eu: como eu pago?", r)

    return r


# ============================================================================
# CENÁRIO B: Agendamento fora do range (03:00)
# ============================================================================
def test_agendamento_fora_range():
    scenario("B: Agendamento fora do range (03:00)")
    cid = uid("sch_b")

    r = chat(cid, "ola")
    show("eu: ola", r)

    r = chat(cid, "ja utilizei")
    show("eu: ja utilizei", r)

    r = chat(cid, "quero a caixa, pode trazer as 3 da manha?")
    show("eu: pode trazer as 3 da manha?", r)

    # Esperar sugestão de horário alternativo
    r = chat(cid, "e de madrugada, tipo 4h?")
    show("eu: e de madrugada, tipo 4h?", r)

    r = chat(cid, "ta bom entao, pode ser as 8 da manha?")
    show("eu: pode ser as 8 da manha?", r)

    return r


# ============================================================================
# CENÁRIO C: Flow completo sem reiniciar (o bug original)
# ============================================================================
def test_flow_sem_reiniciar():
    scenario("C: Flow não reinicia após pergunta de horário")
    cid = uid("nrst")

    r = chat(cid, "ola")
    show("eu: ola", r)

    r = chat(cid, "ja utilizei")
    show("eu: ja utilizei", r)

    r = chat(cid, "sim por favor")
    show("eu: sim por favor", r)

    r = chat(cid, "ah beleza, voces teriam disponibilidade pra qual horario pra trazer aqui?")
    show("eu: disponibilidade pra qual horario?", r)

    # AQUI era onde bugava — repetia resposta
    r = chat(cid, "as 20:30 de amanha")
    show("eu: as 20:30 de amanha", r)

    # AQUI reiniciava — NÃO deve reiniciar
    r = chat(cid, "podemos marcar as 20:30 de amanha?!")
    show("eu: podemos marcar as 20:30?!", r)

    # Verificação
    reiniciou = "você já utiliza" in r.get("response", "").lower()
    if reiniciou:
        print("  ❌ FLOW REINICIOU! Bug ainda existe.")
    else:
        print("  ✅ Flow NÃO reiniciou.")

    return r


# ============================================================================
# CENÁRIO D: Progresso longo no flow — até fechamento
# ============================================================================
def test_progresso_longo():
    scenario("D: Progresso completo do flow até fechamento")
    cid = uid("long")

    r = chat(cid, "ola")
    show("eu: ola", r)

    r = chat(cid, "nunca usei, é minha primeira vez")
    show("eu: nunca usei", r)

    r = chat(cid, "sim, me explica")
    show("eu: sim, me explica", r)

    r = chat(cid, "entendi, quanto custa?")
    show("eu: quanto custa?", r)

    r = chat(cid, "quero a caixa")
    show("eu: quero a caixa", r)

    r = chat(cid, "como eu pago?")
    show("eu: como eu pago?", r)

    r = chat(cid, "pix, pode trazer amanha as 19h?")
    show("eu: pix, amanha 19h?", r)

    return r


# ============================================================================
# CENÁRIO E: Confusão do cliente ("??? que isso?")
# ============================================================================
def test_confusao_cliente():
    scenario("E: Cliente confuso questiona resposta")
    cid = uid("conf")

    r = chat(cid, "ola")
    show("eu: ola", r)

    r = chat(cid, "ja utilizei")
    show("eu: ja utilizei", r)

    r = chat(cid, "sim por favor")
    show("eu: sim por favor", r)

    r = chat(cid, "que horas posso receber?")
    show("eu: que horas posso receber?", r)

    # Se der resposta ruim, questionar
    r = chat(cid, "??? que isso? nao entendi nada")
    show("eu: ??? que isso?", r)

    reiniciou = "você já utiliza" in r.get("response", "").lower()
    if reiniciou:
        print("  ❌ REINICIOU flow ao questionar!")
    else:
        print("  ✅ Não reiniciou ao questionar.")

    return r


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("🔍 Exploração do Bot — Cenários de Agendamento")
    print(f"🌐 {BASE_URL} | 🏷️ {DOMAIN}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print("✅ Servidor online\n")
    except Exception:
        print(f"❌ Servidor offline: {BASE_URL}")
        sys.exit(1)

    scenarios_to_run = [
        test_flow_sem_reiniciar,
        test_agendamento_dentro_range,
        test_agendamento_fora_range,
        test_progresso_longo,
        test_confusao_cliente,
    ]

    for i, fn in enumerate(scenarios_to_run):
        try:
            fn()
        except Exception as e:
            print(f"  ❌ ERRO: {type(e).__name__}: {e}")

        if i < len(scenarios_to_run) - 1:
            print("\n⏳ 5s delay (rate limit)...\n")
            time.sleep(5)

    print("\n" + "=" * 60)
    print("🏁 Exploração concluída.")


if __name__ == "__main__":
    main()
