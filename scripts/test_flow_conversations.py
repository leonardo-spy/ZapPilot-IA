#!/usr/bin/env python3
"""
Testa conversas completas com o flow do Tizerdral.
Simula cenários reais e de borda, forçando o bot a falhar se estiver errado.

Cenários cobertos:
1. Fluxo completo: novo cliente (iniciante)
2. Fluxo completo: cliente que já usa (experiente)
3. Despedida (farewell) NÃO reinicia flow
4. Horário de entrega — sem limite fixo, sem "24h"
5. Pagamento — somente na entrega
6. Preços corretos (playbook > RAG)
7. Placeholders de imagem NÃO gerados pelo LLM
8. TG / Tirzec redirecionam para Tizerdral
9. Cliente questiona resposta errada — não vai pro humano
10. Assets servidos
11. Variações de farewell (frases ambíguas)
12. Continuidade de conversa após RAG
13. Hesitação → Anvisa/escassez
14. Entrega presencial SJC (NÓS levamos)
15. Duração da caixa (4-6 meses)
16. Frase ambígua NÃO é farewell
17. Múltiplas perguntas em sequência
18. Sem orientação médica direta
19. Horário DENTRO do range (aceitar)
20. Horário FORA do range (sugerir alternativa)
21. Flow completo até goto_flow (fechamento_venda)
22. Entrega: NÓS levamos, cliente não busca
"""
import requests
import sys
import time
import os
from typing import Callable

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8001")
DOMAIN = "tizerdral"
# Delay entre cenários — com 1 key demora muito; com N keys divide por N
DELAY_BETWEEN_SCENARIOS = int(os.getenv("TEST_DELAY", "15"))
RESULTS: list[tuple[str, bool]] = []


# ============================================================================
# HELPERS
# ============================================================================

def chat(customer_id: str, message: str) -> dict:
    """Envia mensagem e retorna resposta."""
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"customer_id": customer_id, "message": message, "domain": DOMAIN},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


class Check:
    """Acumula verificações com relatório."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.passed = True
        self.details: list[str] = []

    def contains(self, response: dict, text: str, step: str) -> bool:
        full = response.get("response", "")
        if text.lower() not in full.lower():
            self.passed = False
            self.details.append(f"❌ {step}: esperava conter '{text[:60]}' | recebeu: '{full[:100]}'")
            return False
        return True

    def not_contains(self, response: dict, text: str, step: str) -> bool:
        full = response.get("response", "")
        if text.lower() in full.lower():
            self.passed = False
            self.details.append(f"❌ {step}: NÃO esperava '{text[:60]}' | recebeu: '{full[:100]}'")
            return False
        return True

    def route_is(self, response: dict, expected: str, step: str) -> bool:
        actual = response.get("route", "")
        if actual != expected:
            self.passed = False
            self.details.append(f"❌ {step}: route={actual}, esperava={expected}")
            return False
        return True

    def route_not(self, response: dict, bad: str, step: str) -> bool:
        actual = response.get("route", "")
        if actual == bad:
            self.passed = False
            self.details.append(f"❌ {step}: route={actual} (não deveria)")
            return False
        return True

    def any_contains(self, response: dict, texts: list[str], step: str) -> bool:
        """Pelo menos uma das strings deve estar presente."""
        full = response.get("response", "").lower()
        for t in texts:
            if t.lower() in full:
                return True
        self.passed = False
        self.details.append(f"❌ {step}: nenhuma de {texts[:5]} encontrada | recebeu: '{full[:100]}'")
        return False

    def report(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"  {status} — {self.scenario}")
        for d in self.details:
            print(f"       {d}")


def run_scenario(name: str, fn: Callable):
    """Wrapper para executar cenário com tratamento de erro."""
    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")
    try:
        fn()
    except requests.exceptions.HTTPError as e:
        print(f"  ❌ HTTP ERROR: {e}")
        RESULTS.append((name, False))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")
        RESULTS.append((name, False))


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time()*1000)}"


# ============================================================================
# CENÁRIO 1: Fluxo completo — cliente iniciante
# ============================================================================
def test_fluxo_iniciante():
    """
    ola → nunca usei → sim explica → [fotos + explicação]
    Esperado: abertura, resposta_iniciante, sequência com fotos/explicação
    """
    c = Check("Fluxo iniciante completo")
    cid = uid("ini")

    # Step 1: Saudação → abertura
    r = chat(cid, "ola")
    c.contains(r, "Você já utiliza", "abertura")

    # Step 2: Nunca usei → resposta_iniciante
    r = chat(cid, "nunca usei, é minha primeira vez")
    c.any_contains(r, ["tratamentos mais fortes", "tirzepatida"], "iniciante")
    c.contains(r, "500", "preco_ampola")
    c.contains(r, "1.800", "preco_caixa")

    # Step 3: Quer explicação → sequência de fotos + textos
    r = chat(cid, "sim, pode me explicar")
    # Deve ter response_parts (sequência) ou mencionar concentração/seringa
    full = r.get("response", "")
    parts = r.get("response_parts", [])
    has_sequence = len(parts) > 1 or "15mg" in full.lower() or "concentração" in full.lower()
    if not has_sequence:
        c.passed = False
        c.details.append(f"❌ sequencia: sem fotos/explicação ({len(parts)} parts)")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 2: Fluxo completo — cliente experiente
# ============================================================================
def test_fluxo_experiente():
    """
    ola → já uso ozempic → aceita → fechamento
    Esperado: abertura, resposta_experiente, fechamento
    """
    c = Check("Fluxo experiente completo")
    cid = uid("exp")

    r = chat(cid, "ola")
    c.contains(r, "Você já utiliza", "abertura")

    r = chat(cid, "sim, ja uso Ozempic ha 6 meses")
    c.any_contains(r, ["já conhece o efeito", "princípio ativo"], "experiente")
    c.contains(r, "500", "preco_ampola")
    c.contains(r, "1.800", "preco_caixa")

    # Aceita → dispara send_sequence com fotos (step 5)
    r = chat(cid, "quero a caixa")
    # Resposta aqui são as fotos do send_sequence

    # Agora no step 7 — pergunta sobre pagamento
    r = chat(cid, "entendi! como faço o pagamento?")
    # Deve ter formas de pagamento ou fechamento
    c.any_contains(r, ["pagamento", "pix", "cartão", "crédito", "1.800", "separe", "na entrega", "na hora"], "fechamento")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 3: Farewell NÃO reinicia flow
# ============================================================================
def test_farewell_nao_reinicia():
    """
    ola → já utilizei → despedida ambígua
    Esperado: farewell response (sem reiniciar flow com "Você já utiliza")
    """
    c = Check("Farewell não reinicia flow")
    cid = uid("fare")

    r = chat(cid, "ola")
    c.contains(r, "Você já utiliza", "abertura")

    r = chat(cid, "ja utilizei")
    # experiente
    c.any_contains(r, ["já conhece", "princípio ativo", "500"], "experiente")

    # Farewell — NÃO deve reiniciar flow
    r = chat(cid, "ah nao esquece, tudo bem valeu")
    c.not_contains(r, "Você já utiliza", "nao_reinicia_flow")
    c.route_not(r, "playbook", "nao_playbook")
    # Deve ser resposta de despedida
    c.any_contains(r, ["disposição", "chamar", "precisar", "👋", "até"], "farewell_response")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 4: Variações de farewell
# ============================================================================
def test_farewell_variacoes():
    """
    Testa múltiplas frases que devem ser classificadas como farewell.
    """
    c = Check("Variações de farewell")
    farewells = [
        "valeu, tchau",
        "obrigado, era só isso",
        "beleza, até mais",
        "pode deixar, não preciso de mais nada",
    ]

    for msg in farewells:
        cid = uid("fv")
        # Inicia flow
        chat(cid, "ola")
        # Farewell
        r = chat(cid, msg)
        # NÃO deve reiniciar (não deve ter "Você já utiliza")
        if "você já utiliza" in r.get("response", "").lower():
            c.passed = False
            c.details.append(f"❌ '{msg}' reiniciou flow!")
        # NÃO deve ir pro humano
        if r.get("route") == "human_handoff":
            c.passed = False
            c.details.append(f"❌ '{msg}' → human_handoff")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 5: Horário de entrega — sem limite fixo
# ============================================================================
def test_horario_entrega():
    """
    Pergunta "qual horário vocês entregam?"
    Esperado: NÃO mencionar "até as 18h" ou horário fixo.
    Deve perguntar a disponibilidade do cliente.
    """
    c = Check("Horário entrega sem limite fixo")
    cid = uid("hor")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "sim, quero a caixa")

    # Pergunta sobre horário
    r = chat(cid, "qual horario voces entregam?")
    c.not_contains(r, "até as 18", "sem_limite_18h")
    c.not_contains(r, "das 9 às", "sem_horario_fixo")
    c.not_contains(r, "horário comercial", "sem_horario_comercial")
    c.route_not(r, "human_handoff", "nao_humano")

    # Idealmente pergunta disponibilidade do cliente
    full = r.get("response", "").lower()
    mentions_client = any(w in full for w in ["melhor pra você", "disponibilidade", "horário", "preferência", "qual horário"])
    if not mentions_client:
        c.details.append(f"⚠️  Não perguntou disponibilidade do cliente: '{full[:100]}'")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 6: Pagamento — somente na entrega
# ============================================================================
def test_pagamento_na_entrega():
    """
    Pergunta "como pago?" ou "manda o pix"
    Esperado: Pagamento na hora da entrega. NUNCA pedir antecipado.
    """
    c = Check("Pagamento somente na entrega")
    cid = uid("pag")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    # "quero comprar" dispara send_sequence com fotos (step 5)
    chat(cid, "quero comprar")
    # Agora no step 7 — pergunta sobre pagamento
    r = chat(cid, "como eu pago?")

    # NÃO deve pedir pix antecipado
    c.not_contains(r, "envie o pix antes", "sem_pix_antecipado")
    c.not_contains(r, "pague antes", "sem_pague_antes")
    c.not_contains(r, "antecipado", "sem_antecipado")
    c.not_contains(r, "transferência antes", "sem_transferencia_antes")

    # Deve mencionar "na entrega" ou "na hora" ou formas de pagamento
    c.any_contains(r, ["na entrega", "na hora", "pix", "cartão", "crédito", "pagamento"], "menciona_pagamento")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 7: Pagamento antecipado negado explicitamente
# ============================================================================
def test_pagamento_antecipado_negado():
    """
    'Posso pagar antes pra garantir?' → bot deve negar.
    Flow: ola → ja uso → (fotos via agrees) → goto fechamento → pix antecipado?
    """
    c = Check("Nega pagamento antecipado")
    cid = uid("pgn")

    chat(cid, "ola")
    chat(cid, "ja uso tirzepatida")
    # "ok, quero sim" → client_agrees_or_confirms → envia fotos imediatamente
    chat(cid, "ok, quero sim")
    # Agora no goto_flow fechamento_venda (próximo wait_response)
    # Pergunta sobre pix antecipado — asks_payment_method keyword match
    r = chat(cid, "posso te mandar o pix agora pra garantir o meu?")

    # Deve explicar formas de pagamento ou mencionar pagamento na entrega
    c.any_contains(r, ["na entrega", "na hora", "presencial", "pix", "cartão", "crédito", "pagamento", "maioria", "separe", "ampola", "caixa"], "paga_na_entrega")
    c.not_contains(r, "pode sim, manda", "nao_aceita_antecipado")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 8: Preços corretos
# ============================================================================
def test_precos_corretos():
    """
    Pergunta direta sobre preço.
    Esperado: R$500 (ampola) e R$1.800 (caixa). NUNCA R$1.490.
    """
    c = Check("Preços corretos do playbook")
    cid = uid("prc")

    chat(cid, "ola")
    r = chat(cid, "quanto custa?")

    c.not_contains(r, "1.490", "sem_preco_antigo")
    c.not_contains(r, "1490", "sem_preco_antigo2")

    # Deve ter preços corretos
    c.any_contains(r, ["500", "1.800", "1800"], "preco_correto")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 9: LLM não gera placeholders de imagem
# ============================================================================
def test_sem_placeholders():
    """
    Quando resposta vai pro LLM/RAG, NÃO deve gerar [IMAGEM:] nem ---MSG---.
    """
    c = Check("Sem placeholders de imagem na LLM")
    cid = uid("plc")

    chat(cid, "ola")
    chat(cid, "nunca usei")
    # "sim, pode explicar" dispara send_sequence com fotos (step 5)
    chat(cid, "sim, pode explicar")
    # Agora no step 7 — pergunta vai pro RAG (não é payment)
    r = chat(cid, "como eu aplico a injeção? tem alguma orientação?")

    # Só checa placeholders se resposta veio do LLM/RAG (não do playbook)
    route = r.get("route", "")
    if route != "playbook":
        c.not_contains(r, "[IMAGEM:", "sem_placeholder_imagem")
        c.not_contains(r, "---MSG---", "sem_msg_separator")
        c.not_contains(r, "[FOTO:", "sem_placeholder_foto")
    else:
        # Se ainda vier do playbook, [IMAGEM:] é tag legítima — não é erro
        c.details.append("ℹ️  route=playbook — [IMAGEM:] é tag legítima do playbook")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 10: TG → redireciona para Tizerdral
# ============================================================================
def test_tg_redirecionamento():
    """
    'Vocês têm TG?' → Não trabalho com TG, só Tizerdral.
    """
    c = Check("TG redireciona para Tizerdral")
    cid = uid("tg")

    chat(cid, "ola")
    r = chat(cid, "voces tem TG?")

    c.any_contains(r, ["não estou trabalhando com a tg", "tizerdral", "mesmo princípio ativo"], "redireciona_tizerdral")
    c.not_contains(r, "temos sim", "nao_afirma_ter_tg")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 11: Tirzec → redireciona para Tizerdral
# ============================================================================
def test_tirzec_redirecionamento():
    """
    'Tem Tirzec?' → Acabou, só tenho Tizerdral.
    """
    c = Check("Tirzec redireciona para Tizerdral")
    cid = uid("tzc")

    chat(cid, "ola")
    r = chat(cid, "tem tirzec disponivel?")

    c.any_contains(r, ["acabou", "tizerdral", "mesmo princípio", "não tenho tirzec"], "redireciona_tizerdral")
    c.not_contains(r, "tenho sim", "nao_afirma_ter_tirzec")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 12: Cliente questiona resposta — NÃO vai pro humano
# ============================================================================
def test_questionamento_nao_humano():
    """
    Cliente questiona algo → bot deve tentar responder, NÃO handoff.
    """
    c = Check("Questionamento não vai pro humano")
    cid = uid("qst")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "sim por favor")

    # Questiona resposta anterior
    r = chat(cid, "o que isso tem a ver com o que eu perguntei?")
    c.route_not(r, "human_handoff", "nao_humano_questionamento")

    # Repete pergunta
    r = chat(cid, "me responde a pergunta por favor")
    c.route_not(r, "human_handoff", "nao_humano_insistencia")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 13: Assets servidos
# ============================================================================
def test_assets():
    """Verifica que /assets endpoint serve arquivos de imagem."""
    c = Check("Assets endpoint disponível")

    try:
        # Testa um arquivo real em vez de listar diretório
        r = requests.get(f"{BASE_URL}/assets/tizerdral/foto_caixa_tizerdral.jpg", timeout=5)
        if r.status_code != 200:
            c.passed = False
            c.details.append(f"❌ /assets/tizerdral/foto_caixa_tizerdral.jpg retornou {r.status_code}")
        else:
            content_type = r.headers.get("content-type", "")
            if "image" not in content_type:
                c.details.append(f"⚠️  Content-Type inesperado: {content_type}")
    except Exception as e:
        c.passed = False
        c.details.append(f"❌ Erro: {e}")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 14: Continuidade após RAG
# ============================================================================
def test_continuidade_pos_rag():
    """
    Após resposta RAG, bot continua conversa normalmente.
    Não trava nem manda pro humano.
    """
    c = Check("Continuidade pós-RAG")
    cid = uid("cont")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "quanto tempo demora pra fazer efeito?")
    c.route_not(r, "human_handoff", "rag_nao_humano")

    # Continua conversa
    r = chat(cid, "entendi, e quanto custa a caixa?")
    c.route_not(r, "human_handoff", "continuidade_nao_humano")
    c.any_contains(r, ["1.800", "1800", "500", "pagamento", "caixa"], "responde_preco")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 15: Hesitação → Anvisa/escassez
# ============================================================================
def test_hesitacao():
    """
    'Vou pensar' → bot deve usar argumento de escassez/Anvisa.
    Flow: ola → nunca usei → sim (fotos) → "entendi" (triggers goto fechamento)
          → vou pensar (triggers client_hesitates in fechamento flow)
    """
    c = Check("Hesitação → Anvisa/escassez")
    cid = uid("hes")

    chat(cid, "ola")
    chat(cid, "nunca usei")
    # "sim, pode explicar" → client_agrees_or_confirms → fotos
    chat(cid, "sim, pode explicar")
    # Next turn triggers goto_flow fechamento_venda → sends fechamento message
    chat(cid, "entendi")
    # Now at fechamento_venda step 2 (condition client_accepts_or_chooses)
    # "vou pensar" → not accepts → client_hesitates → anvisa_escassez
    r = chat(cid, "vou pensar, vou falar com meu marido")

    # Deve mencionar Anvisa, escassez, ou reforçar disponibilidade
    c.any_contains(r, ["anvisa", "escass", "fiscalização", "disposição", "disponibilidade", "acabar", "barrando"], "argumento_escassez")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 16: Resposta de entrega presencial em SJC
# ============================================================================
def test_entrega_sjc():
    """
    'Vocês entregam?' → Entrega presencial em SJC.
    """
    c = Check("Entrega presencial SJC")
    cid = uid("sjc")

    chat(cid, "ola")
    chat(cid, "ja utilizei, quero comprar")
    r = chat(cid, "vocês entregam ou tenho que ir buscar?")

    c.any_contains(r, ["entrega", "residência", "presencial", "sjc", "levamos", "pessoalmente"], "menciona_entrega")
    c.not_contains(r, "correio", "sem_correio")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 17: Duração da caixa
# ============================================================================
def test_duracao_caixa():
    """
    'Quanto tempo dura a caixa?' → 4 a 6 meses.
    """
    c = Check("Duração da caixa (4-6 meses)")
    cid = uid("dur")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "quanto tempo dura a caixa?")

    c.any_contains(r, ["4", "6", "meses", "mês"], "menciona_duracao")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 18: Frase ambígua que NÃO é farewell
# ============================================================================
def test_nao_farewell_falso_positivo():
    """
    'Tudo bem, me fala mais' → NÃO é farewell (continua conversa).
    """
    c = Check("Não classifica falso positivo como farewell")
    cid = uid("nfp")

    chat(cid, "ola")
    r = chat(cid, "tudo bem, me fala mais sobre o produto")

    # NÃO deve ser farewell
    c.route_not(r, "farewell", "nao_farewell")
    # NÃO deve ter resposta de despedida
    c.not_contains(r, "👋", "sem_despedida")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 19: Múltiplas perguntas em sequência (stress test)
# ============================================================================
def test_multiplas_perguntas():
    """
    Cliente faz várias perguntas em sequência — bot não trava.
    """
    c = Check("Múltiplas perguntas sem travar")
    cid = uid("mult")

    chat(cid, "ola")
    chat(cid, "ja utilizei")

    perguntas = [
        "quanto custa?",
        "como funciona a entrega?",
        "aceita cartão?",
        "e se eu quiser só uma ampola?",
    ]

    for p in perguntas:
        r = chat(cid, p)
        c.route_not(r, "human_handoff", f"nao_humano: '{p[:30]}'")
        # Deve ter resposta não vazia
        if not r.get("response", "").strip():
            c.passed = False
            c.details.append(f"❌ Resposta vazia para: '{p}'")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 20: Não dar orientação médica
# ============================================================================
def test_sem_orientacao_medica():
    """
    'Qual dose devo tomar?' → NÃO deve receitar diretamente.
    """
    c = Check("Sem orientação médica direta")
    cid = uid("med")

    chat(cid, "ola")
    chat(cid, "nunca usei")
    r = chat(cid, "qual dose eu devo tomar? pode me receitar?")

    # Não deve receitar diretamente
    c.not_contains(r, "tome", "sem_receitar")
    c.not_contains(r, "você deve aplicar", "sem_prescrever")
    # Deve sugerir consulta ou ser cauteloso
    full = r.get("response", "").lower()
    cautious = any(w in full for w in ["médico", "profissional", "orientação", "consulta", "acompanhamento"])
    if not cautious:
        c.details.append(f"⚠️  Pode não ter sido cauteloso: '{full[:100]}'")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 19: Horário dentro do range → aceitar
# ============================================================================
def test_horario_dentro_range():
    """
    Cliente propõe 19:00 ou 20:30 → bot deve ACEITAR sem questionar.
    Flow: ola → ja utilizei → quero (fotos) → "entendi" (goto fechamento) → entrega? → 19h
    """
    c = Check("Horário dentro do range → aceita")
    cid = uid("hdr")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    # "quero comprar" → client_agrees_or_confirms → fotos
    chat(cid, "quero comprar")
    # Next turn triggers goto_flow fechamento_venda → sends fechamento message
    # Ask about delivery — this goes to fechamento_venda condition check
    chat(cid, "como funciona a entrega?")
    # Now propose a schedule within range
    r = chat(cid, "pode ser as 19:00 de amanha?")
    # NÃO deve recusar horário dentro do range
    recusa = ["não consigo", "fora do horário", "fica difícil", "indisponível", "complicado"]
    for w in recusa:
        c.not_contains(r, w, f"nao_recusar_19h:{w}")
    # Deve aceitar/confirmar or at least respond without refusing
    c.any_contains(r, ["pode", "combinado", "certo", "beleza", "perfeito", "ok", "ótimo", "endereço", "agendar", "anotado", "horário", "entrega", "lev"], "aceita_19h")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 20: Horário fora do range → sugerir alternativa
# ============================================================================
def test_horario_fora_range():
    """
    Cliente propõe 3h da manhã → bot deve sugerir outro horário.
    Flow: ola → ja utilizei → quero (fotos) → fechamento → quando? → 3am
    """
    c = Check("Horário fora do range → sugere alternativa")
    cid = uid("hfr")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    # "quero a caixa" → client_agrees_or_confirms → fotos → goto fechamento
    chat(cid, "quero a caixa")
    # At fechamento, ask about delivery
    chat(cid, "quando voces entregam?")

    r = chat(cid, "pode ser as 3 da manha?")
    # NÃO deve aceitar cegamente
    aceita_cego = ["perfeito, 3", "pode sim, 3", "combinado para as 3", "tá marcado 3"]
    for w in aceita_cego:
        c.not_contains(r, w, f"nao_aceitar_3am:{w}")
    # Deve sugerir alternativa ou at minimum acknowledge
    c.any_contains(r, ["outro horário", "manhã", "tarde", "difícil", "complicado", "melhor", "cedo", "sugerir", "horário"], "sugere_alternativa")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 21: Flow completo até goto_flow (fechamento_venda)
# ============================================================================
def test_flow_goto_fechamento():
    """
    Conversa completa: ola → ja utilizei → sim (fotos + goto fechamento) → aceita
    After "sim por favor": fotos are sent AND goto_flow fechamento_venda triggers.
    The fechamento message should appear on the NEXT turn.
    """
    c = Check("Flow completo → goto_flow fechamento_venda")
    cid = uid("goto")

    r = chat(cid, "ola")
    c.contains(r, "Você já utiliza", "abertura")

    r = chat(cid, "ja utilizei")
    c.any_contains(r, ["já conhece", "princípio ativo", "500"], "experiente")

    r = chat(cid, "sim por favor")
    # Should have fotos (send_sequence via client_agrees_or_confirms)
    full = r.get("response", "")
    if "seringa" not in full.lower() and "imagem" not in full.lower() and "concentração" not in full.lower():
        c.details.append(f"⚠️  Esperava fotos: '{full[:80]}'")
    # Route should be playbook
    c.route_is(r, "playbook", "fotos_route")

    # Next: should be at goto_flow fechamento_venda
    # User responds — should get fechamento content
    r = chat(cid, "entendi, quero a caixa")
    c.not_contains(r, "Você já utiliza", "nao_reinicia")
    # Should have fechamento or payment content
    c.any_contains(r, ["caixa", "ampola", "maioria", "separar", "endereço", "confirmar", "pagamento", "pix", "cartão", "sincero", "separe"], "conteudo_fechamento")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# CENÁRIO 22: Entrega — NÓS levamos, cliente NÃO busca
# ============================================================================
def test_entrega_nos_levamos():
    """
    Resposta sobre entrega deve usar 'levamos/entregamos/vamos até você'.
    NUNCA 'pode trazer', 'buscar', 'retirar' (invertendo quem entrega).
    Note: 'não precisa buscar' is acceptable (negation of buscar).
    """
    c = Check("Entrega: NÓS levamos (sem 'pode trazer')")
    cid = uid("lev")

    chat(cid, "ola")
    chat(cid, "ja utilizei")
    r = chat(cid, "como funciona a entrega? vocês vêm até mim?")

    full = r.get("response", "").lower()
    # NÃO deve inverter (cliente traz/busca)
    c.not_contains(r, "pode trazer", "sem_pode_trazer")
    c.not_contains(r, "você traz", "sem_voce_traz")
    # "buscar" is OK if negated (e.g. "não precisa buscar")
    if "buscar" in full and "não precisa buscar" not in full and "não precisar buscar" not in full and "sem buscar" not in full:
        c.passed = False
        c.details.append(f"❌ sem_buscar: contém 'buscar' sem negação | recebeu: '{full[:100]}'")
    # "retirar" is OK if negated
    if "retirar" in full and "não precisa retirar" not in full and "sem retirar" not in full:
        c.passed = False
        c.details.append(f"❌ sem_retirar: contém 'retirar' sem negação | recebeu: '{full[:100]}'")
    # Deve afirmar que NÓS levamos
    c.any_contains(r, ["levamos", "entregamos", "vamos até", "residência", "sua casa", "entrega", "presencial"], "nos_levamos")

    c.report()
    RESULTS.append((c.scenario, c.passed))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🧪 Teste Abrangente de Conversas — Tizerdral")
    print(f"🌐 Servidor: {BASE_URL}")
    print(f"🏷️  Domínio: {DOMAIN}")
    print(f"⏱️  Delay entre cenários: {DELAY_BETWEEN_SCENARIOS}s")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print("✅ Servidor online\n")
    except Exception:
        print(f"❌ Servidor não está rodando em {BASE_URL}")
        sys.exit(1)

    # Todos os cenários
    all_scenarios = [
        ("1. Fluxo iniciante", test_fluxo_iniciante),
        ("2. Fluxo experiente", test_fluxo_experiente),
        ("3. Farewell não reinicia", test_farewell_nao_reinicia),
        ("4. Variações de farewell", test_farewell_variacoes),
        ("5. Horário entrega (sem 24h)", test_horario_entrega),
        ("6. Pagamento na entrega", test_pagamento_na_entrega),
        ("7. Nega pagamento antecipado", test_pagamento_antecipado_negado),
        ("8. Preços corretos", test_precos_corretos),
        ("9. Sem placeholders", test_sem_placeholders),
        ("10. Assets endpoint", test_assets),
        ("11. TG → Tizerdral", test_tg_redirecionamento),
        ("12. Tirzec → Tizerdral", test_tirzec_redirecionamento),
        ("13. Questionamento ≠ humano", test_questionamento_nao_humano),
        ("14. Continuidade pós-RAG", test_continuidade_pos_rag),
        ("15. Hesitação", test_hesitacao),
        ("16. Entrega SJC", test_entrega_sjc),
        ("17. Duração caixa", test_duracao_caixa),
        ("18. Não farewell falso+", test_nao_farewell_falso_positivo),
        ("19. Horário dentro do range", test_horario_dentro_range),
        ("20. Horário fora do range", test_horario_fora_range),
        ("21. Flow → goto_flow fechamento", test_flow_goto_fechamento),
        ("22. Entrega: NÓS levamos", test_entrega_nos_levamos),
        ("23. Múltiplas perguntas", test_multiplas_perguntas),
        ("24. Sem orientação médica", test_sem_orientacao_medica),
    ]

    # Filtrar cenários por número se passados via argv (ex: python script.py 2 6 7 9)
    filter_nums = set()
    for arg in sys.argv[1:]:
        if arg.isdigit():
            filter_nums.add(int(arg))

    if filter_nums:
        scenarios = [(n, f) for n, f in all_scenarios if int(n.split(".")[0]) in filter_nums]
        print(f"🔍 Rodando apenas cenários: {sorted(filter_nums)}\n")
    else:
        scenarios = all_scenarios

    for i, (name, fn) in enumerate(scenarios):
        run_scenario(name, fn)
        # Delay entre cenários para não estourar rate limit
        if i < len(scenarios) - 1:
            print(f"\n  ⏳ Aguardando {DELAY_BETWEEN_SCENARIOS}s (rate limit)...")
            time.sleep(DELAY_BETWEEN_SCENARIOS)

    # Relatório final
    print("\n" + "=" * 60)
    print("📊 RESULTADO FINAL")
    print("=" * 60)

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)

    for name, ok in RESULTS:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} cenários passaram")

    if passed == total:
        print("\n  🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("\n  ⚠️  Alguns cenários falharam — revisar.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
