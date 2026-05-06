"""
Testes de classificação de intent — valida que o LLM classifica corretamente.
Usa a API real (Groq) para testar com o prompt atual.

Uso:
    python scripts/test_classification.py              # Roda todos os testes
    python scripts/test_classification.py --verbose     # Mostra detalhes
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agent.prompts import get_intent_classification_prompt
from llm.providers import get_default_provider

# ==================== CASOS DE TESTE ====================
# (mensagem, intent_esperado, description)
TEST_CASES = [
    # --- Saudação ---
    ("oi", "greeting", "Cumprimento simples"),
    ("ola", "greeting", "Cumprimento 'ola'"),
    ("olá", "greeting", "Cumprimento 'olá'"),
    ("bom dia", "greeting", "Bom dia"),
    ("boa tarde", "greeting", "Boa tarde"),
    ("boa noite", "greeting", "Boa noite"),
    ("eae", "greeting", "Cumprimento informal"),
    ("oi, tudo bem?", "greeting", "Cumprimento com pergunta social"),

    # --- Venda ---
    ("quanto custa?", "sales", "Pergunta preço"),
    ("qual o valor?", "sales", "Pergunta valor"),
    ("quero comprar", "sales", "Quer comprar"),
    ("tem disponível?", "sales", "Pergunta disponibilidade"),
    ("aceita pix?", ["sales", "billing"], "Forma de pagamento"),
    ("quero a caixa", "sales", "Escolhe produto"),
    ("me manda o pix", ["sales", "billing"], "Pede dados de pagamento"),

    # --- Informação ---
    ("como funciona?", "info", "Pergunta genérica"),
    ("o que é tirzepatida?", "info", "Pergunta sobre produto"),
    ("pra que serve?", "info", "Pergunta utilidade"),

    # --- Suporte ---
    ("não tá funcionando", "support", "Problema técnico"),
    ("como aplica?", "support", "Dúvida de uso"),
    ("tá dando erro", "support", "Relato de erro"),
    ("como conservar?", "support", "Dúvida conservação"),

    # --- Feedback positivo ---
    ("deu certo, obrigado!", "feedback_positive", "Agradecimento"),
    ("funcionou perfeitamente", "feedback_positive", "Confirmação positiva"),
    ("valeu, resolveu!", "feedback_positive", "Feedback positivo"),

    # --- Feedback negativo ---
    ("não resolveu", "feedback_negative", "Não resolveu"),
    ("continua com problema", "feedback_negative", "Problema persiste"),

    # --- Humano (SOMENTE pedido explícito) ---
    ("quero falar com um atendente", "human", "Pedido de atendente"),
    ("me passa pro gerente", "human", "Pedido de gerente"),
    ("tem alguém humano aí?", "human", "Pedido explícito de humano"),

    # --- Fora da base ---
    ("qual a previsão do tempo?", "out_of_scope", "Pergunta irrelevante"),
    ("me conta uma piada", "out_of_scope", "Pedido irrelevante"),

    # --- Respostas curtas (NÃO devem ser humano) ---
    ("ok", ["greeting", "feedback_positive"], "Resposta curta 'ok'"),
    ("sim", ["greeting", "sales"], "Resposta curta 'sim'"),
    ("beleza", ["greeting", "feedback_positive"], "Resposta curta 'beleza'"),
]


def _parse_json(text: str) -> dict:
    """Extrai JSON de resposta do LLM."""
    text = text.strip()
    # Tentar extrair JSON de code block
    if "```" in text:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    # Tentar extrair JSON direto
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


def run_tests(verbose: bool = False):
    """Roda todos os testes de classificação."""
    llm = get_default_provider()
    prompt = get_intent_classification_prompt()

    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"  Testes de Classificação de Intent")
    print(f"  LLM: {llm.name()}")
    print(f"  Total: {len(TEST_CASES)} casos")
    print(f"{'='*60}\n")

    for msg, expected, desc in TEST_CASES:
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": msg},
            ]
            response = llm.chat(messages, temperature=0.1, max_tokens=100)
            result = _parse_json(response)
            intent = result.get("intent", "???")
            confidence = result.get("confidence", 0)

            # Verificar se o intent está no esperado (pode ser lista)
            expected_list = expected if isinstance(expected, list) else [expected]
            ok = intent in expected_list

            if ok:
                passed += 1
                status = "✅"
            else:
                failed += 1
                status = "❌"
                errors.append((msg, expected, intent, confidence, desc))

            if verbose or not ok:
                expected_str = "/".join(expected_list)
                print(f"  {status} \"{msg}\" → {intent} (conf: {confidence:.0%}) | esperado: {expected_str} | {desc}")

        except Exception as e:
            failed += 1
            errors.append((msg, expected, f"ERRO: {e}", 0, desc))
            print(f"  💥 \"{msg}\" → ERRO: {e}")

    print(f"\n{'='*60}")
    print(f"  Resultado: {passed}/{len(TEST_CASES)} passaram ({passed/len(TEST_CASES):.0%})")

    if errors:
        print(f"\n  ❌ Falhas ({len(errors)}):")
        for msg, expected, got, conf, desc in errors:
            expected_str = "/".join(expected) if isinstance(expected, list) else expected
            print(f"     \"{msg}\" → {got} (esperado: {expected_str}) | {desc}")

    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    logging.basicConfig(level=logging.WARNING)

    success = run_tests(verbose=verbose)
    sys.exit(0 if success else 1)
