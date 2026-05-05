"""
CLI interativo para testar o chatbot.
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    from agent.graph import run_agent

    customer_id = input("Customer ID (enter para padrão): ").strip() or "test_user_001"
    print(f"\nCustomer: {customer_id}")
    print("Digite 'sair' para encerrar.\n")
    print("=" * 50)

    while True:
        try:
            message = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not message or message.lower() in ("sair", "exit", "quit"):
            break

        result = run_agent(customer_id, message)

        print(f"\n[{result['intent']}] (conf: {result['confidence']:.0%}) route: {result['route']}")
        print(f"\nBot: {result['response']}")

        if result.get("retrieved_docs"):
            print(f"\n  📚 {len(result['retrieved_docs'])} docs recuperados:")
            for d in result["retrieved_docs"][:2]:
                print(f"     [{d['final_score']:.3f}] {d['kb_id']}: {d['content'][:60]}...")

    print("\nAté logo!")


if __name__ == "__main__":
    main()
