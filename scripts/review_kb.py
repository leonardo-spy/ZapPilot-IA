"""
Review da knowledge base — visualiza entradas para curadoria manual.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    data_dir = os.getenv("DATA_DIR", "./data")
    kb_path = f"{data_dir}/knowledge_base.json"

    if not os.path.exists(kb_path):
        print(f"Knowledge base não encontrada em: {kb_path}")
        print("Execute primeiro: python scripts/build_all.py")
        return

    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    print(f"\n{'='*60}")
    print(f"KNOWLEDGE BASE — {len(kb)} entradas")
    print(f"{'='*60}\n")

    for i, entry in enumerate(kb):
        print(f"[{entry['id']}] Cluster {entry['cluster_id']} | {entry['category']} | {entry['intent']}")
        print(f"  Confiança: {entry['confidence']:.2f} | Exemplos: {entry['source_turn_count']}")
        print(f"  {'⚠️  NEEDS REVIEW' if entry.get('needs_human_review') else '✓ OK'}")
        print(f"  Título: {entry['title'][:80]}")
        print(f"  Sintomas: {entry['symptoms'][:3]}")
        print(f"  Resposta: {entry['recommended_response'][:120]}...")
        if entry.get("steps"):
            print(f"  Steps: {entry['steps'][:3]}")
        print()

    # Stats
    categories = {}
    for e in kb:
        cat = e.get("category", "?")
        categories[cat] = categories.get(cat, 0) + 1

    needs_review = sum(1 for e in kb if e.get("needs_human_review"))

    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"  Total: {len(kb)}")
    print(f"  Categorias: {categories}")
    print(f"  Precisam revisão: {needs_review}")
    print(f"  Confiança média: {sum(e['confidence'] for e in kb) / len(kb):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
