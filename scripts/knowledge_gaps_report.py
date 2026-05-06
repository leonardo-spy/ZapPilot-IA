#!/usr/bin/env python3
"""
Knowledge Gaps Report — Identifies what information the KB is missing.

Analyzes recorded gaps (questions where the agent lacked context) and groups
them by similarity to generate actionable recommendations for the operator.

Usage:
    python scripts/knowledge_gaps_report.py [--days 30] [--top 20]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.sqlite_memory import SQLiteMemory


def _normalize_query(q: str) -> str:
    """Normalize query for grouping (lowercase, strip punctuation)."""
    import re
    q = q.lower().strip()
    q = re.sub(r'[?!.,;:]+$', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q


def _simple_cluster(queries: list[str], threshold: float = 0.6) -> list[list[str]]:
    """
    Group similar queries using word overlap (Jaccard similarity).
    Fast, no embeddings needed. Good enough for initial analysis.
    """
    clusters: list[list[str]] = []
    used = set()

    normalized = [_normalize_query(q) for q in queries]

    for i, q1 in enumerate(normalized):
        if i in used:
            continue
        cluster = [queries[i]]
        words1 = set(q1.split())
        used.add(i)

        for j, q2 in enumerate(normalized):
            if j in used or j <= i:
                continue
            words2 = set(q2.split())
            if not words1 or not words2:
                continue
            jaccard = len(words1 & words2) / len(words1 | words2)
            if jaccard >= threshold:
                cluster.append(queries[j])
                used.add(j)

        clusters.append(cluster)

    # Sort by cluster size (most frequent gaps first)
    clusters.sort(key=len, reverse=True)
    return clusters


def generate_report(days: int = 30, top_n: int = 20, domain: str = None) -> str:
    """Generate a knowledge gaps report with recommendations, filtered by domain."""
    import os
    db_path = os.getenv("DATA_DIR", "./data") + "/memory.db"
    memory = SQLiteMemory(db_path)

    if domain is None:
        domain = os.getenv("BOT_DOMAIN", "android_box")

    gaps = memory.get_knowledge_gaps(limit=500, since_days=days, domain=domain)
    summary = memory.get_knowledge_gaps_summary(since_days=days, domain=domain)

    if not gaps:
        return f"✅ Nenhum gap de conhecimento detectado nos últimos {days} dias."

    lines = []
    lines.append(f"📊 Relatório de Gaps de Conhecimento — Últimos {days} dias (domain: {domain})")
    lines.append("=" * 60)
    lines.append(f"\n📈 Total de gaps registrados: {summary['total']}")

    # By intent
    if summary['by_intent']:
        lines.append("\n📋 Por intenção:")
        for item in summary['by_intent']:
            lines.append(f"  • {item['intent'] or 'unknown'}: {item['cnt']} gaps")

    # By route
    if summary['by_route']:
        lines.append("\n🔀 Por rota:")
        for item in summary['by_route']:
            lines.append(f"  • {item['route'] or 'unknown'}: {item['cnt']} gaps")

    # Cluster similar queries
    queries = [g["query"] for g in gaps]
    clusters = _simple_cluster(queries, threshold=0.4)

    lines.append(f"\n{'=' * 60}")
    lines.append("🎯 RECOMENDAÇÕES — Tópicos para adicionar à KB:")
    lines.append("=" * 60)

    for i, cluster in enumerate(clusters[:top_n], 1):
        representative = cluster[0][:120]
        freq = len(cluster)

        # Identify the most common intent for this cluster
        cluster_intents = []
        for q in cluster:
            for g in gaps:
                if g["query"] == q:
                    cluster_intents.append(g["intent"])
                    break
        most_common_intent = Counter(cluster_intents).most_common(1)
        intent_label = most_common_intent[0][0] if most_common_intent else "?"

        lines.append(f"\n  {i}. [{intent_label}] ({freq}x) {representative}")
        if freq > 1:
            # Show a few variations
            for variant in cluster[1:3]:
                lines.append(f"     └─ \"{variant[:80]}\"")

    # Final recommendation
    lines.append(f"\n{'=' * 60}")
    lines.append("💡 AÇÃO SUGERIDA:")
    lines.append("   Adicione respostas para os tópicos acima no knowledge base.")
    lines.append("   Priorize os itens com maior frequência (Nx).")
    lines.append("   Execute: python kb/build_knowledge_base.py após adicionar conteúdo.")

    return "\n".join(lines)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Knowledge Gaps Report")
    parser.add_argument("--days", type=int, default=30, help="Period in days (default: 30)")
    parser.add_argument("--top", type=int, default=20, help="Max recommendations (default: 20)")
    parser.add_argument("--domain", type=str, default=None, help="Domain filter (default: BOT_DOMAIN)")
    args = parser.parse_args()

    report = generate_report(days=args.days, top_n=args.top, domain=args.domain)
    print(report)
