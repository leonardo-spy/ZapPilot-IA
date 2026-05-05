"""
Pipeline completo: ingestão → limpeza → turns → knowledge base → indexação Chroma + BM25.
"""
import os
import sys
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("build_all")


def main():
    start = time.time()

    json_path = os.getenv("WHATSAPP_JSON", "./input/whatsapp_chats.json")
    db_path = os.getenv("WHATSAPP_DB", "./input/msgstore.db")
    data_dir = os.getenv("DATA_DIR", "./data")
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    kb_path = f"{data_dir}/knowledge_base.json"
    chroma_path = f"{data_dir}/chroma_db"

    # ==================== STEP 1: INGESTÃO ====================
    logger.info("=" * 60)
    logger.info("STEP 1: Ingestão do WhatsApp (JSON ou msgstore.db)")
    logger.info("=" * 60)

    from ingestion.whatsapp_loader import load_whatsapp_auto
    messages = load_whatsapp_auto(json_path=json_path, db_path=db_path)
    logger.info(f"→ {len(messages)} mensagens carregadas")

    # ==================== STEP 2: LIMPEZA + MERGE ====================
    logger.info("=" * 60)
    logger.info("STEP 2: Limpeza e merge de mensagens")
    logger.info("=" * 60)

    from preprocessing.cleaner import clean_messages, merge_consecutive
    cleaned = clean_messages(messages)
    logger.info(f"→ {len(cleaned)} mensagens após limpeza")

    merged = merge_consecutive(cleaned)
    logger.info(f"→ {len(merged)} blocos merged")

    # ==================== STEP 3: TURNS ====================
    logger.info("=" * 60)
    logger.info("STEP 3: Construção de turns semânticos")
    logger.info("=" * 60)

    from preprocessing.turns import build_turns
    turns = build_turns(merged)
    logger.info(f"→ {len(turns)} turns construídos")

    if not turns:
        logger.error("Nenhum turn gerado! Verifique os dados.")
        return

    # ==================== STEP 4: KNOWLEDGE BASE ====================
    logger.info("=" * 60)
    logger.info("STEP 4: Geração da Knowledge Base")
    logger.info("=" * 60)

    from kb.build_knowledge_base import build_knowledge_base
    kb = build_knowledge_base(
        turns=turns,
        embed_model=embed_model,
        output_path=kb_path,
    )
    logger.info(f"→ {len(kb)} entradas na knowledge base")

    if not kb:
        logger.error("Knowledge base vazia! Ajuste eps/min_samples do DBSCAN.")
        return

    # ==================== STEP 5: KB COMPLEMENTAR (DOMÍNIO) ====================
    logger.info("=" * 60)
    logger.info("STEP 5: KB complementar do domínio (via LLM)")
    logger.info("=" * 60)

    generate_domain = os.getenv("GENERATE_DOMAIN_KB", "true").lower() == "true"
    domain_count = 0

    if generate_domain:
        try:
            from kb.generate_domain_kb import generate_domain_kb, merge_knowledge_bases
            domain_entries = generate_domain_kb(quantity=10)
            domain_count = len(domain_entries)
            logger.info(f"→ {domain_count} entradas complementares geradas (needs_human_review=True)")

            # Merge com a KB primária
            merged_kb = merge_knowledge_bases()
            kb_path = f"{data_dir}/knowledge_base_full.json"
            logger.info(f"→ KB merged: {len(merged_kb)} total em {kb_path}")
        except Exception as e:
            logger.warning(f"→ Geração de KB complementar falhou: {e}")
            logger.warning("  (Continuando com KB primária apenas)")
    else:
        logger.info("→ Skipped (GENERATE_DOMAIN_KB=false)")

    # ==================== STEP 6: INDEXAÇÃO CHROMA ====================
    logger.info("=" * 60)
    logger.info("STEP 6: Indexação ChromaDB")
    logger.info("=" * 60)

    from retrieval.chroma_index import index_knowledge_base
    chroma_count = index_knowledge_base(kb_path, chroma_path, embed_model)
    logger.info(f"→ {chroma_count} documentos no Chroma")

    # ==================== STEP 7: INDEXAÇÃO BM25 ====================
    logger.info("=" * 60)
    logger.info("STEP 7: Indexação BM25")
    logger.info("=" * 60)

    from retrieval.bm25_index import build_bm25_index
    bm25_count = build_bm25_index(kb_path, data_dir)
    logger.info(f"→ {bm25_count} documentos no BM25")

    # ==================== RESUMO ====================
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETO")
    logger.info("=" * 60)
    logger.info(f"  Mensagens lidas:      {len(messages)}")
    logger.info(f"  Após limpeza:         {len(cleaned)}")
    logger.info(f"  Blocos merged:        {len(merged)}")
    logger.info(f"  Turns:                {len(turns)}")
    logger.info(f"  Knowledge base:       {len(kb)} entradas (dados)")
    logger.info(f"  KB complementar:      {domain_count} entradas (domínio)")
    logger.info(f"  Chroma docs:          {chroma_count}")
    logger.info(f"  BM25 docs:            {bm25_count}")
    logger.info(f"  Domínio:              {os.getenv('BOT_DOMAIN', 'android_box')}")
    logger.info(f"  Tempo total:          {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
