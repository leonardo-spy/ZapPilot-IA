"""
Memória persistente com SQLite.
Armazena histórico de conversas, fatos sobre clientes e casos de suporte.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SQLiteMemory:
    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """Cria tabelas se não existirem."""
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    name TEXT,
                    phone TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    intent TEXT,
                    domain TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                );

                CREATE TABLE IF NOT EXISTS customer_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    source_message TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                );

                CREATE TABLE IF NOT EXISTS support_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    intent TEXT,
                    summary TEXT,
                    solution_tried TEXT,
                    resolved INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_customer
                    ON conversations(customer_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_facts_customer
                    ON customer_facts(customer_id);
                CREATE INDEX IF NOT EXISTS idx_cases_customer
                    ON support_cases(customer_id, status);

                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    route TEXT,
                    confidence REAL,
                    retrieved_docs_count INTEGER DEFAULT 0,
                    domain TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_created
                    ON knowledge_gaps(created_at);
            """)

            # Migration: add domain column to existing tables
            self._migrate_add_column(conn, "conversations", "domain", "TEXT")
            self._migrate_add_column(conn, "knowledge_gaps", "domain", "TEXT")

            # Create domain indexes (after migration ensures columns exist)
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_conversations_domain
                    ON conversations(domain, customer_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_domain
                    ON knowledge_gaps(domain, created_at);
            """)
            self._migrate_add_column(conn, "knowledge_gaps", "domain", "TEXT")

        logger.info(f"SQLite memory inicializada: {self.db_path}")

    def _migrate_add_column(self, conn, table: str, column: str, col_type: str):
        """Adds a column to a table if it doesn't exist (migration-safe)."""
        try:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                logger.info(f"Migration: added '{column}' to '{table}'")
        except Exception as e:
            logger.debug(f"Migration check for {table}.{column}: {e}")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ==================== CUSTOMERS ====================

    def get_or_create_customer(self, customer_id: str, name: str = None, phone: str = None) -> dict:
        """Busca ou cria cliente."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()

            if row:
                return dict(row)

            now = self._now()
            conn.execute(
                "INSERT INTO customers (customer_id, name, phone, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (customer_id, name or "", phone or customer_id, now, now)
            )
            return {"customer_id": customer_id, "name": name, "phone": phone, "created_at": now, "updated_at": now}

    # ==================== CONVERSATIONS ====================

    def save_message(self, customer_id: str, role: str, message: str, intent: str = None, domain: str = None):
        """Salva mensagem no histórico, classificada por domain."""
        self.get_or_create_customer(customer_id)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (customer_id, role, message, intent, domain, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (customer_id, role, message, intent, domain, self._now())
            )

    def get_recent_history(self, customer_id: str, limit: int = 10, domain: str = None) -> list[dict]:
        """Retorna últimas mensagens do cliente, opcionalmente filtradas por domain."""
        with self._get_conn() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT role, message, intent, timestamp FROM conversations "
                    "WHERE customer_id = ? AND domain = ? ORDER BY timestamp DESC LIMIT ?",
                    (customer_id, domain, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, message, intent, timestamp FROM conversations "
                    "WHERE customer_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (customer_id, limit)
                ).fetchall()
        # Retornar em ordem cronológica
        return [dict(r) for r in reversed(rows)]

    # ==================== FACTS ====================

    def save_fact(self, customer_id: str, key: str, value: str, source_message: str = None):
        """Salva fato sobre o cliente (upsert por key)."""
        self.get_or_create_customer(customer_id)
        with self._get_conn() as conn:
            # Verificar se já existe
            existing = conn.execute(
                "SELECT id FROM customer_facts WHERE customer_id = ? AND fact_key = ?",
                (customer_id, key)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE customer_facts SET fact_value = ?, source_message = ?, timestamp = ? WHERE id = ?",
                    (value, source_message, self._now(), existing["id"])
                )
            else:
                conn.execute(
                    "INSERT INTO customer_facts (customer_id, fact_key, fact_value, source_message, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (customer_id, key, value, source_message, self._now())
                )

    def get_customer_facts(self, customer_id: str) -> list[dict]:
        """Retorna todos os fatos conhecidos sobre o cliente."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT fact_key, fact_value, timestamp FROM customer_facts WHERE customer_id = ?",
                (customer_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================== SUPPORT CASES ====================

    def create_or_update_case(
        self,
        customer_id: str,
        intent: str,
        summary: str,
        solution_tried: str = None,
        resolved: bool = False,
    ) -> int:
        """Cria ou atualiza caso de suporte."""
        self.get_or_create_customer(customer_id)
        now = self._now()

        with self._get_conn() as conn:
            # Buscar caso aberto com mesmo intent
            existing = conn.execute(
                "SELECT id FROM support_cases WHERE customer_id = ? AND intent = ? AND resolved = 0",
                (customer_id, intent)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE support_cases SET summary = ?, solution_tried = ?, resolved = ?, "
                    "status = ?, updated_at = ? WHERE id = ?",
                    (summary, solution_tried, int(resolved),
                     "resolved" if resolved else "open", now, existing["id"])
                )
                return existing["id"]
            else:
                cursor = conn.execute(
                    "INSERT INTO support_cases (customer_id, status, intent, summary, solution_tried, resolved, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (customer_id, "resolved" if resolved else "open", intent, summary, solution_tried, int(resolved), now, now)
                )
                return cursor.lastrowid

    def get_open_cases(self, customer_id: str) -> list[dict]:
        """Retorna casos abertos do cliente."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM support_cases WHERE customer_id = ? AND resolved = 0 ORDER BY updated_at DESC",
                (customer_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_case(self, case_id: int):
        """Marca caso como resolvido."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE support_cases SET resolved = 1, status = 'resolved', updated_at = ? WHERE id = ?",
                (self._now(), case_id)
            )

    # ==================== CONTEXT BUILDER ====================

    def get_memory_context(self, customer_id: str, domain: str = None) -> str:
        """Monta contexto de memória formatado para o LLM."""
        parts = []

        # Histórico recente
        history = self.get_recent_history(customer_id, limit=5, domain=domain)
        if history:
            parts.append("Histórico recente:")
            for msg in history:
                parts.append(f"  {msg['role']}: {msg['message'][:150]}")

        # Fatos (excluir internos _flow_*)
        facts = self.get_customer_facts(customer_id)
        visible_facts = [f for f in facts if not f["fact_key"].startswith("_flow_")]
        if visible_facts:
            parts.append("\nFatos sobre o cliente:")
            for f in visible_facts:
                parts.append(f"  {f['fact_key']}: {f['fact_value']}")

        # Casos abertos
        cases = self.get_open_cases(customer_id)
        if cases:
            parts.append("\nCasos abertos:")
            for c in cases:
                parts.append(f"  [{c['intent']}] {c['summary']} (solução tentada: {c['solution_tried'] or 'nenhuma'})")

        return "\n".join(parts) if parts else ""

    # ==================== FLOW STATE ====================

    def get_flow_state(self, customer_id: str) -> dict | None:
        """
        Retorna estado do flow ativo do cliente.
        Returns: {"flow": "nome_do_flow", "step": int, "data": {}} ou None.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT fact_value FROM customer_facts "
                "WHERE customer_id = ? AND fact_key = '_flow_state'",
                (customer_id,)
            ).fetchone()

        if not row:
            return None

        import json
        try:
            state = json.loads(row["fact_value"])
            return state if state.get("flow") else None
        except (json.JSONDecodeError, TypeError):
            return None

    def save_flow_state(self, customer_id: str, flow: str, step: int, data: dict = None):
        """
        Salva estado do flow ativo.

        Args:
            flow: Nome do flow ativo
            step: Índice do step atual (0-based)
            data: Dados extras do flow (ex: respostas coletadas)
        """
        import json
        state = {"flow": flow, "step": step, "data": data or {}}
        self.save_fact(customer_id, "_flow_state", json.dumps(state, ensure_ascii=False))

    def clear_flow_state(self, customer_id: str):
        """Limpa o flow state (flow concluído ou abandonado)."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM customer_facts WHERE customer_id = ? AND fact_key = '_flow_state'",
                (customer_id,)
            )

    # ==================== KNOWLEDGE GAPS ====================

    def record_knowledge_gap(
        self,
        customer_id: str,
        query: str,
        intent: str = "",
        route: str = "",
        confidence: float = 0.0,
        retrieved_docs_count: int = 0,
        domain: str = None,
    ):
        """Records a knowledge gap when the agent lacked context to respond well."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO knowledge_gaps (customer_id, query, intent, route, confidence, retrieved_docs_count, domain, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (customer_id, query[:500], intent, route, confidence, retrieved_docs_count, domain, self._now())
            )

    def get_knowledge_gaps(self, limit: int = 100, since_days: int = 30, domain: str = None) -> list[dict]:
        """Returns recent knowledge gaps for analysis, optionally filtered by domain."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        with self._get_conn() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT query, intent, route, confidence, retrieved_docs_count, domain, created_at "
                    "FROM knowledge_gaps WHERE created_at >= ? AND domain = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (cutoff, domain, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT query, intent, route, confidence, retrieved_docs_count, domain, created_at "
                    "FROM knowledge_gaps WHERE created_at >= ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (cutoff, limit)
                ).fetchall()
        return [dict(r) for r in rows]

    def get_knowledge_gaps_summary(self, since_days: int = 30, domain: str = None) -> dict:
        """Returns aggregated summary of knowledge gaps, optionally filtered by domain."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        domain_clause = " AND domain = ?" if domain else ""
        params_base = (cutoff, domain) if domain else (cutoff,)
        with self._get_conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) as cnt FROM knowledge_gaps WHERE created_at >= ?{domain_clause}",
                params_base
            ).fetchone()["cnt"]

            by_intent = conn.execute(
                f"SELECT intent, COUNT(*) as cnt FROM knowledge_gaps "
                f"WHERE created_at >= ?{domain_clause} GROUP BY intent ORDER BY cnt DESC",
                params_base
            ).fetchall()

            by_route = conn.execute(
                f"SELECT route, COUNT(*) as cnt FROM knowledge_gaps "
                f"WHERE created_at >= ?{domain_clause} GROUP BY route ORDER BY cnt DESC",
                params_base
            ).fetchall()

        return {
            "total": total,
            "period_days": since_days,
            "domain_filter": domain,
            "by_intent": [dict(r) for r in by_intent],
            "by_route": [dict(r) for r in by_route],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mem = SQLiteMemory("./data/memory.db")

    # Teste
    mem.save_message("5511999999999", "user", "quanto custa a tirzec?")
    mem.save_message("5511999999999", "assistant", "A Tirzec está por R$ 1.490,00")
    mem.save_fact("5511999999999", "interesse", "tirzec 15mg")
    mem.save_fact("5511999999999", "cidade", "SJC")
    mem.create_or_update_case("5511999999999", "consulta_preco", "Cliente quer saber preço da Tirzec")

    ctx = mem.get_memory_context("5511999999999")
    print(f"\nContexto de memória:\n{ctx}")
