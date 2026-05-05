"""
Módulo de ingestão direta do msgstore.db decriptado (WhatsApp SQLite).

Baseado na lógica de extração do whatsapp-forensic-tool (github.com/cedroid/whatsapp-forensic-tool).
Lê o banco SQLite decriptado e normaliza para o mesmo formato do whatsapp_loader.py.

Suporta:
- Schema moderno: tabelas 'chat', 'jid', 'message', 'wa_contacts'
- View: 'available_message_view'
- Schema legado: tabela 'messages' com key_remote_jid

Requisito: O arquivo msgstore.db já deve estar decriptado.
Para decriptar, use o whatsapp-forensic-tool ou ferramentas como wa-crypt-tools.
"""
import os
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MsgStoreReader:
    """Leitor do banco SQLite decriptado do WhatsApp (msgstore.db)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.contact_map: dict[str, str] = {}
        self._schema_version: str = "unknown"

    def connect(self) -> bool:
        """Conecta ao banco SQLite."""
        if not os.path.exists(self.db_path):
            logger.error(f"Arquivo de banco não encontrado: {self.db_path}")
            return False

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self._load_contacts()
            self._detect_schema()
            logger.info(
                f"Conectado ao msgstore.db: {self.db_path} "
                f"(schema: {self._schema_version}, contatos: {len(self.contact_map)})"
            )
            return True
        except sqlite3.Error as e:
            logger.error(f"Falha ao conectar ao banco: {e}")
            return False

    def close(self):
        """Fecha a conexão."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _detect_schema(self):
        """Detecta qual versão do schema está sendo usada."""
        tables = self._get_tables()
        if "message" in tables and "jid" in tables:
            self._schema_version = "modern"
        elif "message" in tables:
            self._schema_version = "modern_no_jid"
        elif "available_message_view" in self._get_views():
            self._schema_version = "view"
        elif "messages" in tables:
            self._schema_version = "legacy"
        else:
            self._schema_version = "unknown"

    def _get_tables(self) -> set[str]:
        """Retorna nomes das tabelas no banco."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in self.cursor.fetchall()}

    def _get_views(self) -> set[str]:
        """Retorna nomes das views no banco."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return {row[0] for row in self.cursor.fetchall()}

    def _get_columns(self, table: str) -> list[str]:
        """Retorna nomes das colunas de uma tabela."""
        self.cursor.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in self.cursor.fetchall()]

    def _load_contacts(self):
        """Carrega mapa de contatos (jid → nome)."""
        try:
            tables = self._get_tables()
            if "wa_contacts" not in tables:
                return

            cols = self._get_columns("wa_contacts")

            # Escolher coluna de nome: display_name > wa_name > push_name
            name_col = None
            for candidate in ("display_name", "wa_name", "push_name"):
                if candidate in cols:
                    name_col = candidate
                    break

            if not name_col or "jid" not in cols:
                return

            self.cursor.execute(
                f"SELECT jid, {name_col} FROM wa_contacts WHERE {name_col} IS NOT NULL AND {name_col} != ''"
            )
            for row in self.cursor.fetchall():
                self.contact_map[row[0]] = row[1]

        except sqlite3.Error as e:
            logger.warning(f"Erro ao carregar contatos: {e}")

    def _resolve_sender(self, jid: str) -> str:
        """Resolve JID para nome de contato."""
        if jid in self.contact_map:
            return self.contact_map[jid]
        return jid

    def list_chats(self) -> list[dict]:
        """Lista todas as conversas do banco."""
        if not self.conn:
            return []

        chats = []
        try:
            tables = self._get_tables()

            if "chat" in tables and "jid" in tables:
                # Schema moderno com tabela jid
                query = """
                    SELECT c._id, j.user, j.server, c.subject, c.sort_timestamp
                    FROM chat c
                    LEFT JOIN jid j ON c.jid_row_id = j._id
                    ORDER BY c.sort_timestamp DESC
                """
                self.cursor.execute(query)
                for row in self.cursor.fetchall():
                    chat_id = row[0]
                    user = row[1] or ""
                    server = row[2] or ""
                    subject = row[3] or ""
                    timestamp = row[4]

                    jid = f"{user}@{server}" if user and server else "unknown"

                    chats.append({
                        "id": chat_id,
                        "jid": jid,
                        "subject": subject or self._resolve_sender(jid),
                        "timestamp": timestamp,
                    })

            elif "chat" in tables:
                # Schema moderno sem jid table
                cols = self._get_columns("chat")
                has_jid_col = "jid_row_id" in cols

                query = "SELECT _id, subject, sort_timestamp FROM chat ORDER BY sort_timestamp DESC"
                self.cursor.execute(query)
                for row in self.cursor.fetchall():
                    chats.append({
                        "id": row[0],
                        "jid": f"chat_{row[0]}",
                        "subject": row[1] or f"Chat {row[0]}",
                        "timestamp": row[2],
                    })

            elif "messages" in tables:
                # Schema legado: extrair chats únicos da tabela messages
                query = """
                    SELECT DISTINCT key_remote_jid
                    FROM messages
                    WHERE key_remote_jid IS NOT NULL AND key_remote_jid != ''
                """
                self.cursor.execute(query)
                for idx, row in enumerate(self.cursor.fetchall()):
                    jid = row[0]
                    chats.append({
                        "id": idx + 1,
                        "jid": jid,
                        "subject": self._resolve_sender(jid),
                        "timestamp": 0,
                    })

        except sqlite3.Error as e:
            logger.error(f"Erro ao listar chats: {e}")

        logger.info(f"{len(chats)} chats encontrados no banco")
        return chats

    def get_messages(self, chat_id: int, chat_jid: str) -> list[dict]:
        """
        Retorna mensagens de um chat específico.
        Tenta 3 estratégias (moderna → view → legado).
        """
        if not self.conn:
            return []

        messages = []

        try:
            # Estratégia 1: tabela 'message' (schema moderno)
            messages = self._get_messages_modern(chat_id, chat_jid)

            # Estratégia 2: view 'available_message_view'
            if not messages:
                messages = self._get_messages_view(chat_id)

            # Estratégia 3: tabela 'messages' (legado)
            if not messages:
                messages = self._get_messages_legacy(chat_jid)

        except sqlite3.Error as e:
            logger.error(f"Erro ao buscar mensagens do chat {chat_id}: {e}")

        return messages

    def _get_messages_modern(self, chat_id: int, chat_jid: str) -> list[dict]:
        """Busca mensagens na tabela 'message' (schema moderno)."""
        tables = self._get_tables()
        if "message" not in tables:
            return []

        cols = self._get_columns("message")
        has_jid_table = "jid" in tables

        # Verificar coluna de sender
        sender_col = "sender_jid_row_id" if "sender_jid_row_id" in cols else None

        # Colunas base
        select = "m._id, m.text_data, m.timestamp, m.from_me"
        if sender_col:
            select += f", m.{sender_col}"
        else:
            select += ", 0"

        query = f"""
            SELECT {select}
            FROM message m
            WHERE m.chat_row_id = ?
            ORDER BY m.timestamp ASC
        """

        self.cursor.execute(query, (chat_id,))
        messages = []

        for row in self.cursor.fetchall():
            msg_id = row[0]
            text = row[1]
            ts = row[2]
            from_me = bool(row[3])
            sender_row_id = row[4]

            # Resolver sender
            sender_jid = "me"
            if not from_me and has_jid_table and sender_row_id and sender_row_id > 0:
                try:
                    self.cursor.execute("SELECT user, server FROM jid WHERE _id=?", (sender_row_id,))
                    jid_res = self.cursor.fetchone()
                    if jid_res:
                        sender_jid = f"{jid_res[0]}@{jid_res[1]}"
                except sqlite3.Error:
                    sender_jid = chat_jid
            elif not from_me:
                sender_jid = chat_jid

            # Filtrar mensagens sem texto
            if not text:
                continue

            # Normalizar timestamp (ms → s)
            if ts and ts > 1e12:
                ts = ts / 1000

            messages.append({
                "msg_id": str(msg_id),
                "text": text,
                "timestamp": ts or 0,
                "from_me": from_me,
                "sender": self._resolve_sender(sender_jid),
                "sender_jid": sender_jid,
            })

        return messages

    def _get_messages_view(self, chat_id: int) -> list[dict]:
        """Busca mensagens na view 'available_message_view'."""
        views = self._get_views()
        if "available_message_view" not in views:
            return []

        query = """
            SELECT text_data, timestamp, from_me
            FROM available_message_view
            WHERE chat_row_id = ?
            ORDER BY timestamp ASC
        """

        self.cursor.execute(query, (chat_id,))
        messages = []

        for idx, row in enumerate(self.cursor.fetchall()):
            text = row[0]
            ts = row[1]
            from_me = bool(row[2])

            if not text:
                continue

            if ts and ts > 1e12:
                ts = ts / 1000

            messages.append({
                "msg_id": str(idx),
                "text": text,
                "timestamp": ts or 0,
                "from_me": from_me,
                "sender": "me" if from_me else "contact",
                "sender_jid": "me" if from_me else "contact",
            })

        return messages

    def _get_messages_legacy(self, chat_jid: str) -> list[dict]:
        """Busca mensagens na tabela 'messages' (schema legado)."""
        tables = self._get_tables()
        if "messages" not in tables:
            return []

        query = """
            SELECT _id, data, timestamp, key_from_me, remote_resource
            FROM messages
            WHERE key_remote_jid = ?
            ORDER BY timestamp ASC
        """

        self.cursor.execute(query, (chat_jid,))
        messages = []

        for row in self.cursor.fetchall():
            msg_id = row[0]
            text = row[1]
            ts = row[2]
            from_me = bool(row[3])
            remote_resource = row[4]

            if not text:
                continue

            sender_jid = "me"
            if not from_me:
                sender_jid = remote_resource if remote_resource else chat_jid

            if ts and ts > 1e12:
                ts = ts / 1000

            messages.append({
                "msg_id": str(msg_id),
                "text": text,
                "timestamp": ts or 0,
                "from_me": from_me,
                "sender": self._resolve_sender(sender_jid),
                "sender_jid": sender_jid,
            })

        return messages

    def get_stats(self) -> dict:
        """Retorna estatísticas do banco."""
        if not self.conn:
            return {}

        stats = {
            "schema_version": self._schema_version,
            "contacts": len(self.contact_map),
            "tables": sorted(self._get_tables()),
        }

        # Contar mensagens
        try:
            tables = self._get_tables()
            if "message" in tables:
                self.cursor.execute("SELECT COUNT(*) FROM message")
                stats["total_messages"] = self.cursor.fetchone()[0]
            elif "messages" in tables:
                self.cursor.execute("SELECT COUNT(*) FROM messages")
                stats["total_messages"] = self.cursor.fetchone()[0]

            if "chat" in tables:
                self.cursor.execute("SELECT COUNT(*) FROM chat")
                stats["total_chats"] = self.cursor.fetchone()[0]
        except sqlite3.Error:
            pass

        return stats


def load_whatsapp_db(db_path: str) -> list[dict]:
    """
    Carrega mensagens do msgstore.db decriptado e retorna no formato normalizado.

    Formato de saída é idêntico ao whatsapp_loader.load_whatsapp_json():
    [
        {
            "chat_id": str,
            "subject": str,
            "message_id": str,
            "sender": str,
            "from_me": bool,
            "timestamp": float,
            "text": str,
        },
        ...
    ]
    """
    logger.info(f"Carregando msgstore.db: {db_path}")

    reader = MsgStoreReader(db_path)
    if not reader.connect():
        raise FileNotFoundError(f"Não foi possível abrir o banco: {db_path}")

    try:
        stats = reader.get_stats()
        logger.info(
            f"Schema: {stats.get('schema_version')}, "
            f"mensagens totais: {stats.get('total_messages', '?')}, "
            f"chats: {stats.get('total_chats', '?')}"
        )

        chats = reader.list_chats()
        all_messages = []
        skipped = 0

        for chat in chats:
            messages = reader.get_messages(chat["id"], chat["jid"])

            for msg in messages:
                text = (msg["text"] or "").strip()

                # Filtrar mensagens sem texto útil
                if not text:
                    skipped += 1
                    continue

                # Filtrar placeholders de mídia
                if text.startswith("<Media") and text.endswith(">"):
                    skipped += 1
                    continue

                all_messages.append({
                    "chat_id": chat["jid"],
                    "subject": chat["subject"],
                    "message_id": msg["msg_id"],
                    "sender": msg["sender"],
                    "from_me": msg["from_me"],
                    "timestamp": msg["timestamp"],
                    "text": text,
                })

        # Ordenar por timestamp
        all_messages.sort(key=lambda x: x["timestamp"])

        logger.info(
            f"msgstore.db: {len(chats)} chats, "
            f"{len(all_messages)} mensagens retidas, {skipped} filtradas"
        )
        return all_messages

    finally:
        reader.close()


def export_db_to_json(db_path: str, output_path: str) -> str:
    """
    Exporta msgstore.db para JSON no formato direto (compatível com whatsapp_loader).

    Útil para gerar o JSON a partir do banco e depois usar o pipeline normal.
    """
    import json

    logger.info(f"Exportando {db_path} → {output_path}")

    reader = MsgStoreReader(db_path)
    if not reader.connect():
        raise FileNotFoundError(f"Não foi possível abrir o banco: {db_path}")

    try:
        chats = reader.list_chats()
        export_data = []

        for chat in chats:
            messages = reader.get_messages(chat["id"], chat["jid"])

            # Filtrar mensagens vazias
            filtered_msgs = []
            for msg in messages:
                text = (msg["text"] or "").strip()
                if not text or (text.startswith("<Media") and text.endswith(">")):
                    continue
                filtered_msgs.append({
                    "id": msg["msg_id"],
                    "text": text,
                    "timestamp": msg["timestamp"],
                    "from_me": msg["from_me"],
                    "sender": msg["sender"],
                })

            if filtered_msgs:
                export_data.append({
                    "id": chat["jid"],
                    "jid": chat["jid"],
                    "subject": chat["subject"],
                    "messages": filtered_msgs,
                })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exportado: {len(export_data)} chats, {sum(len(c['messages']) for c in export_data)} mensagens → {output_path}")
        return output_path

    finally:
        reader.close()


if __name__ == "__main__":
    """CLI: python -m ingestion.msgstore_loader [db_path] [--export output.json] [--stats]"""
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    db_path = os.getenv("WHATSAPP_DB", "./input/msgstore.db")

    # Parse args
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        db_path = args.pop(0)

    export_path = None
    show_stats = False

    while args:
        arg = args.pop(0)
        if arg == "--export" and args:
            export_path = args.pop(0)
        elif arg == "--stats":
            show_stats = True

    if show_stats:
        reader = MsgStoreReader(db_path)
        if reader.connect():
            stats = reader.get_stats()
            print("\n=== Estatísticas do msgstore.db ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            reader.close()
        sys.exit(0)

    if export_path:
        export_db_to_json(db_path, export_path)
    else:
        messages = load_whatsapp_db(db_path)
        print(f"\n{len(messages)} mensagens carregadas do banco.")
        if messages:
            print(f"Primeiro: {messages[0]['timestamp']} - {messages[0]['text'][:80]}")
            print(f"Último:   {messages[-1]['timestamp']} - {messages[-1]['text'][:80]}")
