import sqlite3
import datetime
import json
from typing import Optional

from config import CONFIG

# Global cache to store in-memory database connections
_memory_db_cache: Optional[sqlite3.Connection] = None

# Global cache to store the last prompt_id returned for the current session
_prompt_id_cache = None

# Global cache to track which database paths have been initialized
_inited_cache = {}


def sql_db():
    """Create tables and indices, then return a database connection.

    Returns a database connection with the prompt table and indices created.
    The caller is responsible for closing the connection or using it as a context manager.
    """
    # Special handling for in-memory databases - always return cached connection
    if str(CONFIG.logging_sqlite_path) == ":memory:":
        global _memory_db_cache
        if _memory_db_cache is None:
            _memory_db_cache = sqlite3.connect(":memory:")
            _create_all_tables(_memory_db_cache)
            _inited_cache[CONFIG.logging_sqlite_path] = True
        return _memory_db_cache

    # For file-based databases
    if CONFIG.logging_sqlite_path not in _inited_cache:
        _inited_cache[CONFIG.logging_sqlite_path] = True
        with sqlite3.connect(CONFIG.logging_sqlite_path) as db:
            _create_all_tables(db)
    db = sqlite3.connect(CONFIG.logging_sqlite_path)

    return db


def _create_all_tables(db: sqlite3.Connection):
    """Create all required tables in the database."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS prompt(id INTEGER PRIMARY KEY, project TEXT NOT NULL, log TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS prompt_proj_idx ON prompt(project)")

    db.execute(
        "CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL)"
    )

    db.execute(
        "CREATE TABLE IF NOT EXISTS generation_history("
        "prompt_id INTEGER NOT NULL, "
        "llm_id INTEGER NOT NULL, "
        "num INTEGER PRIMARY KEY AUTOINCREMENT, "
        "messages TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_hist_llm_idx ON generation_history(llm_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_hist_prompt_idx ON generation_history(prompt_id)"
    )

    db.execute(
        "CREATE TABLE IF NOT EXISTS patch_fail("
        "history_id INTEGER NOT NULL, "
        "orig TEXT NOT NULL, "
        "patch TEXT NOT NULL)"
    )


def prompt_id() -> int:
    """Get or generate a prompt_id.
    If it was not called before, return max(id) from `prompt` + 1 (or 1 if no records exist).
    If it was called before - return the cached value.
    """
    global _prompt_id_cache
    if _prompt_id_cache is not None:
        return _prompt_id_cache

    with sql_db() as sql:
        result = sql.execute("SELECT COALESCE(MAX(id), 0) FROM prompt").fetchone()
        _prompt_id_cache = result[0] + 1
    return _prompt_id_cache


def reset_all_caches():
    """Reset all SQL caches. Useful for testing."""
    global _prompt_id_cache
    global _memory_db_cache
    global _inited_cache
    _prompt_id_cache = None
    if _memory_db_cache is not None:
        _memory_db_cache.close()
        _memory_db_cache = None
    _inited_cache.clear()


def llm_id() -> int:
    """Generate a new LLM session ID.
    Creates a new session in the sessions table and returns its ID.
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()
    with sql_db() as sql:
        sql.execute("INSERT INTO sessions(created_at) VALUES(?)", (now,))
        result = sql.execute("SELECT last_insert_rowid()").fetchone()
    return result[0]


def log_generation(prompt_id: int, llm_id: int, messages: list[dict]) -> int:
    """Log generation history for an LLM.

    Args:
        prompt_id: The prompt ID (from prompt_id())
        llm_id: The LLM instance ID (from llm_id())
        messages: List of messages to log

    Returns:
        The num (history_id) of the logged generation
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()
    messages_json = json.dumps(messages, ensure_ascii=False)

    with sql_db() as sql:
        sql.execute(
            "INSERT INTO generation_history(prompt_id, llm_id, messages, created_at) VALUES(?,?,?,?)",
            (prompt_id, llm_id, messages_json, now),
        )
        result = sql.execute("SELECT last_insert_rowid()").fetchone()
    return result[0]


def log_prompt(project: str, prompt: str):
    """Log used prompt for the project"""
    now = datetime.datetime.now(datetime.UTC).isoformat()

    with sql_db() as sql:
        sql.execute(
            "INSERT INTO prompt(id, project, log, created_at) VALUES(?,?,?,?)",
            (prompt_id(), project, prompt, now),
        )


def log_patch_fail(history_id: int, orig: str, patch: str) -> int:
    """Log a failed patch attempt.

    Args:
        history_id: The generation_history num (from the tool call)
        orig: Original file content
        patch: The patch that failed

    Returns:
        The history_id that was logged
    """
    with sql_db() as sql:
        sql.execute(
            "INSERT INTO patch_fail(history_id, orig, patch) VALUES(?,?,?)",
            (history_id, orig, patch),
        )
    return history_id
