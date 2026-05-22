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
        "tools_id INTEGER, "
        "num INTEGER PRIMARY KEY AUTOINCREMENT, "
        "created_at TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_hist_llm_idx ON generation_history(llm_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_hist_prompt_idx ON generation_history(prompt_id)"
    )

    # generation_message stores unique message texts
    db.execute(
        "CREATE TABLE IF NOT EXISTS generation_message("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT UNIQUE NOT NULL)"
    )

    # generation_context links generation_history to messages
    db.execute(
        "CREATE TABLE IF NOT EXISTS generation_context("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "llm_num_id INTEGER NOT NULL, "
        "message_id INTEGER NOT NULL)"
    )
    # Indexes for efficient joins
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_ctx_llm_num_idx ON generation_context(llm_num_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS gen_ctx_msg_idx ON generation_context(message_id)"
    )
    # llm_tools stores unique tool info texts (normalized like generation_message)
    db.execute(
        "CREATE TABLE IF NOT EXISTS llm_tools("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT UNIQUE NOT NULL)"
    )
    # tool_error logs errors that occur during tool calling
    db.execute(
        "CREATE TABLE IF NOT EXISTS tool_error("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "llm_id INTEGER NOT NULL, "
        "reason TEXT NOT NULL, "
        "name_tag TEXT NOT NULL, "
        "tool_list TEXT NOT NULL, "
        "function_name TEXT, "
        "function_args TEXT, "
        "exception_traceback TEXT, "
        "created_at TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS tool_error_llm_idx ON tool_error(llm_id)"
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
        c = sql.execute(
            "INSERT INTO sessions(created_at) VALUES(?) RETURNING id", (now,)
        )
        [result] = c.fetchone()
        return result


def log_tools(db: sqlite3.Connection, tools: dict) -> int:
    """Log tools info and return the tools_id.

    Uses normalized llm_tools table to store unique tool info texts,
    so duplicate tool configurations share the same text record.

    Returns:
        The id of the tools record in llm_tools table
    """
    # Build json of tool keys
    tool_list = list(x for x in sorted(tools.keys()))
    tool_info = json.dumps(tool_list, ensure_ascii=False, sort_keys=True)

    # Try to insert tool_info text; ignore if already exists (unique index)
    c = db.execute(
        "INSERT INTO llm_tools(text) VALUES(?) "
        "ON CONFLICT(text) DO UPDATE SET text=text RETURNING id",
        (tool_info,),
    )
    [tool_id] = c.fetchone()

    return tool_id


def log_generation(
    prompt_id: int, llm_id: int, messages: list[dict], tools: dict
) -> int:
    """Log generation history for an LLM.

    Args:
        prompt_id: The prompt ID (from prompt_id())
        llm_id: The LLM instance ID (from llm_id())
        messages: List of messages to log
        tools: Dict of available tools

    Returns:
        The num (history_id) of the logged generation
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()

    with sql_db() as sql:
        # First, log tools and get tools_id
        tools_id = log_tools(sql, tools)

        # Insert into generation_history with tools_id
        c = sql.execute(
            "INSERT INTO generation_history(prompt_id, llm_id, tools_id, created_at) VALUES(?,?,?,?) RETURNING num",
            (prompt_id, llm_id, tools_id, now),
        )
        [llm_num_id] = c.fetchone()

        # Store each message in generation_message (skip if exists) and link in generation_context
        for msg in messages:
            msg_json = json.dumps(msg, ensure_ascii=False, sort_keys=True)
            # Try to insert; ignore if already exists (unique index on text)
            c = sql.execute(
                "INSERT INTO generation_message(text) VALUES(?) ON CONFLICT(text) DO UPDATE SET text=text RETURNING id",
                (msg_json,),
            )
            [msg_id] = c.fetchone()
            # Link the message to this generation
            sql.execute(
                "INSERT INTO generation_context(llm_num_id, message_id) VALUES(?,?)",
                (llm_num_id, msg_id),
            )

        return llm_num_id


def log_prompt(project: str, prompt: str):
    """Log used prompt for the project"""
    now = datetime.datetime.now(datetime.UTC).isoformat()

    with sql_db() as sql:
        sql.execute(
            "INSERT INTO prompt(id, project, log, created_at) VALUES(?,?,?,?)",
            (prompt_id(), project, prompt, now),
        )


def log_tool_error(
    llm_id: int,
    reason: str,
    name_tag: str,
    tool_list: str,
    function_name: str = None,
    function_args: str = None,
    exception_traceback: str = None,
) -> int:
    """Log a tool error that occurred during tool calling.

    Args:
        llm_id: The LLM instance ID
        reason: Reason for the error (e.g., "non-existing function", "execution error")
        name_tag: The LLM's name_tag
        tool_list: String representation of available tools
        function_name: Name of the function that was attempted (optional)
        function_args: Arguments passed to the function (optional)
        exception_traceback: Traceback of the exception (optional)

    Returns:
        The id of the inserted row
    """
    import traceback

    if exception_traceback is None:
        exception_traceback = "".join(traceback.format_stack())

    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    conn = sql_db()
    cursor = conn.execute(
        "INSERT INTO tool_error "
        "(llm_id, reason, name_tag, tool_list, function_name, function_args, "
        "exception_traceback, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            llm_id,
            reason,
            name_tag,
            tool_list,
            function_name,
            function_args,
            exception_traceback,
            created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid
