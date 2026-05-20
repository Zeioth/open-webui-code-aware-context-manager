"""
title: Code-Aware Context Manager with LTM & Summarization (v5.10)
description: Full-featured context manager for coding assistants. Persists state, tracks line ranges, applies diffs, compresses LTM, scores importance, learns from assistant responses, summarizes inactive code, supports manual importance markers, robust LLM fallback, natural language forget commands, feedback tracking, hierarchical project-based memory, LRU cache, optional reranking, semantic dependency tracking, and automatic handling of oversized code blocks.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.10.0
license: GPL3
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb, rapidfuzz
"""

import os
import time
import re
import hashlib
import sqlite3
from collections import OrderedDict
import json
import asyncio
import difflib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE = True
except ImportError:
    HAS_SENTENCE = False

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from rapidfuzz import fuzz

    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

from loguru import logger

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class ContentType(str, Enum):
    BASE_CODE = "base_code"
    PROPOSED_CHANGE = "proposed_change"
    COMMITTED_CHANGE = "committed_change"
    GENERAL = "general"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class CodeBlock(BaseModel):
    content: str
    content_type: ContentType
    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    timestamp: float = Field(default_factory=time.time)
    is_active: bool = True
    hash: str = ""
    importance_score: float = 1.0
    mention_count: int = 1
    last_mentioned: float = Field(default_factory=time.time)
    generated_by_assistant: bool = False
    dependencies: List[str] = Field(default_factory=list)
    potentially_affected: bool = False
    affected_timestamp: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
        self._update_importance()

    def _update_importance(self):
        base_score = {
            ContentType.BASE_CODE: 8.0,
            ContentType.ERROR: 7.0,
            ContentType.COMMITTED_CHANGE: 6.0,
            ContentType.PROPOSED_CHANGE: 5.0,
            ContentType.TOOL_CALL: 3.0,
            ContentType.GENERAL: 2.0,
        }.get(self.content_type, 2.0)

        keyword_boost = 0.0
        if re.search(
            r"\b(fix|bug|security|critical|important|todo)\b",
            self.content,
            re.IGNORECASE,
        ):
            keyword_boost = 2.0

        if self.generated_by_assistant:
            base_score *= 0.8

        mention_boost = min(self.mention_count / 5, 3.0)
        age_hours = (time.time() - self.last_mentioned) / 3600
        recency_factor = 0.5**age_hours
        penalty = 0.7 if self.potentially_affected else 1.0
        self.importance_score = (
            (base_score + keyword_boost) * mention_boost * recency_factor * penalty
        )


class AppliedChangeFeedback(BaseModel):
    change_hash: str
    change_description: str
    file_path: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    success: bool = True
    user_comment: str = ""
    resolved: bool = False


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority level.")
        max_turns: int = Field(
            default=20, description="Max non-system messages to keep."
        )
        debug: bool = Field(default=False, description="Verbose debug logging.")
        state_db_path: str = Field(
            default="/app/backend/data/conversation_state.db",
            description="SQLite DB path.",
        )
        track_line_numbers: bool = Field(
            default=True, description="Extract line numbers."
        )
        adaptive_trim: bool = Field(
            default=True, description="Trim only when exceeding tokens."
        )
        context_window_tokens: int = Field(
            default=8192, description="Context window size."
        )
        use_tiktoken: bool = Field(
            default=True, description="Use tiktoken for token counting."
        )

        long_term_memory_dir: str = Field(
            default="/app/backend/data/long_term_memory", description="ChromaDB dir."
        )
        long_term_memory_expiration_days: int = Field(
            default=30, description="LTM expiration days."
        )
        long_term_memory_top_k: int = Field(
            default=10, description="LTM retrieval count."
        )
        long_term_memory_similarity_threshold: float = Field(
            default=0.65, description="Cosine threshold."
        )
        ltm_time_decay_hours: float = Field(
            default=24.0, description="Time decay for LTM."
        )
        enable_reranking: bool = Field(
            default=False, description="Use cross-encoder reranking for LTM results."
        )
        reranker_model: str = Field(
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            description="Cross-encoder model.",
        )
        reranker_top_k: int = Field(
            default=5, description="Number of results after reranking."
        )

        summarize_old_messages: bool = Field(
            default=True, description="Summarize discarded blocks."
        )
        summarization_model: str = Field(
            default="gpt-3.5-turbo", description="Default summarization model."
        )
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"),
            description="API base.",
        )
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", "dummy"), description="API key."
        )

        enable_code_awareness: bool = Field(
            default=True, description="Enable all code features."
        )
        code_similarity_threshold: float = Field(
            default=0.85, description="Duplicate similarity."
        )
        max_base_code_blocks: int = Field(
            default=3, description="Max base code blocks."
        )

        project_id: str = Field(
            default="default", description="Project identifier (shared memory)."
        )

        max_proposed_changes: int = Field(
            default=5, description="Max proposed changes."
        )
        max_committed_changes: int = Field(
            default=10, description="Max committed changes."
        )
        prioritize_recent_code: bool = Field(
            default=True, description="Keep newest version."
        )
        auto_detect_code_blocks: bool = Field(
            default=True, description="Detect fenced/indented blocks."
        )
        max_cached_projects: int = Field(
            default=10, description="Max projects in LRU cache."
        )
        track_file_paths: bool = Field(default=True, description="Extract file paths.")
        max_active_blocks: int = Field(
            default=50, description="Max active blocks per conversation."
        )
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b",
            description="Regex for file paths.",
        )

        # Oversized code block handling
        max_code_block_tokens: int = Field(
            default=20000, description="Max tokens for a code block (0=no limit)."
        )
        code_block_overflow_action: str = Field(
            default="summarize",
            description="Action for oversized blocks: 'truncate', 'summarize', 'warn'.",
        )
        code_block_summary_model: str = Field(
            default="", description="Model for summarizing oversized blocks."
        )
        code_block_truncate_keep_head: int = Field(
            default=50, description="Lines to keep from start when truncating."
        )
        code_block_truncate_keep_tail: int = Field(
            default=50, description="Lines to keep from end when truncating."
        )
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]",
            description="Replacement text for warn action.",
        )

        importance_mention_boost: float = Field(
            default=0.2, description="Mention boost."
        )
        importance_recency_half_life_hours: float = Field(
            default=2.0, description="Recency half-life."
        )

        ltm_compress_after_messages: int = Field(
            default=50, description="Messages before LTM compression."
        )
        ltm_summarization_trigger_similarity: float = Field(
            default=0.85, description="Similarity for compression."
        )

        enable_diff_application: bool = Field(
            default=True, description="Apply unified diffs."
        )
        preserve_error_context: bool = Field(
            default=True, description="Never drop error messages."
        )
        error_retention_turns: int = Field(
            default=15, description="Number of turns to keep errors."
        )
        block_expiration_hours: float = Field(
            default=24.0, description="Hours after which blocks expire."
        )
        proposed_change_retention_turns: int = Field(
            default=20, description="Turns to keep proposed changes."
        )
        preserve_tool_calls: bool = Field(
            default=True, description="Keep tool call chains."
        )

        enable_feedback_tracking: bool = Field(
            default=True, description="Record feedback about changes."
        )
        feedback_history_limit: int = Field(
            default=10, description="Max feedback entries per project."
        )
        inject_feedback_context: bool = Field(
            default=True, description="Inject feedback into system prompt."
        )
        feedback_importance_penalty_for_failure: float = Field(
            default=2.0, description="Penalty for failed changes."
        )

        code_block_pattern: str = Field(
            default="```(\\w*)\\n(.*?)```", description="Regex for fenced code blocks."
        )
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@",
            description="Diff hunk regex.",
        )
        commit_pattern: str = Field(
            default="commit\\s+([a-f0-9]{7,40})", description="Commit hash regex."
        )

        enable_dependency_tracking: bool = Field(
            default=False, description="Extract dependencies and mark affected blocks."
        )
        dependency_extraction_model: str = Field(
            default="", description="Model for dependency extraction."
        )
        dependency_refresh_on_update: bool = Field(
            default=True, description="Re-extract dependencies on change."
        )
        affected_importance_penalty: float = Field(
            default=0.7, description="Importance multiplier for affected blocks."
        )
        affected_decay_hours: float = Field(
            default=4.0, description="Hours until affected flag clears."
        )

        track_active_code_age: bool = Field(
            default=True, description="Mark code inactive after timeout."
        )
        active_code_timeout_minutes: int = Field(
            default=30, description="Timeout for active code."
        )

        summarize_inactive_code: bool = Field(
            default=True, description="Summarize inactive code blocks."
        )
        inactive_code_summary_model: str = Field(
            default="gpt-3.5-turbo", description="Model for inactive summaries."
        )

        llm_model: str = Field(
            default="",
            description="Preferred model (e.g., 'ollama/llama3.2:3b'). Falls back to summarization_model.",
        )

        enable_forget_command: bool = Field(
            default=True, description="Allow /forget commands."
        )
        enable_natural_language_forget: bool = Field(
            default=True, description="Interpret natural language forget."
        )
        natural_language_forget_model: str = Field(
            default="", description="Model for forget intent parsing."
        )

    class UserValves(BaseModel):
        max_turns: Optional[int] = Field(default=None)
        enable_code_awareness: Optional[bool] = Field(default=None)

    def __init__(self):
        self.valves = self.Valves()
        self.embedder = None
        self.chroma_client = None
        self.memory_collection = None
        self.tokenizer = None
        self._db_conn = None
        self._cross_encoder = None
        self._init_state_db()
        self._conversation_state: OrderedDict = OrderedDict()
        self._state_factory = lambda: {
            "active_blocks": {},
            "recent_changes": [],
            "committed_changes": [],
            "message_count": 0,
            "feedback_history": [],
        }
        self.code_pattern = re.compile(self.valves.code_block_pattern, re.DOTALL)
        self.diff_pattern = re.compile(self.valves.diff_pattern)
        self.commit_pattern = re.compile(self.valves.commit_pattern, re.IGNORECASE)

        if HAS_TIKTOKEN and self.valves.use_tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._log_debug("Tiktoken initialised")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")

        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._init_long_term_memory()
        else:
            logger.warning("Long‑term memory or code awareness disabled")

        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._load_reranker()

    # --------------------------------------------------------------------------
    # LRU cache
    # --------------------------------------------------------------------------
    def _get_state(self, project_id: str) -> Dict:
        if project_id in self._conversation_state:
            self._conversation_state.move_to_end(project_id)
            return self._conversation_state[project_id]
        state = self._load_state_from_db(project_id)
        if not state:
            state = self._state_factory()
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        while len(self._conversation_state) > self.valves.max_cached_projects:
            oldest = next(iter(self._conversation_state))
            self._log_debug(f"Evicting project {oldest} from cache")
            del self._conversation_state[oldest]
        return state

    def _set_state(self, project_id: str, state: Dict):
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        self._save_state_to_db(project_id, state)

    # --------------------------------------------------------------------------
    # DB
    # --------------------------------------------------------------------------
    def _init_state_db(self):
        db_path = self.valves.state_db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                project_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._log_debug(f"State DB initialised at {db_path}")

    def _get_project_id(self) -> str:
        return self.valves.project_id

    def _load_state_from_db(self, project_id: str) -> Optional[Dict]:
        cur = self._db_conn.execute(
            "SELECT state_json FROM conversation_state WHERE project_id = ?",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        if "feedback_history" not in data:
            data["feedback_history"] = []
        active = {}
        for k, v in data.get("active_blocks", {}).items():
            active[k] = CodeBlock(**v)
        recent = [CodeBlock(**b) for b in data.get("recent_changes", [])]
        committed = [CodeBlock(**b) for b in data.get("committed_changes", [])]
        feedback = [
            AppliedChangeFeedback(**fb) for fb in data.get("feedback_history", [])
        ]
        return {
            "active_blocks": active,
            "recent_changes": recent,
            "committed_changes": committed,
            "feedback_history": feedback,
            "message_count": data.get("message_count", 0),
        }

    def _save_state_to_db(self, project_id: str, state: Dict):
        serializable = {
            "active_blocks": {k: v.dict() for k, v in state["active_blocks"].items()},
            "recent_changes": [b.dict() for b in state["recent_changes"]],
            "committed_changes": [b.dict() for b in state["committed_changes"]],
            "feedback_history": [fb.dict() for fb in state["feedback_history"]],
            "message_count": state["message_count"],
        }
        self._db_conn.execute(
            "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(serializable), time.time()),
        )
        self._db_conn.commit()

    # --------------------------------------------------------------------------
    # Debug
    # --------------------------------------------------------------------------
    def _log_debug(self, msg: str):
        if self.valves.debug:
            logger.debug(msg)

    # --------------------------------------------------------------------------
    # Oversized code block handling
    # --------------------------------------------------------------------------
    def _estimate_code_tokens(self, code: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(code))
        return len(code) // 4

    async def _handle_oversized_code_block(self, code: str, language: str) -> str:
        max_tokens = self.valves.max_code_block_tokens
        if max_tokens <= 0:
            return code
        estimated = self._estimate_code_tokens(code)
        if estimated <= max_tokens:
            return code

        action = self.valves.code_block_overflow_action.lower()
        self._log_debug(
            f"Code block of {estimated} tokens exceeds limit ({max_tokens}). Action: {action}"
        )

        if action == "truncate":
            lines = code.splitlines()
            head = self.valves.code_block_truncate_keep_head
            tail = self.valves.code_block_truncate_keep_tail
            if len(lines) <= head + tail:
                return code
            truncated = "\n".join(
                lines[:head]
                + [f"... [{len(lines) - head - tail} lines truncated] ..."]
                + lines[-tail:]
            )
            return truncated

        elif action == "summarize":
            model = (
                self.valves.code_block_summary_model
                or self.valves.llm_model
                or self.valves.summarization_model
            )
            prompt = f"""Summarize the following {language} code block. Focus on:
- What the code does (purpose)
- Key functions/classes and their signatures
- Important logic or algorithms
- Any relevant external dependencies

Keep the summary concise (max 300 words).

Code:
```{language}
{code[:8000]}
```"""
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You are a code summarization assistant. Output only the summary, no extra text.",
                model_override=model,
                max_tokens=500,
                temperature=0.2,
            )
            if summary:
                return (
                    f"[Automatic summary of a {estimated} token code block]\n{summary}"
                )
            else:
                return f"[Code block too large, could not summarize] Original size: {estimated} tokens."

        elif action == "warn":
            return self.valves.code_block_warn_message

        else:
            return code

    # --------------------------------------------------------------------------
    # Dependency extraction
    # --------------------------------------------------------------------------
    async def _extract_dependencies(
        self, code: str, file_path: Optional[str] = None
    ) -> List[str]:
        if not self.valves.enable_dependency_tracking:
            return []
        model = (
            self.valves.dependency_extraction_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analyze the following code and extract dependencies:
- Import statements
- Function calls to external/user-defined functions (by name)
- Class instantiations or references
- File paths (e.g., './utils.py')

Output a JSON array of strings, each string a simple identifier or path.
If no dependencies, output [].

Code:
```{code[:1500]}```

Example output: ["os", "Path", "utils.py", "calculate_total"]
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output only JSON arrays.",
            model_override=model,
            max_tokens=300,
            temperature=0.1,
        )
        if not response:
            return []
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            deps = json.loads(response)
            if isinstance(deps, list):
                return list(set(deps))
        except:
            pass
        return []

    async def _update_dependencies(self, block_hash: str, state: Dict):
        block = state["active_blocks"].get(block_hash)
        if not block:
            return
        deps = await self._extract_dependencies(block.content, block.file_path)
        block.dependencies = deps
        self._log_debug(f"Updated dependencies for block {block_hash}: {deps}")

    async def _mark_affected_blocks(self, changed_hash: str, state: Dict):
        changed_block = state["active_blocks"].get(changed_hash)
        if not changed_block:
            return
        affected_identifiers = set()
        if changed_block.file_path:
            base = os.path.splitext(os.path.basename(changed_block.file_path))[0]
            affected_identifiers.add(base)
            affected_identifiers.add(changed_block.file_path)
        sig = self._extract_signature(changed_block.content)
        if sig:
            name_match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)", sig)
            if name_match:
                affected_identifiers.add(name_match.group(1))
        for h, block in state["active_blocks"].items():
            if h == changed_hash:
                continue
            if any(dep in affected_identifiers for dep in block.dependencies):
                block.potentially_affected = True
                block.affected_timestamp = time.time()
                block._update_importance()
                self._log_debug(
                    f"Block {h} marked as potentially affected due to dependency on {changed_hash}"
                )

    async def _refresh_dependencies_for_block(self, block_hash: str, project_id: str):
        if not self.valves.enable_dependency_tracking:
            return
        state = self._get_state(project_id)
        if not state or block_hash not in state["active_blocks"]:
            return
        await self._update_dependencies(block_hash, state)
        await self._mark_affected_blocks(block_hash, state)
        self._set_state(project_id, state)

    async def _clean_affected_flags(self, project_id: str):
        if not self.valves.enable_dependency_tracking:
            return
        state = self._get_state(project_id)
        if not state:
            return
        now = time.time()
        decay = self.valves.affected_decay_hours * 3600
        if decay <= 0:
            return
        changed = False
        for block in state["active_blocks"].values():
            if block.potentially_affected and (now - block.affected_timestamp) > decay:
                block.potentially_affected = False
                block._update_importance()
                changed = True
                self._log_debug(f"Cleared affected flag for block {block.hash}")
        if changed:
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Reranking
    # --------------------------------------------------------------------------
    def _load_reranker(self):
        if not self.valves.enable_reranking or not HAS_CROSS_ENCODER:
            return
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.valves.reranker_model)
                self._log_debug(f"Loaded reranker model {self.valves.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")
                self.valves.enable_reranking = False

    async def _rerank_results(
        self, query: str, documents: List[str], top_k: int
    ) -> List[str]:
        if not self.valves.enable_reranking or not self._cross_encoder or not documents:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = self._cross_encoder.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    # --------------------------------------------------------------------------
    # Code extraction and classification
    # --------------------------------------------------------------------------
    async def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        blocks = []
        if not self.valves.auto_detect_code_blocks:
            return blocks
        for match in self.code_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            code = await self._handle_oversized_code_block(code, lang)
            blocks.append({"language": lang, "code": code, "type": "fenced"})
        lines = content.split("\n")
        indented = []
        for line in lines:
            if line.startswith(("    ", "\t")):
                indented.append(line.lstrip(" \t"))
            else:
                if len(indented) >= 3:
                    code = "\n".join(indented)
                    code = await self._handle_oversized_code_block(code, "text")
                    blocks.append(
                        {"language": "text", "code": code, "type": "indented"}
                    )
                indented = []
        if len(indented) >= 3:
            code = "\n".join(indented)
            code = await self._handle_oversized_code_block(code, "text")
            blocks.append({"language": "text", "code": code, "type": "indented"})
        return blocks

    def _extract_line_range(
        self, content: str
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        if not self.valves.track_line_numbers:
            return None, None, None
        pattern = r"(?:^|\s)([a-zA-Z0-9_\-\./]+\.\w+):(\d+)(?:-(\d+))?"
        match = re.search(pattern, content)
        if match:
            file_path = match.group(1)
            line_start = int(match.group(2))
            line_end = int(match.group(3)) if match.group(3) else line_start
            return file_path, line_start, line_end
        return None, None, None

    def _extract_file_paths(self, content: str) -> List[str]:
        if not self.valves.track_file_paths:
            return []
        return re.findall(self.valves.file_path_pattern, content)

    def _classify_content(
        self, content: str, extracted_blocks: List[Dict]
    ) -> ContentType:
        cl = content.lower()
        if self.valves.enable_feedback_tracking:
            if re.search(
                r"\b(funcionó|sí|correcto|bien|resuelto|funciona)\b", cl
            ) and re.search(r"\b(cambio|solución|arreglo|diff)\b", cl):
                return ContentType.GENERAL
            if re.search(
                r"\b(no funcionó|falló|error|sigue igual|no resuelve|incorrecto)\b", cl
            ):
                return ContentType.GENERAL
            if content.startswith("/feedback"):
                return ContentType.GENERAL
        if self.diff_pattern.search(content) or "diff --git" in content:
            return ContentType.PROPOSED_CHANGE
        if self.commit_pattern.search(content):
            if "applied" in cl or "committed" in cl or "merged" in cl:
                return ContentType.COMMITTED_CHANGE
            return ContentType.PROPOSED_CHANGE
        if "error" in cl or "exception" in cl or "traceback" in cl:
            return ContentType.ERROR
        if '"tool_calls"' in content or '"function"' in content:
            return ContentType.TOOL_CALL
        for blk in extracted_blocks:
            if blk["language"] in [
                "python",
                "javascript",
                "typescript",
                "go",
                "rust",
                "java",
                "cpp",
            ]:
                if (
                    "def " in blk["code"]
                    or "class " in blk["code"]
                    or "function " in blk["code"]
                ):
                    return ContentType.BASE_CODE
        return ContentType.GENERAL

    # --------------------------------------------------------------------------
    # LLM helper
    # --------------------------------------------------------------------------
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        model_override: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Optional[str]:
        if not HAS_AIOHTTP:
            return None
        models_to_try = []
        if model_override:
            models_to_try.append(model_override)
        if self.valves.llm_model:
            models_to_try.append(self.valves.llm_model)
        models_to_try.append(self.valves.summarization_model)
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for model in models_to_try:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.valves.openai_api_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.valves.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=30,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"].strip()
                        else:
                            self._log_debug(
                                f"LLM call failed with model {model}, status {resp.status}"
                            )
            except Exception as e:
                self._log_debug(f"LLM model {model} error: {e}")
                continue
        logger.warning(f"All LLM models failed for prompt: {prompt[:100]}...")
        return None

    def _extract_function_names(self, code: str) -> List[str]:
        names = []
        names.extend(
            re.findall(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code, re.MULTILINE)
        )
        names.extend(
            re.findall(r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        names.extend(
            re.findall(
                r"^\s*(?:function|async function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                code,
                re.MULTILINE,
            )
        )
        names.extend(
            re.findall(
                r"^\s*(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(?",
                code,
                re.MULTILINE,
            )
        )
        names.extend(
            re.findall(r"^\s*fn\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        names.extend(
            re.findall(r"^\s*func\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
        )
        return list(set(names))

    def _extract_signature(self, code: str) -> str:
        func_match = re.search(
            r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*[^:]*)?\s*:",
            code,
            re.MULTILINE,
        )
        if func_match:
            name = func_match.group(1)
            params = func_match.group(2).strip()
            docstring = ""
            doc_match = re.search(
                r'^\s*"""(.*?)"""', code[func_match.end() :], re.DOTALL
            )
            if not doc_match:
                doc_match = re.search(
                    r"^\s*'''(.*?)'''", code[func_match.end() :], re.DOTALL
                )
            if doc_match:
                docstring = doc_match.group(1).strip()[:100]
            return (
                f"Function `{name}({params})` - {docstring}"
                if docstring
                else f"Function `{name}({params})`"
            )
        class_match = re.search(
            r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:",
            code,
            re.MULTILINE,
        )
        if class_match:
            name = class_match.group(1)
            doc_match = re.search(
                r'^\s*"""(.*?)"""', code[class_match.end() :], re.DOTALL
            )
            if not doc_match:
                doc_match = re.search(
                    r"^\s*'''(.*?)'''", code[class_match.end() :], re.DOTALL
                )
            docstring = doc_match.group(1).strip()[:100] if doc_match else ""
            return f"Class `{name}` - {docstring}" if docstring else f"Class `{name}`"
        return ""

    def _update_mentions_from_message(self, state: Dict, message_content: str):
        for block in state["active_blocks"].values():
            names = self._extract_function_names(block.content)
            if not names:
                continue
            for name in names:
                if re.search(r"\b" + re.escape(name) + r"\b", message_content):
                    block.mention_count += 1
                    block.last_mentioned = time.time()
                    block._update_importance()
                    self._log_debug(
                        f"Boosted importance of {block.hash} due to mention of '{name}'"
                    )
                    break

    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        if not HAS_FUZZ:
            min_len = min(len(code1), len(code2))
            if min_len == 0:
                return 0.0
            common = sum(1 for a, b in zip(code1[:min_len], code2[:min_len]) if a == b)
            return common / max(len(code1), len(code2))
        return fuzz.token_sort_ratio(code1, code2) / 100.0

    # --------------------------------------------------------------------------
    # Diff application
    # --------------------------------------------------------------------------
    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        if not self.valves.enable_diff_application:
            return None
        lines = original.splitlines(keepends=False)
        result_lines = lines[:]
        hunks = []
        for match in re.finditer(
            r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*?)(?=@@|\Z)", diff_text, re.DOTALL
        ):
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            hunk_body = match.group(5).strip("\n")
            old_lines = []
            new_lines = []
            for line in hunk_body.split("\n"):
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
            hunks.append((old_start - 1, old_lines, new_lines))
        for old_start, old_lines, new_lines in reversed(hunks):
            if result_lines[old_start : old_start + len(old_lines)] != old_lines:
                self._log_debug("Diff hunk does not match current code; skipping.")
                continue
            result_lines = (
                result_lines[:old_start]
                + new_lines
                + result_lines[old_start + len(old_lines) :]
            )
        return "\n".join(result_lines)

    def _apply_change_with_diff(
        self, base_block: CodeBlock, proposed_block: CodeBlock
    ) -> bool:
        if proposed_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        if not (
            "@@" in proposed_block.content
            and ("-" in proposed_block.content or "+" in proposed_block.content)
        ):
            return False
        new_code = self._apply_unified_diff(base_block.content, proposed_block.content)
        if new_code and new_code != base_block.content:
            base_block.content = new_code
            base_block.hash = hashlib.md5(new_code.encode()).hexdigest()[:16]
            base_block.timestamp = time.time()
            base_block.is_active = True
            base_block.potentially_affected = False
            base_block.importance_score = min(base_block.importance_score + 2.0, 10.0)
            self._log_debug(f"Applied diff to base block {base_block.hash}")
            return True
        return False

    # --------------------------------------------------------------------------
    # Conflict detection
    # --------------------------------------------------------------------------
    def _has_conflicting_proposed_changes(
        self, state: Dict, new_block: CodeBlock
    ) -> bool:
        if new_block.content_type != ContentType.PROPOSED_CHANGE:
            return False
        for existing in state["recent_changes"]:
            if existing.hash == new_block.hash:
                continue
            same_file = (
                existing.file_path
                and new_block.file_path
                and existing.file_path == new_block.file_path
            )
            if (
                same_file
                or self._calculate_code_similarity(existing.content, new_block.content)
                > 0.8
            ):
                self._log_debug(
                    f"Conflict detected between proposed changes {existing.hash} and {new_block.hash}"
                )
                return True
        return False

    # --------------------------------------------------------------------------
    # Expiration
    # --------------------------------------------------------------------------
    async def _expire_blocks_by_time(self, project_id: str):
        state = self._get_state(project_id)
        if not state:
            return
        now = time.time()
        expiration_seconds = self.valves.block_expiration_hours * 3600
        to_remove = []
        for h, block in state["active_blocks"].items():
            age = now - block.last_mentioned
            if (
                block.content_type == ContentType.ERROR
                and self.valves.error_retention_turns > 0
            ):
                turn_time = self.valves.error_retention_turns * 300
                if age > max(turn_time, expiration_seconds):
                    to_remove.append(h)
            elif (
                block.content_type == ContentType.PROPOSED_CHANGE
                and self.valves.proposed_change_retention_turns > 0
            ):
                turn_time = self.valves.proposed_change_retention_turns * 300
                if age > max(turn_time, expiration_seconds):
                    to_remove.append(h)
        for h in to_remove:
            del state["active_blocks"][h]
            self._log_debug(f"Expired block {h}")
        if to_remove:
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Summarization helpers
    # --------------------------------------------------------------------------
    async def _summarize_inactive_blocks_safely(self, project_id: str):
        if not self.valves.summarize_inactive_code:
            return
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return
        now = time.time()
        timeout = self.valves.active_code_timeout_minutes * 60
        to_summarize = []
        for h, block in state["active_blocks"].items():
            if (
                not block.is_active
                and (now - block.timestamp) > timeout
                and block.importance_score < 5.0
            ):
                to_summarize.append((h, block))
        if not to_summarize:
            return
        tasks = [self._summarize_code_block(block) for _, block in to_summarize]
        summaries = await asyncio.gather(*tasks)
        for (h, block), summary in zip(to_summarize, summaries):
            if summary:
                sig = self._extract_signature(block.content)
                if sig:
                    summary = f"{sig}\n\n{summary}"
                summary_block = CodeBlock(
                    content=f"[Summary of inactive code]\n{summary}",
                    content_type=ContentType.GENERAL,
                    timestamp=time.time(),
                    is_active=False,
                    importance_score=block.importance_score * 0.5,
                )
                state["active_blocks"][h] = summary_block
                self._log_debug(f"Summarized inactive block {h}")
        self._set_state(project_id, state)

    async def _summarize_code_block(self, block: CodeBlock) -> Optional[str]:
        if not self.valves.summarize_inactive_code or not HAS_AIOHTTP:
            return None
        sig = self._extract_signature(block.content)
        if sig:
            prompt = f"""The code block has signature: {sig}
Provide a very brief (max 50 words) description of what this code does.
Code:
```{block.content[:1000]}```
"""
        else:
            prompt = f"""Summarize the following code block. Include:
1. What the code does (purpose)
2. Key functions/classes/variables
3. Any important logic or edge cases
Keep the summary under 150 words.

```{block.content[:1500]}```
"""
        return await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant.",
            model_override=self.valves.inactive_code_summary_model,
            max_tokens=200,
            temperature=0.2,
        )

    # --------------------------------------------------------------------------
    # Active code tracking (main)
    # --------------------------------------------------------------------------
    def _update_active_code(self, message: dict, project_id: str):
        if not self.valves.enable_code_awareness:
            return
        asyncio.create_task(self._summarize_inactive_blocks_safely(project_id))
        state = self._get_state(project_id)
        content = message.get("content", "")
        role = message.get("role", "")

        self._update_mentions_from_message(state, content)

        for block in state["active_blocks"].values():
            if (
                block.content
                and self._calculate_code_similarity(block.content[:200], content[:200])
                > 0.7
            ):
                block.mention_count += 1
                block.last_mentioned = time.time()
                block._update_importance()

        extracted = asyncio.run(self._extract_code_blocks(content))
        if not content and not extracted:
            return

        content_type = self._classify_content(content, extracted)
        file_path, line_start, line_end = self._extract_line_range(content)

        for block_info in extracted:
            blk_file = file_path or (
                self._extract_file_paths(content)[0]
                if self.valves.track_file_paths
                else None
            )
            new_block = CodeBlock(
                content=block_info["code"],
                content_type=content_type,
                generated_by_assistant=(role == "assistant"),
                file_path=blk_file,
                line_range=(line_start, line_end) if line_start and line_end else None,
                timestamp=time.time(),
                is_active=True,
                mention_count=1,
                dependencies=[],
                potentially_affected=False,
            )

            if "[KEEP]" in content or "#important" in content.lower():
                new_block.importance_score = 10.0
                self._log_debug(
                    f"Manual importance marker detected for block {new_block.hash}"
                )

            is_dup, existing = self._is_duplicate_code(
                new_block, list(state["active_blocks"].values())
            )
            if is_dup and existing:
                if self.valves.prioritize_recent_code:
                    existing.content = new_block.content
                    existing.hash = new_block.hash
                    existing.timestamp = time.time()
                    existing.mention_count += 1
                    existing.last_mentioned = time.time()
                    existing._update_importance()
                    self._log_debug(f"Updated existing block {existing.hash}")
                    if (
                        self.valves.enable_dependency_tracking
                        and self.valves.dependency_refresh_on_update
                    ):
                        asyncio.create_task(
                            self._refresh_dependencies_for_block(
                                existing.hash, project_id
                            )
                        )
                continue

            if (
                content_type == ContentType.PROPOSED_CHANGE
                and self._has_conflicting_proposed_changes(state, new_block)
            ):
                new_block.importance_score = max(new_block.importance_score, 7.0)
                self._log_debug(
                    f"Proposed change {new_block.hash} marked as conflicting"
                )

            state["active_blocks"][new_block.hash] = new_block
            self._log_debug(f"New {content_type.value} block: {new_block.hash}")

            if content_type == ContentType.PROPOSED_CHANGE:
                state["recent_changes"].append(new_block)
                conflict = self._has_conflicting_proposed_changes(state, new_block)
                if self.valves.enable_diff_application and not conflict:
                    for base in list(state["active_blocks"].values()):
                        if (
                            base.content_type == ContentType.BASE_CODE
                            and base.file_path == new_block.file_path
                        ):
                            if self._apply_change_with_diff(base, new_block):
                                state["recent_changes"] = [
                                    c
                                    for c in state["recent_changes"]
                                    if c.hash != new_block.hash
                                ]
                                state["committed_changes"].append(new_block)
                                break
                else:
                    self._log_debug(
                        f"Skipped auto-apply for conflicting proposed change {new_block.hash}"
                    )
            elif content_type == ContentType.COMMITTED_CHANGE:
                state["committed_changes"].append(new_block)
            elif (
                content_type == ContentType.ERROR and self.valves.preserve_error_context
            ):
                new_block.importance_score = min(new_block.importance_score + 3.0, 10.0)

            if len(state["recent_changes"]) > self.valves.max_proposed_changes:
                state["recent_changes"] = sorted(
                    state["recent_changes"],
                    key=lambda b: b.importance_score,
                    reverse=True,
                )[: self.valves.max_proposed_changes]
            if len(state["committed_changes"]) > self.valves.max_committed_changes:
                state["committed_changes"] = sorted(
                    state["committed_changes"],
                    key=lambda b: b.importance_score,
                    reverse=True,
                )[: self.valves.max_committed_changes]
            if len(state["active_blocks"]) > self.valves.max_active_blocks:
                sorted_blocks = sorted(
                    state["active_blocks"].values(),
                    key=lambda b: b.importance_score,
                    reverse=True,
                )
                keep = sorted_blocks[: self.valves.max_active_blocks]
                state["active_blocks"] = {b.hash: b for b in keep}

            # Extract dependencies for new blocks (non-blocking)
            if self.valves.enable_dependency_tracking and content_type in (
                ContentType.BASE_CODE,
                ContentType.PROPOSED_CHANGE,
                ContentType.COMMITTED_CHANGE,
            ):
                asyncio.create_task(
                    self._refresh_dependencies_for_block(new_block.hash, project_id)
                )

        # Assistant learning: update best matching base block
        if role == "assistant" and len(extracted) > 0:
            for block_info in extracted:
                best_base = None
                best_sim = 0.0
                for base in state["active_blocks"].values():
                    if base.content_type == ContentType.BASE_CODE:
                        if base.file_path and file_path and base.file_path == file_path:
                            sim = self._calculate_code_similarity(
                                base.content, block_info["code"]
                            )
                            if sim > best_sim:
                                best_sim = sim
                                best_base = base
                        else:
                            sim = self._calculate_code_similarity(
                                base.content, block_info["code"]
                            )
                            if sim > best_sim and sim > 0.6:
                                best_sim = sim
                                best_base = base
                if best_base and best_sim > 0.6 and best_sim < 0.95:
                    best_base.content = block_info["code"]
                    best_base.hash = hashlib.md5(
                        block_info["code"].encode()
                    ).hexdigest()[:16]
                    best_base.timestamp = time.time()
                    best_base.is_active = True
                    best_base.potentially_affected = False
                    best_base.importance_score = min(
                        best_base.importance_score + 1.0, 10.0
                    )
                    self._log_debug(
                        f"Assistant updated base code block {best_base.hash} (sim={best_sim:.2f})"
                    )
                    if (
                        self.valves.enable_dependency_tracking
                        and self.valves.dependency_refresh_on_update
                    ):
                        asyncio.create_task(
                            self._refresh_dependencies_for_block(
                                best_base.hash, project_id
                            )
                        )

        state["message_count"] += 1
        asyncio.create_task(self._expire_blocks_by_time(project_id))
        asyncio.create_task(self._clean_affected_flags(project_id))
        self._set_state(project_id, state)

    def _is_duplicate_code(
        self, new_block: CodeBlock, existing_blocks: List[CodeBlock]
    ) -> Tuple[bool, Optional[CodeBlock]]:
        for ex in existing_blocks:
            sim = self._calculate_code_similarity(new_block.content, ex.content)
            if sim >= self.valves.code_similarity_threshold:
                return True, ex
        return False, None

    def _get_active_code_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        now = time.time()
        active = []
        for block in state["active_blocks"].values():
            if not block.is_active and self.valves.track_active_code_age:
                if now - block.timestamp > self.valves.active_code_timeout_minutes * 60:
                    continue
            active.append(block)
        if not active:
            return ""
        active.sort(key=lambda b: b.importance_score, reverse=True)
        base_codes = [b for b in active if b.content_type == ContentType.BASE_CODE][
            : self.valves.max_base_code_blocks
        ]
        proposed = [b for b in active if b.content_type == ContentType.PROPOSED_CHANGE][
            : self.valves.max_proposed_changes
        ]
        committed = [
            b for b in active if b.content_type == ContentType.COMMITTED_CHANGE
        ][: self.valves.max_committed_changes]
        errors = (
            [b for b in active if b.content_type == ContentType.ERROR][:3]
            if self.valves.preserve_error_context
            else []
        )

        parts = ["## Currently Active Code Context (by importance)\n"]
        if base_codes:
            parts.append("### Base Code (current work):")
            for b in base_codes:
                loc = (
                    f" (file: {b.file_path}"
                    + (
                        f", lines {b.line_range[0]}-{b.line_range[1]}"
                        if b.line_range
                        else ""
                    )
                    + ")"
                    if b.file_path
                    else ""
                )
                aff = (
                    " [AFFECTED BY DEPENDENCY CHANGE]" if b.potentially_affected else ""
                )
                parts.append(
                    f"```\n{b.content[:600]}\n```{loc}  (importance: {b.importance_score:.1f}){aff}"
                )
        if proposed:
            parts.append("### Proposed Changes (pending review):")
            for b in proposed:
                parts.append(f"```diff\n{b.content[:500]}\n```")
        if committed:
            parts.append("### Recently Committed Changes:")
            for b in committed:
                parts.append(f"```\n{b.content[:300]}\n```")
        if errors:
            parts.append("### Recent Errors:")
            for b in errors:
                parts.append(f"```\n{b.content[:500]}\n```")
        return "\n".join(parts)

    # --------------------------------------------------------------------------
    # Token estimation
    # --------------------------------------------------------------------------
    def _estimate_tokens(self, messages: List[dict]) -> int:
        if self.tokenizer:
            total = 0
            for m in messages:
                content = str(m.get("content", ""))
                total += len(self.tokenizer.encode(content))
                total += 4
            return total
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        total_chars += sum(30 for _ in messages)
        return total_chars // 4

    # --------------------------------------------------------------------------
    # LTM with compression
    # --------------------------------------------------------------------------
    def _init_long_term_memory(self):
        os.makedirs(self.valves.long_term_memory_dir, exist_ok=True)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(
            path=self.valves.long_term_memory_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="conversation_memory", metadata={"hnsw:space": "cosine"}
        )
        self._purge_expired_memories()
        self._log_debug("LTM ready")

    def _purge_expired_memories(self):
        if not HAS_CHROMA or self.memory_collection is None:
            return
        if self.valves.long_term_memory_expiration_days <= 0:
            return
        try:
            now = datetime.utcnow().isoformat()
            expired = self.memory_collection.get(where={"expires_at": {"$lt": now}})
            if expired and expired["ids"]:
                self.memory_collection.delete(ids=expired["ids"])
                self._log_debug(f"Purged {len(expired['ids'])} expired memories")
        except Exception as e:
            logger.warning(f"Purge failed: {e}")

    def _store_message_in_memory(self, message: dict, project_id: str):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        content = message.get("content", "")
        if not content or len(content.strip()) < 15:
            return
        extracted = asyncio.run(self._extract_code_blocks(content))
        content_type = self._classify_content(content, extracted)
        msg_id = f"{project_id}_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        expires_at = None
        if self.valves.long_term_memory_expiration_days > 0:
            expires_at = (
                datetime.utcnow()
                + timedelta(days=self.valves.long_term_memory_expiration_days)
            ).isoformat()
        embedding = self.embedder.encode(content).tolist()
        self.memory_collection.upsert(
            ids=[msg_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "role": message.get("role"),
                    "project_id": project_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "expires_at": expires_at,
                    "content_type": content_type.value,
                    "has_code": len(extracted) > 0,
                }
            ],
            documents=[content],
        )
        self._log_debug(f"Stored message {msg_id} in LTM")

        state = self._get_state(project_id)
        msg_count = state.get("message_count", 0)
        if msg_count > 0 and msg_count % self.valves.ltm_compress_after_messages == 0:
            asyncio.create_task(self._compress_ltm_for_conversation(project_id))

    async def _compress_ltm_for_conversation(self, project_id: str):
        if not HAS_AIOHTTP or not self.memory_collection:
            return
        try:
            results = self.memory_collection.get(where={"project_id": project_id})
            if (
                not results
                or len(results["ids"]) < self.valves.ltm_compress_after_messages
            ):
                return
            ids = results["ids"]
            docs = results["documents"]
            metadatas = results["metadatas"]
            pairs = sorted(
                zip(ids, docs, metadatas), key=lambda x: x[2].get("timestamp", "")
            )
            to_compress = pairs[: max(len(pairs) // 3, 5)]
            if len(to_compress) < 2:
                return
            texts = "\n---\n".join([doc for _, doc, _ in to_compress])
            prompt = f"Summarise the following conversation segment, keeping key technical decisions and code changes:\n\n{texts[:3000]}"
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="You produce concise, information-dense summaries.",
                max_tokens=500,
                temperature=0.3,
            )
            if summary:
                self.memory_collection.delete(ids=[id for id, _, _ in to_compress])
                summary_id = f"{project_id}_summary_{int(time.time())}"
                summary_embedding = self.embedder.encode(summary).tolist()
                self.memory_collection.upsert(
                    ids=[summary_id],
                    embeddings=[summary_embedding],
                    metadatas=[
                        {
                            "project_id": project_id,
                            "is_summary": True,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ],
                    documents=[summary],
                )
                self._log_debug(
                    f"Compressed {len(to_compress)} messages into summary for {project_id}"
                )
        except Exception as e:
            logger.warning(f"LTM compression failed: {e}")

    async def _retrieve_relevant_memories(
        self,
        query: str,
        project_id: str,
        content_type_filter: Optional[ContentType] = None,
    ) -> List[str]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []
        try:
            q_emb = self.embedder.encode(query[:1000]).tolist()
            where_filter = {"project_id": project_id}
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["expires_at"] = {"$gt": datetime.utcnow().isoformat()}
            if content_type_filter:
                where_filter["content_type"] = content_type_filter.value
            results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=(
                    self.valves.long_term_memory_top_k * 2
                    if self.valves.enable_reranking
                    else self.valves.long_term_memory_top_k
                ),
                where=where_filter,
            )
            retrieved = []
            docs_with_scores = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    sim = 1 - results["distances"][0][i]
                    if self.valves.ltm_time_decay_hours > 0 and results["metadatas"]:
                        age_str = results["metadatas"][0][i].get("timestamp", "")
                        if age_str:
                            age_hours = (
                                datetime.utcnow() - datetime.fromisoformat(age_str)
                            ).total_seconds() / 3600
                            sim *= 0.5 ** (age_hours / self.valves.ltm_time_decay_hours)
                    if sim >= self.valves.long_term_memory_similarity_threshold:
                        docs_with_scores.append((doc, sim))
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            retrieved = [doc for doc, _ in docs_with_scores]
            if self.valves.enable_reranking and self._cross_encoder:
                rerank_k = (
                    self.valves.reranker_top_k
                    if self.valves.reranker_top_k > 0
                    else self.valves.long_term_memory_top_k
                )
                rerank_k = min(rerank_k, 50)
                if len(retrieved) > 0:
                    retrieved = await self._rerank_results(
                        query, retrieved[: rerank_k * 2], rerank_k
                    )
            else:
                retrieved = retrieved[: self.valves.long_term_memory_top_k]
            return retrieved
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    # --------------------------------------------------------------------------
    # Feedback handling
    # --------------------------------------------------------------------------
    async def _parse_feedback_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_feedback_tracking:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""You are a feedback interpreter. The user is giving feedback about a code change that was applied.
Possible outcomes:
- success: the change worked, problem solved
- failure: the change did not work, error remains
- neutral: unclear or not feedback.

User message: "{user_message}"

If feedback, output JSON: {{"action": "feedback", "outcome": "success/failure", "comment": "..."}}
If not feedback: {{"action": "none"}}

Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") == "feedback":
                self._log_debug(f"Parsed feedback intent: {data}")
                return data
        except:
            pass
        return None

    async def _record_feedback(self, project_id: str, outcome: str, comment: str):
        state = self._get_state(project_id)
        if not state or not state["committed_changes"]:
            self._log_debug("No committed changes to associate feedback with.")
            return
        last_commit = max(state["committed_changes"], key=lambda c: c.timestamp)
        success = outcome == "success"
        feedback = AppliedChangeFeedback(
            change_hash=last_commit.hash,
            change_description=last_commit.content[:200],
            file_path=last_commit.file_path,
            timestamp=time.time(),
            success=success,
            user_comment=comment,
        )
        state["feedback_history"].append(feedback)
        if len(state["feedback_history"]) > self.valves.feedback_history_limit:
            state["feedback_history"] = state["feedback_history"][
                -self.valves.feedback_history_limit :
            ]
        if not success:
            last_commit.importance_score /= (
                self.valves.feedback_importance_penalty_for_failure
            )
            last_commit.importance_score = max(0.5, last_commit.importance_score)
            self._log_debug(
                f"Reduced importance of failed change {last_commit.hash} to {last_commit.importance_score:.1f}"
            )
        else:
            last_commit.importance_score = min(10.0, last_commit.importance_score + 1.0)
        self._set_state(project_id, state)
        self._log_debug(f"Recorded feedback for change {last_commit.hash}: {outcome}")

    def _get_feedback_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["feedback_history"]:
            return ""
        lines = ["## Recent Change Feedback\n"]
        for fb in state["feedback_history"][-5:]:
            status = "✅ SUCCESS" if fb.success else "❌ FAILED"
            desc = fb.change_description.replace("\n", " ")[:100]
            lines.append(f'- {status}: `{desc}` - User: "{fb.user_comment}"')
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Summarization helper for messages
    # --------------------------------------------------------------------------
    async def _summarize_messages(
        self, messages: List[dict], is_code_context: bool = False
    ) -> Optional[str]:
        if not HAS_AIOHTTP or not messages:
            return None
        conv_text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in messages
        )
        if is_code_context:
            prompt = f"Summarise technical conversation. Keep code, decisions, actions, and unresolved issues.\n\n{conv_text}"
        else:
            prompt = f"Summarise conversation, keep key points.\n\n{conv_text}"
        return await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code-aware assistant.",
            max_tokens=500,
            temperature=0.3,
        )

    # --------------------------------------------------------------------------
    # Forget command handling
    # --------------------------------------------------------------------------
    async def _parse_forget_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_natural_language_forget:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpret forget intent. Possible actions: forget_last, forget_n (with n), forget_file (with file), forget_block (with hash), forget_all.
User: "{user_message}"
Output JSON: {{"action": "...", "n": N, "file": "...", "hash": "..."}} or {{"action": "none"}}
Output only JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You output JSON only.",
            model_override=model,
            max_tokens=150,
            temperature=0.1,
        )
        if not response:
            return None
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response)
            if data.get("action") != "none":
                self._log_debug(f"Parsed forget intent: {data}")
                return data
        except:
            pass
        return None

    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
        state = self._get_state(project_id)
        if not state:
            return "No active context to forget."
        action = intent.get("action")
        if action == "forget_last":
            if state["active_blocks"]:
                last_hash = max(
                    state["active_blocks"].keys(),
                    key=lambda h: state["active_blocks"][h].timestamp,
                )
                del state["active_blocks"][last_hash]
                self._log_debug(f"Forget last: removed block {last_hash}")
            return "Olvidado el último bloque de contexto."
        elif action == "forget_n":
            n = intent.get("n", 1)
            blocks_by_time = sorted(
                state["active_blocks"].items(),
                key=lambda x: x[1].timestamp,
                reverse=True,
            )
            removed = 0
            for h, _ in blocks_by_time[:n]:
                if h in state["active_blocks"]:
                    del state["active_blocks"][h]
                    removed += 1
            return f"Olvidados los últimos {removed} bloques de contexto."
        elif action == "forget_file":
            file_path = intent.get("file", "")
            if not file_path:
                return "No se especificó archivo."
            to_remove = [
                h
                for h, blk in state["active_blocks"].items()
                if blk.file_path and file_path in blk.file_path
            ]
            for h in to_remove:
                del state["active_blocks"][h]
            return f"Olvidados {len(to_remove)} bloques relacionados con {file_path}."
        elif action == "forget_block":
            block_id = intent.get("hash") or intent.get("id") or ""
            if not block_id:
                return "No se especificó bloque."
            if block_id in state["active_blocks"]:
                del state["active_blocks"][block_id]
                return f"Olvidado bloque {block_id}."
            matches = [h for h in state["active_blocks"] if block_id in h]
            if matches:
                for h in matches:
                    del state["active_blocks"][h]
                return f"Olvidados {len(matches)} bloques que coinciden con {block_id}."
            return f"No se encontró bloque {block_id}."
        elif action == "forget_all":
            state["active_blocks"].clear()
            state["recent_changes"].clear()
            state["committed_changes"].clear()
            return "Olvidado todo el contexto."
        else:
            return "No se pudo interpretar la intención de olvido."

    async def _handle_forget_command(
        self, messages: List[dict], project_id: str, __user__: Optional[dict]
    ) -> Tuple[List[dict], bool]:
        if not (
            self.valves.enable_forget_command
            or self.valves.enable_natural_language_forget
        ):
            return messages, False
        if not messages:
            return messages, False
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return messages, False
        content = last_msg.get("content", "").strip()

        if self.valves.enable_natural_language_forget:
            intent = await self._parse_forget_intent(content)
            if intent and intent.get("action") != "none":
                confirmation = await self._execute_forget_intent(project_id, intent)
                self._set_state(project_id, self._get_state(project_id))
                messages.pop()
                messages.append({"role": "assistant", "content": confirmation})
                return messages, True

        if self.valves.enable_forget_command and content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            state = self._get_state(project_id)
            if not state:
                return messages, False
            if target == "all":
                state["active_blocks"].clear()
                state["recent_changes"].clear()
                state["committed_changes"].clear()
                confirmation = "Olvidado todo el contexto."
            elif target == "last":
                if state["active_blocks"]:
                    last_hash = max(
                        state["active_blocks"].keys(),
                        key=lambda h: state["active_blocks"][h].timestamp,
                    )
                    del state["active_blocks"][last_hash]
                    confirmation = "Olvidado el último bloque de contexto."
                else:
                    confirmation = "No hay bloques para olvidar."
            else:
                to_remove = [
                    h
                    for h, blk in state["active_blocks"].items()
                    if (blk.file_path and target in blk.file_path) or target in h
                ]
                for h in to_remove:
                    del state["active_blocks"][h]
                confirmation = (
                    f"Olvidados {len(to_remove)} bloques que coinciden con '{target}'."
                )
            self._set_state(project_id, state)
            messages.pop()
            messages.append({"role": "assistant", "content": confirmation})
            return messages, True

        return messages, False

    # --------------------------------------------------------------------------
    # Inlet
    # --------------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("inlet called")
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        self._log_debug(f"Project ID: {project_id}")
        if not messages:
            return body

        state = self._get_state(project_id)

        if (
            self.valves.enable_forget_command
            or self.valves.enable_natural_language_forget
        ):
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                body["messages"] = new_messages
                return body

        if self.valves.enable_feedback_tracking:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
                feedback_intent = await self._parse_feedback_intent(
                    last_user_msg.get("content", "")
                )
                if feedback_intent and feedback_intent.get("action") == "feedback":
                    await self._record_feedback(
                        project_id,
                        feedback_intent.get("outcome"),
                        feedback_intent.get("comment", ""),
                    )

        if self.valves.enable_code_awareness:
            for msg in messages[-5:]:
                self._update_active_code(msg, project_id)

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if (
            last_user_msg
            and HAS_SENTENCE
            and HAS_CHROMA
            and self.valves.enable_code_awareness
        ):
            query = last_user_msg.get("content", "")
            base_mem = await self._retrieve_relevant_memories(
                query, project_id, ContentType.BASE_CODE
            )
            error_mem = (
                await self._retrieve_relevant_memories(
                    query, project_id, ContentType.ERROR
                )
                if self.valves.preserve_error_context
                else []
            )
            general_mem = await self._retrieve_relevant_memories(
                query, project_id, None
            )
            all_mem = list(dict.fromkeys(base_mem + error_mem + general_mem))[:5]
            if all_mem:
                ctx = "## Relevant Past Context\n\n" + "\n---\n".join(all_mem)
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = ctx + "\n\n" + sys_msgs[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": ctx})
                body["messages"] = messages

        if self.valves.enable_code_awareness:
            active_ctx = self._get_active_code_context(project_id)
            if active_ctx:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = (
                        active_ctx + "\n\n" + sys_msgs[0]["content"]
                    )
                else:
                    messages.insert(0, {"role": "system", "content": active_ctx})
                body["messages"] = messages

        if self.valves.enable_feedback_tracking and self.valves.inject_feedback_context:
            feedback_ctx = self._get_feedback_context(project_id)
            if feedback_ctx:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = (
                        feedback_ctx + "\n\n" + sys_msgs[0]["content"]
                    )
                else:
                    messages.insert(0, {"role": "system", "content": feedback_ctx})
                body["messages"] = messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        trim_needed = False
        if self.valves.adaptive_trim:
            total_tokens = self._estimate_tokens(system_msgs + history_msgs)
            if total_tokens > self.valves.context_window_tokens:
                trim_needed = True
        else:
            user_max = (
                __user__["valves"].get("max_turns")
                if __user__ and "valves" in __user__
                else None
            )
            eff_max = user_max if user_max is not None else self.valves.max_turns
            if len(history_msgs) > eff_max:
                trim_needed = True

        if trim_needed and len(history_msgs) > self.valves.max_turns:
            keep = self.valves.max_turns
            old_block = history_msgs[:-keep] if keep > 0 else []
            kept_block = history_msgs[-keep:] if keep > 0 else []
            if self.valves.summarize_old_messages and old_block:
                has_code = any("```" in m.get("content", "") for m in old_block)
                summary = await self._summarize_messages(
                    old_block, is_code_context=has_code
                )
                if summary:
                    history_msgs = [
                        {
                            "role": "assistant",
                            "content": f"[Summary of earlier conversation]\n{summary}",
                        }
                    ] + kept_block
                else:
                    history_msgs = kept_block
            else:
                history_msgs = kept_block
            if self.valves.preserve_tool_calls:
                while history_msgs and history_msgs[0].get("role") == "tool":
                    history_msgs.pop(0)
                if (
                    history_msgs
                    and history_msgs[0].get("role") == "assistant"
                    and history_msgs[0].get("tool_calls")
                ):
                    tool_call_ids = {
                        tc.get("id") for tc in history_msgs[0]["tool_calls"]
                    }
                    tool_response_ids = {
                        m.get("tool_call_id")
                        for m in history_msgs[1:]
                        if m.get("role") == "tool"
                    }
                    if not tool_call_ids.issubset(tool_response_ids):
                        history_msgs.pop(0)

        body["messages"] = system_msgs + history_msgs
        return body

    # --------------------------------------------------------------------------
    # Outlet
    # --------------------------------------------------------------------------
    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("outlet called")
        if not (HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness):
            return body
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                self._update_active_code(msg, project_id)
                self._store_message_in_memory(msg, project_id)
        self._purge_expired_memories()
        return body
