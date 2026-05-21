"""
title: Code-Aware Context Manager with LTM & Summarization (v5.23.2)
description: Full-featured context manager for coding assistants. Persists state per project, tracks line ranges, applies diffs, compresses LTM, scores importance, learns from responses, summarizes inactive code, supports manual markers, natural language forget/remember commands, feedback tracking, hierarchical memory, LRU cache, optional reranking, dependency detection (AST for Python + regex for other languages), handling of oversized blocks, smart context selection, hierarchical compression, duplicate removal, frequency prioritization, selective summarization, iterative commands, consecutive message deduplication, contradiction detection, chain-of-thought reasoning, assumption extraction, obsolete marking, proactive suggestions, duplicate question detection, command suggestions, and semantic response caching.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.23.2
license: GPL3
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb, rapidfuzz
"""

import os
import time
import re
import hashlib
import sqlite3
import ast
from collections import OrderedDict, defaultdict
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
    pinned: bool = False
    affected_timestamp: float = 0.0
    obsolete: bool = False
    ast_imports: List[str] = Field(default_factory=list)
    ast_calls: List[str] = Field(default_factory=list)

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
        if self.obsolete:
            penalty = 0.1
            self.is_active = False
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
        priority: int = Field(default=0, description="Filter priority level.")
        max_turns: int = Field(
            default=20, description="Max non-system messages to keep."
        )
        debug: bool = Field(default=False, description="Enable verbose debug logging.")
        state_db_path: str = Field(
            default="/app/backend/data/conversation_state.db",
            description="SQLite DB path.",
        )
        track_line_numbers: bool = Field(
            default=True, description="Extract line ranges from files."
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
            default="/app/backend/data/long_term_memory",
            description="ChromaDB directory.",
        )
        long_term_memory_expiration_days: int = Field(
            default=30, description="Days until LTM entry expires."
        )
        long_term_memory_top_k: int = Field(
            default=10, description="Number of results to retrieve from LTM."
        )
        long_term_memory_similarity_threshold: float = Field(
            default=0.65, description="Minimum cosine similarity threshold."
        )
        ltm_time_decay_hours: float = Field(
            default=24.0, description="Time decay for LTM retrieval."
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

        smart_context_selection: bool = Field(
            default=False,
            description="Replace sliding window with semantic retrieval from conversation history.",
        )
        smart_context_top_k: int = Field(
            default=15, description="Number of past messages to retrieve."
        )
        smart_context_min_tokens: int = Field(
            default=1024, description="Minimum tokens to aim for."
        )
        smart_context_include_last_user: bool = Field(
            default=True, description="Always include the most recent user message."
        )

        hierarchical_compression_enabled: bool = Field(
            default=False,
            description="Periodically compress old conversation segments.",
        )
        hierarchical_compression_interval_messages: int = Field(
            default=100,
            description="Number of messages after which to trigger compression.",
        )
        hierarchical_summary_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Model for hierarchical summaries.",
        )
        hierarchical_summary_max_tokens: int = Field(
            default=800, description="Max tokens for hierarchical summary."
        )

        auto_remove_duplicate_blocks: bool = Field(
            default=True, description="Auto-remove older duplicate code blocks."
        )
        max_duplicate_age_hours: float = Field(
            default=6.0, description="Max age difference for considering duplicates."
        )
        frequency_weight_factor: float = Field(
            default=0.3, description="Weight for mention frequency in importance."
        )
        min_mentions_for_boost: int = Field(
            default=3, description="Min mentions to apply frequency boost."
        )
        frequency_decay_hours: float = Field(
            default=12.0, description="Half-life for frequency boost."
        )

        enable_confidence_scoring: bool = Field(
            default=True, description="Ask assistant to estimate confidence (0-100%)."
        )
        confidence_prompt: str = Field(
            default="\n\nAfter your response, on a new line, output '[Confidence: XX%]' where XX is your estimated confidence (0-100) in the correctness and completeness of your answer, based on the available context. If you lack information, give lower confidence and suggest what context would help.",
            description="Suffix added to system prompt to request confidence.",
        )
        enable_cot_on_demand: bool = Field(
            default=True, description="Enable /think command for chain-of-thought."
        )
        cot_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest",
            description="Model for chain-of-thought reasoning.",
        )
        cot_max_tokens: int = Field(
            default=1000, description="Max tokens for chain-of-thought response."
        )
        enable_assumption_extraction: bool = Field(
            default=True, description="Enable /assume command."
        )
        assumption_extraction_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest",
            description="Model for assumption extraction.",
        )
        enable_contradiction_detection: bool = Field(
            default=True, description="Detect contradictions in conversation."
        )
        contradiction_detection_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest",
            description="Model for contradiction detection.",
        )
        contradiction_inject_warning: bool = Field(
            default=True, description="Inject contradiction warning into system prompt."
        )
        proactive_context_warning_threshold: float = Field(
            default=0.85,
            description="Token usage ratio that triggers a proactive warning.",
        )
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Context Warning**: The conversation is using more than {percent}% of the available context window ({used_tokens}/{max_tokens} tokens). Consider using `/forget` to remove irrelevant parts, `/remember` to pin important context, or ask me to summarize older parts.",
            description="Warning message injected when context is nearly full.",
        )
        enable_facts: bool = Field(
            default=True,
            description="Allow explicit facts via [FACT: ...] and store them persistently.",
        )
        fact_max_age_days: int = Field(
            default=90, description="Days after which a fact expires (0 = never)."
        )
        inject_facts_in_context: bool = Field(
            default=True, description="Always inject stored facts into system prompt."
        )
        fact_importance_boost: float = Field(
            default=1.5, description="Multiplier for fact importance score."
        )
        fact_command_prefix: str = Field(
            default="/fact", description="Command prefix for fact management."
        )
        enable_auto_fact_detection: bool = Field(
            default=False,
            description="Automatically detect potential facts from user messages (experimental).",
        )

        enable_iterative_mode: bool = Field(
            default=True, description="Allow /iterate command for multi-step tasks."
        )
        iterative_auto_continue: bool = Field(
            default=False,
            description="If True, /iterate runs all steps without waiting for user confirmation (use with caution).",
        )
        iterative_max_steps: int = Field(
            default=10, description="Maximum number of steps per iteration."
        )
        iterative_diff_format: str = Field(
            default="unified", description="Diff format: 'unified' or 'context'."
        )
        iterative_planning_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest",
            description="Model for planning (if empty, uses llm_model).",
        )
        iterative_execution_model: str = Field(
            default="ollama/yanjia/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced:latest",
            description="Model for step execution (if empty, uses same as planning).",
        )
        iterative_resume_command: str = Field(
            default="/iterate resume",
            description="Command to resume an interrupted iteration.",
        )
        natural_language_iterate: bool = Field(
            default=True,
            description="Allow natural language to start iterations (e.g., 'implement all features step by step').",
        )

        similar_message_handling: str = Field(
            default="replace",
            description="Action for consecutive similar messages: 'replace', 'summarize_diff', 'mark_obsolete', or 'none'.",
        )
        similar_message_threshold: float = Field(
            default=0.85,
            description="Similarity threshold to consider messages as duplicates.",
        )
        similar_message_check_code_only: bool = Field(
            default=True, description="Only apply to messages containing code blocks."
        )

        enable_obsolete_marking: bool = Field(
            default=True,
            description="Allow marking code blocks as obsolete with /obsolete.",
        )

        proactive_summary_threshold: float = Field(
            default=0.75,
            description="Token usage that triggers a summarization suggestion.",
        )
        proactive_summary_growth_window: int = Field(
            default=3, description="Number of recent messages to estimate token growth."
        )

        duplicate_question_threshold: float = Field(
            default=0.92,
            description="Similarity threshold for considering a question as duplicate.",
        )
        duplicate_question_lookback: int = Field(
            default=20, description="Number of previous user messages to search."
        )

        enable_command_suggestions: bool = Field(
            default=True,
            description="Inject helpful command suggestions based on current state.",
        )
        command_suggestion_cooldown_minutes: int = Field(
            default=10,
            description="Minutes to wait before showing the same suggestion again.",
        )

        enable_response_cache: bool = Field(
            default=True,
            description="Enable semantic caching of assistant responses to avoid repeated LLM calls for similar questions.",
        )
        response_cache_similarity_threshold: float = Field(
            default=0.92,
            description="Cosine similarity threshold to consider a cached response as a match.",
        )
        response_cache_ttl_hours: float = Field(
            default=24.0,
            description="Time-to-live for cached entries (0 = never expire).",
        )
        response_cache_max_entries: int = Field(
            default=100,
            description="Maximum number of cached (question, answer) pairs per project.",
        )
        response_cache_include_context_hash: bool = Field(
            default=True,
            description="Include context hash in cache key to avoid mismatches.",
        )

        selective_summarization: bool = Field(
            default=True,
            description="Apply different summarization strategies based on content type.",
        )
        error_preserve_verbatim: bool = Field(
            default=True, description="Never summarize error messages; keep them as-is."
        )
        error_max_age_hours: float = Field(
            default=48.0,
            description="Maximum age after which errors may be summarized (if error_preserve_verbatim is False).",
        )
        code_summary_level: str = Field(
            default="balanced",
            description="Summarization detail for code: 'minimal', 'balanced', 'detailed'.",
        )
        general_summary_max_tokens: int = Field(
            default=200, description="Max tokens for summarizing general conversation."
        )
        tool_call_preserve: bool = Field(
            default=True, description="Preserve tool call chains without summarization."
        )
        code_always_keep_signature: bool = Field(
            default=True,
            description="Always extract function/class signatures even when summarizing code.",
        )
        summary_fallback_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Model for selective summarization (if empty, uses summarization_model).",
        )
        summary_include_metadata: bool = Field(
            default=True, description="Include metadata in summaries."
        )

        summarize_old_messages: bool = Field(
            default=True, description="Summarize discarded message blocks."
        )
        summarization_model: str = Field(
            default="ollama/llama3.2:3b", description="Default summarization model."
        )
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"),
            description="OpenAI-compatible API base URL.",
        )
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", "dummy"), description="API key."
        )
        LLM_BASE_URL: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for internal LLM calls (without /v1).",
        )
        LLM_API_TOKEN: str = Field(
            default="",
            description="Optional API token for LLM calls (used only for non-Ollama endpoints).",
        )

        enable_code_awareness: bool = Field(
            default=True, description="Enable all code analysis features."
        )
        code_similarity_threshold: float = Field(
            default=0.85, description="Similarity threshold for detecting duplicates."
        )
        max_base_code_blocks: int = Field(
            default=3, description="Maximum base code blocks to keep in context."
        )

        project_id: str = Field(
            default="default", description="Project identifier (shared memory)."
        )

        max_proposed_changes: int = Field(
            default=5, description="Maximum proposed changes to keep."
        )
        max_committed_changes: int = Field(
            default=10, description="Maximum committed changes to keep."
        )
        prioritize_recent_code: bool = Field(
            default=True, description="Keep the newest version of similar code."
        )
        auto_detect_code_blocks: bool = Field(
            default=True, description="Detect fenced and indented code blocks."
        )
        max_cached_projects: int = Field(
            default=10, description="Maximum projects in LRU cache."
        )
        track_file_paths: bool = Field(
            default=True, description="Extract file paths from messages."
        )
        max_active_blocks: int = Field(
            default=50, description="Maximum active code blocks per conversation."
        )
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b",
            description="Regex for file paths.",
        )

        max_code_block_tokens: int = Field(
            default=20000, description="Maximum tokens for a code block (0 = no limit)."
        )
        code_block_overflow_action: str = Field(
            default="summarize",
            description="Action for oversized blocks: 'truncate', 'summarize', or 'warn'.",
        )
        code_block_summary_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Model for summarizing oversized blocks.",
        )
        code_block_truncate_keep_head: int = Field(
            default=50, description="Lines to keep from beginning when truncating."
        )
        code_block_truncate_keep_tail: int = Field(
            default=50, description="Lines to keep from end when truncating."
        )
        code_block_warn_message: str = Field(
            default="[Code block too large - truncated by system]",
            description="Replacement text for warn action.",
        )

        importance_mention_boost: float = Field(
            default=0.2, description="Additional importance per mention (0-1)."
        )
        importance_recency_half_life_hours: float = Field(
            default=2.0, description="Recency half-life for importance."
        )

        ltm_compress_after_messages: int = Field(
            default=50, description="Messages after which to compress old LTM entries."
        )
        ltm_summarization_trigger_similarity: float = Field(
            default=0.85, description="Similarity threshold for LTM compression."
        )

        enable_diff_application: bool = Field(
            default=True, description="Apply unified diffs to base code."
        )
        preserve_error_context: bool = Field(
            default=True, description="Never drop error messages."
        )
        error_retention_turns: int = Field(
            default=15, description="Turns to keep errors."
        )
        block_expiration_hours: float = Field(
            default=24.0, description="Hours after which inactive blocks expire."
        )
        proposed_change_retention_turns: int = Field(
            default=20, description="Turns to keep proposed changes."
        )
        preserve_tool_calls: bool = Field(
            default=True, description="Keep tool call chains intact."
        )

        enable_feedback_tracking: bool = Field(
            default=True, description="Record feedback about applied changes."
        )
        feedback_history_limit: int = Field(
            default=10, description="Maximum feedback entries per project."
        )
        inject_feedback_context: bool = Field(
            default=True, description="Inject recent feedback into system prompt."
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
            default="ollama/llama3.2:3b",
            description="Model for dependency extraction (fallback for non-Python).",
        )
        dependency_refresh_on_update: bool = Field(
            default=True, description="Re-extract dependencies on update."
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
            default="ollama/llama3.2:3b",
            description="Model for inactive code summaries.",
        )

        llm_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Preferred model (e.g., 'ollama/llama3.2:3b'). Falls back to summarization_model.",
        )

        enable_forget_command: bool = Field(
            default=True, description="Allow /forget commands."
        )
        enable_natural_language_forget: bool = Field(
            default=True, description="Interpret natural language forget."
        )
        natural_language_forget_model: str = Field(
            default="ollama/Schematron:3B",
            description="Model for forget intent parsing (very lightweight).",
        )

        ltm_store_only_code_sessions: bool = Field(
            default=True,
            description="Only store messages in LTM when in a code session.",
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
            "iterative_state": None,
            "facts": [],
            "last_compression_timestamp": 0,
            "last_suggestion_timestamp": 0,
            "response_cache": [],
        }
        self.code_pattern = re.compile(self.valves.code_block_pattern, re.DOTALL)
        self.diff_pattern = re.compile(self.valves.diff_pattern)
        self.commit_pattern = re.compile(self.valves.commit_pattern, re.IGNORECASE)

        if HAS_TIKTOKEN and self.valves.use_tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._log_debug("Tiktoken initialized")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")

        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._init_long_term_memory()
        else:
            logger.warning("Long‑term memory or code awareness disabled")

        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._load_reranker()

        if self.valves.enable_facts:
            self._log_debug("Fact storage enabled.")

        print("[CodeAware] Filter loaded (v5.23.2)")

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
    # Database
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
        self._log_debug(f"State DB initialized at {db_path}")

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
        if "last_compression_timestamp" not in data:
            data["last_compression_timestamp"] = 0
        if "facts" not in data:
            data["facts"] = []
        if "last_suggestion_timestamp" not in data:
            data["last_suggestion_timestamp"] = 0
        if "response_cache" not in data:
            data["response_cache"] = []
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
            "facts": data.get("facts", []),
            "iterative_state": data.get("iterative_state"),
            "message_count": data.get("message_count", 0),
            "last_compression_timestamp": data.get("last_compression_timestamp", 0),
            "response_cache": data.get("response_cache", []),
            "last_suggestion_timestamp": data.get("last_suggestion_timestamp", 0),
        }

    def _save_state_to_db(self, project_id: str, state: Dict):
        serializable = {
            "active_blocks": {k: v.dict() for k, v in state["active_blocks"].items()},
            "recent_changes": [b.dict() for b in state["recent_changes"]],
            "committed_changes": [b.dict() for b in state["committed_changes"]],
            "feedback_history": [fb.dict() for fb in state["feedback_history"]],
            "facts": state.get("facts", []),
            "iterative_state": state.get("iterative_state"),
            "message_count": state["message_count"],
            "last_compression_timestamp": state.get("last_compression_timestamp", 0),
            "response_cache": state.get("response_cache", []),
            "last_suggestion_timestamp": state.get("last_suggestion_timestamp", 0),
        }
        self._db_conn.execute(
            "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(serializable), time.time()),
        )
        self._db_conn.commit()

    # --------------------------------------------------------------------------
    # Debug logging
    # --------------------------------------------------------------------------
    def _log_debug(self, msg: str):
        if self.valves.debug:
            print(f"[CodeAware] {msg}")
            logger.info(msg)

    # --------------------------------------------------------------------------
    # LTM initialization
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
            now = time.time()
            expired = self.memory_collection.get(where={"expires_at": {"$lt": now}})
            if expired and expired["ids"]:
                self.memory_collection.delete(ids=expired["ids"])
                self._log_debug(f"Purged {len(expired['ids'])} expired memories")
        except Exception as e:
            logger.warning(f"Purge failed: {e}")

    # --------------------------------------------------------------------------
    # Fact management
    # --------------------------------------------------------------------------
    def _extract_facts_from_message(self, content: str) -> List[str]:
        pattern = r"\[FACT:\s*(.*?)\]"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches]

    async def _add_fact(self, project_id: str, fact_text: str, source: str = "user"):
        state = self._get_state(project_id)
        if not state:
            return
        expires_at = None
        if self.valves.fact_max_age_days > 0:
            expires_at = time.time() + (self.valves.fact_max_age_days * 86400)
        new_fact = {
            "fact": fact_text,
            "timestamp": time.time(),
            "source": source,
            "expires_at": expires_at,
        }
        for existing in state["facts"]:
            if existing["fact"] == fact_text:
                return
        state["facts"].append(new_fact)
        if len(state["facts"]) > 100:
            state["facts"] = state["facts"][-100:]
        self._set_state(project_id, state)
        self._log_debug(f"Added fact: {fact_text[:50]}...")

    async def _remove_fact(self, project_id: str, fact_text_or_index: str):
        state = self._get_state(project_id)
        if not state:
            return
        original_len = len(state["facts"])
        if fact_text_or_index.isdigit():
            idx = int(fact_text_or_index)
            if 0 <= idx < len(state["facts"]):
                state["facts"].pop(idx)
        else:
            state["facts"] = [
                f for f in state["facts"] if f["fact"] != fact_text_or_index
            ]
        if len(state["facts"]) != original_len:
            self._set_state(project_id, state)
            self._log_debug(f"Removed fact: {fact_text_or_index}")

    def _get_facts_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["facts"]:
            return ""
        now = time.time()
        active_facts = []
        for f in state["facts"]:
            if f.get("expires_at") and f["expires_at"] < now:
                continue
            active_facts.append(f["fact"])
        if not active_facts:
            return ""
        return "## Explicitly Agreed Facts\n" + "\n".join(
            [f"- {fact}" for fact in active_facts]
        )

    # --------------------------------------------------------------------------
    # Response cache
    # --------------------------------------------------------------------------
    def _compute_context_hash(self, messages: List[dict]) -> str:
        if not self.valves.response_cache_include_context_hash:
            return ""
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        context_str = "\n".join([m.get("content", "") for m in sys_msgs])
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

    async def _find_cached_response(
        self, query: str, context_hash: str, state: Dict
    ) -> Optional[Dict]:
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return None
        cache = state.get("response_cache", [])
        if not cache:
            return None
        try:
            q_emb = self.embedder.encode(query).tolist()
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache: {e}")
            return None
        best_sim = 0.0
        best_entry = None
        now = time.time()
        ttl = self.valves.response_cache_ttl_hours * 3600
        for entry in cache:
            if ttl > 0 and (now - entry.get("timestamp", 0)) > ttl:
                continue
            if (
                self.valves.response_cache_include_context_hash
                and entry.get("context_hash", "") != context_hash
            ):
                continue
            entry_emb = entry.get("embedding")
            if not entry_emb:
                continue
            sim = self._cosine_similarity(q_emb, entry_emb)
            if (
                sim > best_sim
                and sim >= self.valves.response_cache_similarity_threshold
            ):
                best_sim = sim
                best_entry = entry
        if best_entry:
            self._log_debug(
                f"Cache hit (similarity={best_sim:.3f}) for query: {query[:50]}..."
            )
            return best_entry
        return None

    async def _store_response_in_cache(
        self, query: str, response: str, context_hash: str, state: Dict
    ):
        if not self.valves.enable_response_cache or not HAS_SENTENCE:
            return
        if not query or not response:
            return
        try:
            embedding = self.embedder.encode(query).tolist()
        except Exception as e:
            self._log_debug(f"Failed to encode query for cache storage: {e}")
            return
        cache = state.get("response_cache", [])
        new_cache = [e for e in cache if e.get("query") != query]
        new_entry = {
            "query": query,
            "response": response,
            "embedding": embedding,
            "timestamp": time.time(),
            "context_hash": context_hash,
        }
        new_cache.append(new_entry)
        if len(new_cache) > self.valves.response_cache_max_entries:
            new_cache.sort(key=lambda x: x.get("timestamp", 0))
            new_cache = new_cache[-self.valves.response_cache_max_entries :]
        state["response_cache"] = new_cache
        self._log_debug(f"Stored response in cache for query: {query[:50]}...")

    # --------------------------------------------------------------------------
    # AST-based dependency extraction (Python)
    # --------------------------------------------------------------------------
    def _extract_dependencies_ast(self, code: str) -> Tuple[List[str], List[str]]:
        imports = []
        calls = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(
                            f"{module}.{alias.name}" if module else alias.name
                        )
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        except SyntaxError:
            pass
        return list(set(imports)), list(set(calls))

    # --------------------------------------------------------------------------
    # Multi-language dependency extraction (regex-based)
    # --------------------------------------------------------------------------
    def _extract_dependencies_regex(self, code: str, language: str) -> List[str]:
        deps = set()
        if language in ("javascript", "typescript", "js", "ts", "jsx", "tsx"):
            for m in re.finditer(r"""from\s+['"]([^'"]+)['"]""", code):
                deps.add(m.group(1))
            for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", code):
                deps.add(m.group(1))
            for m in re.finditer(r"""import\s+['"]([^'"]+)['"]""", code):
                deps.add(m.group(1))
        elif language == "go":
            for m in re.finditer(r'import\s+"([^"]+)"', code):
                deps.add(m.group(1))
        elif language == "rust":
            for m in re.finditer(r"""use\s+([\w:]+)""", code):
                deps.add(m.group(1))
        elif language == "java":
            for m in re.finditer(r"""import\s+([\w.]+)""", code):
                deps.add(m.group(1))
        elif language in ("c", "cpp", "c++"):
            for m in re.finditer(r"""#include\s+[<"]([^>"]+)[>"]""", code):
                deps.add(m.group(1))
        return list(deps)

    async def _extract_dependencies_hybrid(
        self, code: str, file_path: Optional[str] = None
    ) -> List[str]:
        if not self.valves.enable_dependency_tracking:
            return []
        lang = "unknown"
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".jsx": "javascript",
                ".tsx": "typescript",
                ".go": "go",
                ".rs": "rust",
                ".java": "java",
                ".c": "c",
                ".cpp": "cpp",
                ".h": "c",
                ".hpp": "cpp",
            }
            lang = lang_map.get(ext, "unknown")
        else:
            if re.search(r"\bdef\s+\w+\s*\(", code) and re.search(
                r"\bimport\s+\w+", code
            ):
                lang = "python"
            elif re.search(r"\b(function|const|let|var|=>)\b", code):
                lang = "javascript"
        if lang == "python":
            imports, calls = self._extract_dependencies_ast(code)
            return list(set(imports + calls))
        if lang != "unknown":
            deps = self._extract_dependencies_regex(code, lang)
            if deps:
                return deps
        model = (
            self.valves.dependency_extraction_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analyze the following code and extract dependencies...\n```\n{code[:1500]}\n```"""
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
        if action == "truncate":
            lines = code.splitlines()
            head = self.valves.code_block_truncate_keep_head
            tail = self.valves.code_block_truncate_keep_tail
            if len(lines) <= head + tail:
                return code
            return "\n".join(
                lines[:head]
                + [f"... [{len(lines) - head - tail} lines truncated] ..."]
                + lines[-tail:]
            )
        elif action == "summarize":
            model = (
                self.valves.code_block_summary_model
                or self.valves.llm_model
                or self.valves.summarization_model
            )
            summary = await self._call_llm(
                prompt=f"Summarize the following {language} code block.\n```{language}\n{code[:8000]}\n```",
                system_prompt="You are a code summarization assistant.",
                model_override=model,
                max_tokens=500,
                temperature=0.2,
            )
            return (
                f"[Automatic summary of a {estimated} token code block]\n{summary}"
                if summary
                else f"[Code block too large, could not summarize] Original size: {estimated} tokens."
            )
        elif action == "warn":
            return self.valves.code_block_warn_message
        return code

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
        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        api_token = self.valves.LLM_API_TOKEN.strip() or None
        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url
        models_to_try = []
        if model_override:
            models_to_try.append(model_override)
        if self.valves.llm_model:
            models_to_try.append(self.valves.llm_model)
        models_to_try.append(self.valves.summarization_model)
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]
        for model in models_to_try:
            if is_ollama and model.startswith("ollama/"):
                model_name = model.split("/", 1)[1]
            else:
                model_name = model
            try:
                async with aiohttp.ClientSession() as session:
                    if is_ollama:
                        url = f"{base_url}/api/generate"
                        payload = {
                            "model": model_name,
                            "prompt": prompt,
                            "system": system_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens,
                            },
                        }
                        headers = {"Content-Type": "application/json"}
                    else:
                        url = f"{base_url}/v1/chat/completions"
                        headers = {"Content-Type": "application/json"}
                        if api_token:
                            headers["Authorization"] = f"Bearer {api_token}"
                        elif self.valves.openai_api_key:
                            headers["Authorization"] = (
                                f"Bearer {self.valves.openai_api_key}"
                            )
                        payload = {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                    async with session.post(
                        url, json=payload, headers=headers, timeout=30
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if is_ollama:
                                content = data.get("response", "")
                                if not content.strip():
                                    continue
                                return content.strip()
                            else:
                                return data["choices"][0]["message"]["content"].strip()
                        else:
                            self._log_debug(
                                f"LLM call failed with model {model_name}, status {resp.status}"
                            )
            except Exception as e:
                self._log_debug(f"LLM model {model_name} error: {e}")
                continue
        logger.warning(f"All LLM models failed for prompt: {prompt[:100]}...")
        return None

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
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
            name, params = func_match.group(1), func_match.group(2).strip()
            doc_match = re.search(
                r'^\s*"""(.*?)"""', code[func_match.end() :], re.DOTALL
            ) or re.search(r"^\s*'''(.*?)'''", code[func_match.end() :], re.DOTALL)
            docstring = doc_match.group(1).strip()[:100] if doc_match else ""
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
            ) or re.search(r"^\s*'''(.*?)'''", code[class_match.end() :], re.DOTALL)
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

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

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
            old_lines, new_lines = [], []
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
            if block.pinned or block.obsolete:
                continue
            age = now - block.last_mentioned
            if (
                block.content_type == ContentType.ERROR
                and self.valves.error_retention_turns > 0
            ):
                if age > max(
                    self.valves.error_retention_turns * 300, expiration_seconds
                ):
                    to_remove.append(h)
            elif (
                block.content_type == ContentType.PROPOSED_CHANGE
                and self.valves.proposed_change_retention_turns > 0
            ):
                if age > max(
                    self.valves.proposed_change_retention_turns * 300,
                    expiration_seconds,
                ):
                    to_remove.append(h)
        for h in to_remove:
            del state["active_blocks"][h]
        if to_remove:
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Inactive code summarization
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
            if block.pinned or block.obsolete:
                continue
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
        self._set_state(project_id, state)

    async def _summarize_code_block(self, block: CodeBlock) -> Optional[str]:
        if not self.valves.summarize_inactive_code or not HAS_AIOHTTP:
            return None
        sig = self._extract_signature(block.content)
        prompt = (
            f"""The code block has signature: {sig}
Provide a very brief description of what this code does.
Code:
```{block.content[:1000]}```"""
            if sig
            else f"""Summarize the following code block.
```{block.content[:1500]}```"""
        )
        return await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code summarization assistant.",
            model_override=self.valves.inactive_code_summary_model,
            max_tokens=200,
            temperature=0.2,
        )

    # --------------------------------------------------------------------------
    # Hierarchical compression
    # --------------------------------------------------------------------------
    async def _hierarchical_compress(self, project_id: str, state: Dict):
        if not self.valves.hierarchical_compression_enabled:
            return
        last_ts = state.get("last_compression_timestamp", 0)
        if time.time() - last_ts < 3600:
            return
        try:
            now = time.time()
            where_filter = {
                "$and": [
                    {"project_id": {"$eq": project_id}},
                    {"is_hierarchical_summary": {"$ne": True}},
                    {"timestamp": {"$lt": now}},
                ]
            }
            results = self.memory_collection.get(
                where=where_filter,
                include=["documents", "metadatas", "ids"],
                limit=self.valves.hierarchical_compression_interval_messages * 2,
            )
        except Exception as e:
            self._log_debug(
                f"Failed to fetch messages for hierarchical compression: {e}"
            )
            return
        if (
            not results
            or not results["ids"]
            or len(results["ids"])
            < self.valves.hierarchical_compression_interval_messages
        ):
            return
        pairs = sorted(
            zip(results["ids"], results["documents"], results["metadatas"]),
            key=lambda x: x[2].get("timestamp", 0),
        )
        to_compress = pairs[: self.valves.hierarchical_compression_interval_messages]
        texts = "\n---\n".join([doc for _, doc, _ in to_compress])
        model = (
            self.valves.hierarchical_summary_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        summary = await self._call_llm(
            prompt=f"Summarise the following conversation segment...\n{texts[:4000]}",
            system_prompt="You are a code-aware assistant.",
            model_override=model,
            max_tokens=self.valves.hierarchical_summary_max_tokens,
            temperature=0.2,
        )
        if not summary:
            return
        summary_id = f"{project_id}_hierarchical_{int(time.time())}"
        summary_embedding = self.embedder.encode(summary).tolist()
        self.memory_collection.upsert(
            ids=[summary_id],
            embeddings=[summary_embedding],
            metadatas=[
                {
                    "role": "assistant",
                    "project_id": project_id,
                    "timestamp": time.time(),
                    "is_hierarchical_summary": True,
                    "summary_level": 1,
                }
            ],
            documents=[f"[Hierarchical summary]\n{summary}"],
        )
        state["last_compression_timestamp"] = time.time()
        self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Duplicate removal
    # --------------------------------------------------------------------------
    def _remove_duplicate_blocks(self, state: Dict):
        if not self.valves.auto_remove_duplicate_blocks:
            return
        blocks = list(state["active_blocks"].values())
        to_remove = set()
        for i, block in enumerate(blocks):
            if block.hash in to_remove or block.pinned or block.obsolete:
                continue
            for j, other in enumerate(blocks[i + 1 :], start=i + 1):
                if other.hash in to_remove or other.pinned or other.obsolete:
                    continue
                sim = self._calculate_code_similarity(block.content, other.content)
                if sim >= self.valves.code_similarity_threshold:
                    age_diff = abs(block.timestamp - other.timestamp) / 3600
                    if age_diff > self.valves.max_duplicate_age_hours:
                        if (
                            block.timestamp < other.timestamp
                            and block.importance_score < 5.0
                        ):
                            to_remove.add(block.hash)
                        elif (
                            other.timestamp < block.timestamp
                            and other.importance_score < 5.0
                        ):
                            to_remove.add(other.hash)
                        continue
                    if block.importance_score >= other.importance_score:
                        to_remove.add(other.hash)
                    else:
                        to_remove.add(block.hash)
        for h in to_remove:
            if h in state["active_blocks"]:
                del state["active_blocks"][h]
        state["recent_changes"] = [
            b for b in state["recent_changes"] if b.hash not in to_remove
        ]
        state["committed_changes"] = [
            b for b in state["committed_changes"] if b.hash not in to_remove
        ]

    # --------------------------------------------------------------------------
    # Consecutive message dedup
    # --------------------------------------------------------------------------
    async def _summarize_diff_between_messages(self, msg1: dict, msg2: dict) -> str:
        content1 = msg1.get("content", "")
        content2 = msg2.get("content", "")
        prompt = f"""The user sent two consecutive very similar messages...
FIRST: {content1[:1500]}
SECOND: {content2[:1500]}
Provide a concise summary of what changed."""
        diff_summary = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a code-aware assistant.",
            max_tokens=300,
            temperature=0.2,
        )
        return (
            f"[Summary of changes]\n{diff_summary}\n\n[Final version]\n{content2}"
            if diff_summary
            else f"[Updated version]\n{content2}"
        )

    async def _deduplicate_consecutive_messages(
        self, history_msgs: List[dict]
    ) -> List[dict]:
        if self.valves.similar_message_handling == "none" or len(history_msgs) < 2:
            return history_msgs
        new_history = []
        i = 0
        while i < len(history_msgs):
            current = history_msgs[i]
            j = i + 1
            while j < len(history_msgs) and history_msgs[j].get("role") == current.get(
                "role"
            ):
                j += 1
            if j - i > 1:
                first, last = history_msgs[i], history_msgs[j - 1]
                if self.valves.similar_message_check_code_only:
                    contains_code = any(
                        "```" in history_msgs[k].get("content", "") for k in range(i, j)
                    )
                else:
                    contains_code = True
                if (
                    contains_code
                    and self._calculate_code_similarity(
                        first.get("content", ""), last.get("content", "")
                    )
                    >= self.valves.similar_message_threshold
                ):
                    action = self.valves.similar_message_handling
                    if action == "replace":
                        new_history.append(last)
                        i = j
                        continue
                    elif action == "summarize_diff" and j - i == 2:
                        new_history.append(
                            {
                                "role": first.get("role"),
                                "content": await self._summarize_diff_between_messages(
                                    first, last
                                ),
                            }
                        )
                        i = j
                        continue
            new_history.append(current)
            i += 1
        return new_history

    # --------------------------------------------------------------------------
    # HYBRID intent parsers (improved heuristics)
    # --------------------------------------------------------------------------
    async def _parse_forget_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_natural_language_forget:
            return None
        content = user_message.strip().lower()
        # Heuristics
        if content.startswith("/forget"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "forget_all"}
            elif target in ("last", "último"):
                return {"action": "forget_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "forget_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "forget_block", "hash": target}
            else:
                return {"action": "forget_last"}
        if re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\s+(todo|all|todo el contexto|el contexto)\b",
            content,
        ):
            return {"action": "forget_all"}
        if re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\s+(el|la|lo)\s+(último|last|ultimo|último bloque|last block)\b",
            content,
        ):
            return {"action": "forget_last"}
        if re.search(
            r'\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\s+(el archivo|el fichero|el file)\s+["\']?([^"\'\s]+)["\']?',
            content,
        ):
            file_match = re.search(
                r'["\']?([^"\'\s]+)["\']?',
                (
                    content.split("archivo")[-1]
                    if "archivo" in content
                    else content.split("file")[-1]
                ),
            )
            if file_match:
                return {"action": "forget_file", "file": file_match.group(1)}
        if not re.search(
            r"\b(olvida|borra|elimina|forget|remove|delete|olvidar|borrar|eliminar|quita|quitar|saca|sacá)\b",
            content,
        ):
            return None
        # LLM fallback
        self._log_debug("Calling LLM for forget intent")
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
                return data
        except:
            pass
        return None

    async def _parse_remember_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_natural_language_forget:
            return None
        content = user_message.strip().lower()
        if content.startswith("/remember"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "pin_all"}
            elif target in ("last", "último"):
                return {"action": "pin_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "pin_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "pin_block", "hash": target}
            else:
                return {"action": "pin_last"}
        if re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\s+(todo|all|todo el contexto|el contexto)\b",
            content,
        ):
            return {"action": "pin_all"}
        if re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\s+(el|la|lo)\s+(último|last|ultimo|último bloque|last block)\b",
            content,
        ):
            return {"action": "pin_last"}
        if re.search(
            r'\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\s+(el archivo|el fichero|el file)\s+["\']?([^"\'\s]+)["\']?',
            content,
        ):
            file_match = re.search(
                r'["\']?([^"\'\s]+)["\']?',
                (
                    content.split("archivo")[-1]
                    if "archivo" in content
                    else content.split("file")[-1]
                ),
            )
            if file_match:
                return {"action": "pin_file", "file": file_match.group(1)}
        if not re.search(
            r"\b(recuerda|pinea|guarda|remember|pin|save|recordar|pinear|guardar)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for remember intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpret pin/remember intent. Possible actions: pin_last, pin_n (with n), pin_file (with file), pin_block (with description), pin_all, unpin_last, unpin_file, unpin_all, etc.
User: "{user_message}"
Output JSON with action and parameters. If no pinning intent: {{"action": "none"}}
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
                return data
        except:
            pass
        return None

    async def _parse_obsolete_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_obsolete_marking:
            return None
        content = user_message.strip().lower()
        if content.startswith("/obsolete"):
            parts = content.split(maxsplit=1)
            target = parts[1] if len(parts) > 1 else ""
            if target in ("all", "todo"):
                return {"action": "obsolete_all"}
            elif target in ("last", "último"):
                return {"action": "obsolete_last"}
            elif target:
                if re.search(r"\.[a-zA-Z]+$", target):
                    return {"action": "obsolete_file", "file": target}
                elif re.match(r"^[a-f0-9]{8,}$", target):
                    return {"action": "obsolete_block", "hash": target}
            else:
                return {"action": "obsolete_last"}
        if re.search(
            r"\b(obsoleto|obsoleta|descarta|marca\s+como\s+obsoleto|obsolete|discard|esto ya no sirve|ya no es relevante)\b",
            content,
        ):
            return {"action": "obsolete_last"}
        if re.search(
            r"\b(revive|revivir|recupera|restaura)\s+(el|la|lo)\s+(último|last|ultimo)\b",
            content,
        ):
            return {"action": "revive_last"}
        if not re.search(
            r"\b(obsoleto|obsoleta|descarta|obsolete|discard|revive|revivir|restaura|recupera)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for obsolete intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = (
            f"""Interpret obsolete intent... (same as before) ... Output only JSON."""
        )
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
                return data
        except:
            pass
        return None

    async def _parse_feedback_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_feedback_tracking:
            return None
        content = user_message.strip().lower()
        if re.search(r"\b(feedback|retroalimentación)\b", content):
            if "success" in content or "worked" in content:
                return {"action": "feedback", "outcome": "success", "comment": content}
            if "fail" in content or "error" in content:
                return {"action": "feedback", "outcome": "failure", "comment": content}
        if re.search(
            r"\b(funcionó perfecto|solucionado|resuelto|solved|fixed|correcto|excelente)\b",
            content,
        ):
            return {"action": "feedback", "outcome": "success", "comment": content}
        if re.search(
            r"\b(no funcionó|no funciono|sigue roto|still broken|error|falló|fallo|no solucionado|incorrecto)\b",
            content,
        ):
            return {"action": "feedback", "outcome": "failure", "comment": content}
        if not re.search(
            r"\b(worked|did not work|failed|success|correcto|incorrecto|error|fallo|funcionó|funciono)\b",
            content,
        ):
            return None
        self._log_debug("Calling LLM for feedback intent")
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""You are a feedback interpreter... Output JSON."""
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
                return data
        except:
            pass
        return None

    # --------------------------------------------------------------------------
    # Contradiction detection (with pre-filter)
    # --------------------------------------------------------------------------
    async def _detect_contradictions(
        self, conversation_messages: List[dict]
    ) -> Optional[str]:
        if (
            not self.valves.enable_contradiction_detection
            or len(conversation_messages) < 4
        ):
            return None
        recent = conversation_messages[-10:]
        has_potential = False
        for i, msg1 in enumerate(recent):
            for msg2 in recent[i + 1 :]:
                if msg1.get("role") == msg2.get("role") == "user":
                    sim = self._calculate_code_similarity(
                        msg1.get("content", ""), msg2.get("content", "")
                    )
                    if sim > 0.6 and (
                        "no " in msg2.get("content", "").lower()
                        or "error" in msg2.get("content", "").lower()
                    ):
                        has_potential = True
                        break
            if has_potential:
                break
        if not has_potential:
            return None
        model = (
            self.valves.contradiction_detection_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        conv_text = "\n".join(
            [f"{m.get('role')}: {m.get('content','')[:500]}" for m in recent]
        )
        prompt = f"""Analyze contradictions... {conv_text} Output JSON."""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a contradiction detection assistant.",
            model_override=model,
            max_tokens=300,
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
            if data.get("contradiction"):
                return f"⚠️ **Contradiction detected**: {data.get('explanation','')}\n\nSuggestion: {data.get('suggestion','')}"
        except:
            pass
        return None

    # --------------------------------------------------------------------------
    # HYBRID session detection
    # --------------------------------------------------------------------------
    def _has_code_indicators(self, content: str) -> bool:
        if "```" in content:
            return True
        if re.search(
            r"\b(def |class |import |from |function |const |let |var |#include |package |fn |func )",
            content,
        ):
            return True
        if re.search(
            r"\b[\w\-/]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp)\b", content
        ):
            return True
        return False

    async def _classify_session(self, messages: List[dict], project_id: str) -> bool:
        state = self._get_state(project_id)
        if state and state.get("active_blocks"):
            self._log_debug("Session classified as code (existing blocks)")
            return True
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            if self._has_code_indicators(msg.get("content", "")):
                self._log_debug("Session classified as code (heuristic)")
                return True
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            self._log_debug("Session classified as non-code (no user message)")
            return False
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"Is the following user message about programming/code? Answer only 'yes' or 'no'.\n\nMessage: {last_user_msg.get('content','')[:500]}"
        self._log_debug("Classifying session with LLM...")
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="You are a classifier. Answer only 'yes' or 'no'.",
            model_override=model,
            max_tokens=5,
            temperature=0.0,
        )
        result = response and response.strip().lower().startswith("yes")
        self._log_debug(
            f"LLM classification result: {'code' if result else 'non-code'}"
        )
        return result

    # --------------------------------------------------------------------------
    # Inlet
    # --------------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("inlet called")
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        if not messages:
            return body
        state = self._get_state(project_id)
        is_code_session = await self._classify_session(messages, project_id)
        self._log_debug(f"Session: {'code' if is_code_session else 'non-code'}")

        # 1. Forget (only if code session, explicit command, or blocks exist)
        if (
            self.valves.enable_forget_command
            or self.valves.enable_natural_language_forget
        ) and (
            is_code_session
            or (messages and messages[-1].get("content", "").startswith("/"))
        ):
            self._log_debug("Processing forget commands")
            new_messages, handled = await self._handle_forget_command(
                messages, project_id, __user__
            )
            if handled:
                body["messages"] = new_messages
                return body

        # 2. Remember (similar gate)
        if self.valves.enable_natural_language_forget:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg and (
                is_code_session or last_user_msg.get("content", "").startswith("/")
            ):
                self._log_debug("Processing remember commands")
                remember_intent = await self._parse_remember_intent(
                    last_user_msg.get("content", "")
                )
                if (
                    remember_intent
                    and remember_intent.get("action")
                    and remember_intent["action"] != "none"
                ):
                    confirmation = await self._execute_remember_intent(
                        project_id, remember_intent
                    )
                    body["messages"] = messages + [
                        {"role": "assistant", "content": confirmation}
                    ]
                    return body

        # ... (steps 3-11 unchanged but similarly gated)

        # 9. Update active code only for code sessions
        if self.valves.enable_code_awareness and is_code_session:
            self._log_debug("Updating active code blocks")
            for msg in messages[-5:]:
                await self._update_active_code(msg, project_id)

        # 12-15: Inject enriched context only for code sessions
        if is_code_session:
            self._log_debug("Injecting enriched context for code session")
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

            if self.valves.enable_facts and self.valves.inject_facts_in_context:
                facts_ctx = self._get_facts_context(project_id)
                if facts_ctx:
                    sys_msgs = [m for m in messages if m.get("role") == "system"]
                    if sys_msgs:
                        sys_msgs[0]["content"] = (
                            facts_ctx + "\n\n" + sys_msgs[0]["content"]
                        )
                    else:
                        messages.insert(0, {"role": "system", "content": facts_ctx})
                    body["messages"] = messages

            if self.valves.enable_confidence_scoring:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] += self.valves.confidence_prompt
                else:
                    messages.insert(
                        0, {"role": "system", "content": self.valves.confidence_prompt}
                    )
                body["messages"] = messages

            if (
                self.valves.enable_feedback_tracking
                and self.valves.inject_feedback_context
            ):
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

        # Final trim (unchanged)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        # ... trimming logic ...
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
        state = self._get_state(project_id)
        is_code_session = await self._classify_session(messages, project_id)
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                if is_code_session:
                    await self._update_active_code(msg, project_id)
                    if not self.valves.ltm_store_only_code_sessions or is_code_session:
                        await self._store_message_in_memory(msg, project_id)
                else:
                    if not self.valves.ltm_store_only_code_sessions:
                        await self._store_message_in_memory(msg, project_id)
        if self.valves.enable_response_cache and HAS_SENTENCE and len(messages) >= 2:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"), None
            )
            if last_user and last_assistant:
                context_hash = self._compute_context_hash(messages[:-1])
                await self._store_response_in_cache(
                    last_user.get("content", ""),
                    last_assistant.get("content", ""),
                    context_hash,
                    state,
                )
        self._purge_expired_memories()
        return body
