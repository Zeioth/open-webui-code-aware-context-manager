# region ── Imports ────────────────────────────────────────────────────────────
import json
import logging
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple
import anyio
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
# endregion


class Filter:
    # region ── Configuration (Valves) ────────────────────────────────────────
    class Valves(BaseModel):
        # -- Expert definitions --
        experts_json: str = Field(
            default="""[
    {
        "id": "experto-en-neovim",
        "name": "Neovim",
        "keywords": ["neovim", "nvim", "vim", "lsp", "treesitter", "buffer", "init.lua", "keymap"],
        "description": "Configuration, plugins, and usage of Neovim editor",
        "examples": ["how to configure LSP in nvim", "best neovim plugins 2025", "treesitter setup init.lua"],
        "knowledge_base": "the official Neovim documentation and community plugins",
        "collection_name": "neovim_docs"
    },
    {
        "id": "experto-en-arch-linux",
        "name": "Arch Linux",
        "keywords": ["arch", "archlinux", "pacman", "yay", "aur", "systemd", "grub", "hyprland", "wayland"],
        "description": "Installation, maintenance, and troubleshooting of Arch Linux",
        "examples": ["how to install yay", "hyprland config", "pacman update error"],
        "knowledge_base": "the official Arch Linux wiki",
        "collection_name": "archwiki"
    }
]""",
            description="JSON with expert definitions. Each must have 'id', 'name', 'keywords' (list). Optional: 'description', 'examples', 'knowledge_base', 'collection_name'.",
        )
        default_model: str = Field(
            default="generalista",
            description="ID of the generalist model (used when no expert detected).",
        )
        change_threshold: int = Field(
            default=2,
            description="Minimum number of shared words with previous context to NOT switch experts.",
        )
        notify_change: bool = Field(
            default=True,
            description="Show notifications in the UI when the expert changes.",
        )
        notification_template: str = Field(
            default="🎯 Using expert: {expert}",
            description="Notification message template. Use {expert} for the expert name.",
        )
        # -- LLM call settings --
        LLM_BASE_URL: str = Field(
            default="http://host.docker.internal:11434/",
            description="Base URL for LLM API (Ollama or OpenAI compatible).",
        )
        LLM_API_TOKEN: str = Field(
            default="",
            description="API token (leave empty if not required).",
        )
        # -- Classifier LLM settings --
        classifier_model: str = Field(
            default="ollama/llama3.2:3b",
            description="Small model used for LLM-based classification and query rewriting.",
        )
        classifier_temperature: float = Field(
            default=0.0,
            description="Temperature for the classifier LLM.",
        )
        classifier_timeout: int = Field(
            default=10,
            description="Timeout (seconds) for classifier LLM calls.",
        )
        # -- Query rewriting for better RAG retrieval --
        enable_query_rewriting: bool = Field(
            default=True,
            description="Rewrite user query to a more precise technical phrase before retrieval.",
        )
        # -- Native RAG injection (via Open WebUI built-in pipeline) --
        enable_rag_injection: bool = Field(
            default=True,
            description="Guide the built-in RAG to use the expert's knowledge collection and focus the answer.",
        )
        # -- Performance: cache sizes and TTLs --
        rewrite_cache_max_size: int = Field(
            default=500,
            description="Maximum number of cached rewritten queries (keyed by original + expert).",
        )
        string_cache_max_size: int = Field(
            default=1000,
            description="Maximum number of cached expert classification results. Set to 0 for unlimited.",
        )
        string_cache_ttl: int = Field(
            default=1800,  # 30 minutes
            description="Time-to-live (seconds) for classification cache entries. Set to 0 for no expiry.",
        )
        rewrite_cache_ttl: int = Field(
            default=3600,  # 1 hour
            description="Time-to-live (seconds) for rewritten query cache entries. Set to 0 for no expiry.",
        )
        # -- Debug mode --
        DEBUG: bool = Field(
            default=False,
            description="Enable detailed logging of routing decisions.",
        )

    # endregion

    # region ── Cache helper class (thread‑safe with TTL) ─────────────────────
    class _Cache:
        """Simple async‑safe cache with max size and TTL.
        - max_size = 0  -> unlimited size.
        - ttl = 0       -> no expiry.
        """

        def __init__(self, max_size: int = 1000, ttl: int = 1800):
            self.max_size = max_size
            self.ttl = ttl
            self._store: Dict[str, Tuple[object, float]] = {}
            self._lock = asyncio.Lock()

        async def get(self, key: str) -> Optional[object]:
            async with self._lock:
                entry = self._store.get(key)
                if entry is None:
                    return None
                value, timestamp = entry
                # Expire only if ttl > 0
                if self.ttl > 0 and time.time() - timestamp > self.ttl:
                    del self._store[key]
                    return None
                # Move to end to simulate LRU ordering (optional)
                self._store.move_to_end(key)
                return value

        async def set(self, key: str, value: object):
            async with self._lock:
                # Evict oldest if max_size > 0 and we are at capacity (key not already present)
                if (
                    self.max_size > 0
                    and len(self._store) >= self.max_size
                    and key not in self._store
                ):
                    oldest = next(iter(self._store))
                    del self._store[oldest]
                self._store[key] = (value, time.time())
                # Maintain insertion order for eviction policy
                self._store.move_to_end(key)

        async def clear(self):
            async with self._lock:
                self._store.clear()

    # endregion

    # region ── Initialization ─────────────────────────────────────────────────
    def __init__(self):
        self.valves = self.Valves()
        self._experts: List[dict] = []
        self._experts_json_hash: Optional[str] = (
            None  # hash of the JSON that produced _experts
        )
        self._load_experts()  # initial load

        # Initialize caches (they will be synced in inlet if valves change)
        self._string_cache = self._Cache(
            max_size=self.valves.string_cache_max_size, ttl=self.valves.string_cache_ttl
        )
        self._rewrite_cache = self._Cache(
            max_size=self.valves.rewrite_cache_max_size,
            ttl=self.valves.rewrite_cache_ttl,
        )

    def _load_experts(self):
        """Parse and validate expert JSON configuration. Only reload if JSON changed."""
        try:
            new_json = self.valves.experts_json
            new_hash = hashlib.md5(new_json.encode()).hexdigest()
            if self._experts_json_hash == new_hash:
                # No change, keep existing experts
                if self.valves.DEBUG:
                    logger.info(
                        "[Router] Expert configuration unchanged, keeping previous."
                    )
                return

            # Attempt to parse new JSON
            parsed = json.loads(new_json)
            for exp in parsed:
                required = {"id", "name", "keywords"}
                if not required.issubset(exp.keys()):
                    raise ValueError(f"Expert missing required fields: {exp}")
            # Valid: update
            self._experts = parsed
            self._experts_json_hash = new_hash
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] Loaded {len(self._experts)} experts. JSON hash updated."
                )
        except Exception as e:
            logger.error(
                f"[Router] Error parsing experts_json: {e}. Keeping previous expert list."
            )
            # _experts remains unchanged (previous valid list, or empty if never loaded successfully)

    async def _sync_cache_config(self):
        """Ensure cache sizes/ttl match current valves (called in inlet)."""
        # String cache
        if (
            self._string_cache.max_size != self.valves.string_cache_max_size
            or self._string_cache.ttl != self.valves.string_cache_ttl
        ):
            self._string_cache = self._Cache(
                max_size=self.valves.string_cache_max_size,
                ttl=self.valves.string_cache_ttl,
            )
        # Rewrite cache
        if (
            self._rewrite_cache.max_size != self.valves.rewrite_cache_max_size
            or self._rewrite_cache.ttl != self.valves.rewrite_cache_ttl
        ):
            self._rewrite_cache = self._Cache(
                max_size=self.valves.rewrite_cache_max_size,
                ttl=self.valves.rewrite_cache_ttl,
            )

    # endregion

    # region ── LLM call (reused from shared code) ────────────────────────────
    async def _call_llm(
        self,
        prompt: str,
        system: str = "",
        provider: str = "",
        temperature: float = 0.3,
        max_tokens: int = 700,
        timeout: int = 120,
    ) -> str:
        """Send a prompt to the configured LLM provider (Ollama or OpenAI‑compatible)."""
        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        api_token = (
            self.valves.LLM_API_TOKEN.strip()
            if self.valves.LLM_API_TOKEN and self.valves.LLM_API_TOKEN.strip()
            else None
        )
        model_str = provider or self.valves.classifier_model
        if "/" in model_str:
            model_str = model_str.split("/", 1)[1]
        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

        def _blocking():
            if is_ollama:
                url = f"{base_url}/api/generate"
                payload = {
                    "model": model_str,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Ollama error {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                if "error" in data:
                    logger.error(f"Ollama model error: {data['error']}")
                    return ""
                return data.get("response", "")
            else:
                headers = {"Content-Type": "application/json"}
                if api_token:
                    headers["Authorization"] = f"Bearer {api_token}"
                payload = {
                    "model": model_str,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"OpenAI error {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()["choices"][0]["message"]["content"]

        content = await anyio.to_thread.run_sync(_blocking)
        if not content or not content.strip():
            logger.warning(f"[Router] LLM returned empty content for '{model_str}'.")
            return ""
        if self.valves.DEBUG:
            logger.debug(f"[Router] LLM raw response: {content[:300]}")
        return content

    # endregion

    # region ── Expert classification (LLM + keyword fallback) ───────────────
    async def _classify_with_llm(self, user_query: str) -> Optional[str]:
        """Use the small LLM to decide which expert should handle the query."""
        lines = []
        for exp in self._experts:
            line = f"- {exp['id']}: {exp['name']}"
            if exp.get("description"):
                line += f" ({exp['description']})"
            lines.append(line)
        expert_list = "\n".join(lines)

        examples_section = ""
        for exp in self._experts:
            if exp.get("examples"):
                for ex in exp["examples"]:
                    examples_section += f"User: {ex}\nExpert: {exp['id']}\n"

        system_prompt = (
            "You are a smart router. Choose the expert that best matches the user query.\n"
            "Reply with ONLY the expert ID (one word) or 'generalista'.\n\n"
            f"Available experts:\n{expert_list}\n"
        )
        if examples_section:
            system_prompt += f"\nExamples:\n{examples_section}\n"
        system_prompt += "\nExpert ID:"

        try:
            response = await self._call_llm(
                prompt=user_query,
                system=system_prompt,
                provider=self.valves.classifier_model,
                temperature=self.valves.classifier_temperature,
                max_tokens=10,
                timeout=self.valves.classifier_timeout,
            )
            if not response:
                return None
            first_word = response.strip().split()[0].lower()
            valid_ids = {exp["id"] for exp in self._experts}
            valid_ids.add("generalista")
            if first_word in valid_ids:
                return first_word
            else:
                logger.warning(
                    f"[Router] Unexpected LLM classifier output: '{first_word}'"
                )
                return None
        except Exception as e:
            logger.error(f"[Router] LLM classifier error: {e}")
            return None

    def _keyword_fallback(self, text: str) -> Optional[str]:
        """Original keyword matching as last resort."""
        text_lower = text.lower()
        for exp in self._experts:
            if any(kw in text_lower for kw in exp.get("keywords", [])):
                return exp["id"]
        return None

    async def _classify_query(self, user_query: str) -> str:
        """Classify the query using string cache -> LLM -> keyword fallback."""
        # 1. Check cache (with TTL)
        cache_key = user_query.lower()
        cached = await self._string_cache.get(cache_key)
        if cached is not None:
            if self.valves.DEBUG:
                logger.info(f"[Router] String cache hit: {cached}")
            return cached

        # 2. LLM classification
        expert_id = await self._classify_with_llm(user_query)
        if expert_id:
            await self._string_cache.set(cache_key, expert_id)
            return expert_id

        # 3. Keyword fallback
        expert_id = self._keyword_fallback(user_query)
        if expert_id:
            await self._string_cache.set(cache_key, expert_id)
            return expert_id

        # 4. Default
        return self.valves.default_model

    # endregion

    # region ── Query rewriting (domain‑aware + rewrite cache) ───────────────
    async def _rewrite_query(self, original: str, expert_id: str) -> str:
        """Rewrite the user query into a precise search phrase, using cache if available."""
        cache_key = f"{original.lower()}|{expert_id}"
        cached = await self._rewrite_cache.get(cache_key)
        if cached is not None:
            if self.valves.DEBUG:
                logger.info(f"[Router] Rewrite cache hit: '{cached}'")
            return cached

        kb_desc = None
        for exp in self._experts:
            if exp["id"] == expert_id:
                kb_desc = exp.get("knowledge_base")
                break

        rewritten = original
        if kb_desc:
            prompt = (
                f"The knowledge base is: {kb_desc}.\n"
                f"Rewrite the user question into a SHORT, TECHNICAL, PAGE-TITLE-LIKE phrase (max 5 words).\n"
                f"Use the SAME language as the question. Answer ONLY the phrase, nothing else.\n\n"
                f"Question: {original}\n"
                f"Phrase:"
            )
        else:
            prompt = (
                "Rewrite the following user question into a concise, technical search phrase in the SAME language.\n"
                "Remove filler words, be specific, and use exact technical terms.\n"
                f"Original question: {original}\n"
                "Rewritten phrase:"
            )
        try:
            rewritten = await self._call_llm(
                prompt=prompt,
                provider=self.valves.classifier_model,
                temperature=0.0,
                max_tokens=25 if kb_desc else 50,
                timeout=10,
            )
            rewritten = rewritten.strip().split("\n")[0].strip('"').strip()
            if not rewritten:
                rewritten = original
        except Exception as e:
            logger.error(f"[Router] Query rewriting failed: {e}")
            rewritten = original

        # Store in cache
        await self._rewrite_cache.set(cache_key, rewritten)
        if self.valves.DEBUG:
            logger.info(
                f"[Router] Rewritten query: '{original[:60]}...' -> '{rewritten}'"
            )
        return rewritten

    # endregion

    # region ── Native RAG injection ──────────────────────────────────────────
    def _inject_rag_guidance(self, expert_id: str, messages: list) -> bool:
        """
        Insert a system message that guides the built-in RAG to use the
        expert's knowledge collection and focus the answer.
        """
        collection = None
        kb_desc = None
        for exp in self._experts:
            if exp["id"] == expert_id:
                collection = exp.get("collection_name")
                kb_desc = exp.get("knowledge_base")
                break

        if not collection:
            if self.valves.DEBUG:
                logger.info(
                    f"[Router] No collection for expert '{expert_id}', skipping RAG guidance."
                )
            return False

        guidance = (
            f"You have access to the knowledge base '{collection}' ({kb_desc or 'specialized documents'}). "
            "Use ONLY this knowledge base to answer the user's question. "
            "If the answer is not found there, say so clearly. "
            "Answer in the same language as the user."
        )
        # Insert as the first system message, keeping other system messages if any
        messages.insert(0, {"role": "system", "content": guidance})
        if self.valves.DEBUG:
            logger.info(
                f"[Router] Injected RAG guidance for collection '{collection}'."
            )
        return True

    # endregion

    # region ── Main routing logic ─────────────────────────────────────────────
    async def _detect_expert(self, text: str) -> Tuple[str, str]:
        """Determine the expert for a user message. Returns (model_id, display_name)."""
        expert_id = await self._classify_query(text)
        if expert_id == self.valves.default_model:
            return self.valves.default_model, "General"
        for exp in self._experts:
            if exp["id"] == expert_id:
                return expert_id, exp["name"]
        return self.valves.default_model, "General"

    async def inlet(self, body: dict, **kwargs) -> dict:
        """
        Open WebUI filter inlet. Extracts user and __event_emitter__ from kwargs.
        """
        user = kwargs.get("user", {})
        __event_emitter__ = kwargs.get("__event_emitter__")

        # Sync cache configuration with current valves
        await self._sync_cache_config()
        # Reload experts only if JSON changed (idempotent)
        self._load_experts()

        messages = body.get("messages", [])
        if not messages:
            return body

        metadata = body.get("metadata", {})
        current_expert = metadata.get("router_expert")
        current_query = messages[-1].get("content", "")

        new_expert, new_name = await self._detect_expert(current_query)

        # Decide whether to switch (with context threshold)
        if not current_expert:
            model_to_use = new_expert
            expert_name = new_name
            change = True
        else:
            if new_expert != current_expert:
                prev_words = set()
                for msg in messages[-3:-1]:
                    if msg.get("role") == "user":
                        prev_words.update(msg.get("content", "").lower().split())
                curr_words = set(current_query.lower().split())
                shared = curr_words.intersection(prev_words)
                if len(shared) < self.valves.change_threshold:
                    model_to_use = new_expert
                    expert_name = new_name
                    change = True
                else:
                    model_to_use = current_expert
                    expert_name = self._get_expert_name(current_expert)
                    change = False
            else:
                model_to_use = current_expert
                expert_name = self._get_expert_name(current_expert)
                change = False

        # Optional query rewriting
        if self.valves.enable_query_rewriting and messages:
            original_query = messages[-1].get("content", "")
            if original_query:
                rewritten = await self._rewrite_query(original_query, model_to_use)
                messages[-1]["content"] = rewritten

        # Optional native RAG injection
        if (
            self.valves.enable_rag_injection
            and model_to_use != self.valves.default_model
        ):
            self._inject_rag_guidance(model_to_use, messages)

        if change:
            metadata["router_expert"] = model_to_use
            body["metadata"] = metadata
            if self.valves.DEBUG:
                logger.info(f"[Router] Switched to expert: {expert_name}")

        body["model"] = model_to_use

        # Notification only on actual expert change (including first assignment)
        if __event_emitter__ and self.valves.notify_change and change:
            notif = self.valves.notification_template.format(expert=expert_name)
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {"type": "info", "content": notif},
                }
            )
        return body

    def _get_expert_name(self, expert_id: str) -> str:
        for exp in self._experts:
            if exp["id"] == expert_id:
                return exp["name"]
        return "General"

    # endregion
