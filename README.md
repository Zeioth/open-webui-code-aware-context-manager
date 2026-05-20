# Code-Aware Context Manager for Open WebUI

A powerful context management filter for Open WebUI that transforms any chat model into a **stateful, code-aware assistant** with:

- Long-term memory
- Dependency tracking
- Iterative task execution
- Meta-cognitive capabilities

---

## Overview

This filter extends Open WebUI with a sophisticated memory and context management system designed for:

- Coding assistants
- Researchers
- Power users

It persistently stores conversation state per project, automatically extracts and classifies code blocks, applies unified diffs, tracks dependencies, and provides a rich set of natural-language commands to control context.

---

# Features

## Persistent Memory

- **Project-based persistent memory**  
  SQLite state shared across conversations with the same `project_id`.

- **Long-Term Memory (LTM)**  
  Semantic search over past messages using ChromaDB embeddings.

- **Semantic response caching**  
  Avoid repeated LLM calls for similar questions (TTL + context-aware).

- **Explicit facts support**  
  Store and inject `[FACT: ...]` statements that expire after a configurable time.

---

## Code Awareness

- Detects fenced and indented code blocks
- Classifies blocks as:
  - base code
  - proposed changes
  - commits
  - errors
  - tool calls

- Tracks file paths and line numbers
- Supports oversized code block handling:
  - truncate
  - summarize
  - warn

---

## Smart Context Management

- **Smart context selection**  
  Replaces sliding-window context with semantic retrieval.

- **Automatic duplicate removal**  
  Keeps only the most relevant version of similar code blocks.

- **Frequency-based prioritization**  
  Frequently mentioned blocks gain importance and expire more slowly.

- **Selective summarization**  
  Different strategies for:
  - code
  - errors
  - tool calls
  - general conversation

- **Hierarchical compression**  
  Compresses older conversation segments into LTM summaries.

---

## Code Modification

- **Unified diff application**
  - Applies diffs automatically
  - Detects conflicts

- **Dependency tracking**
  - AST-based for Python
  - LLM fallback for other languages
  - Marks affected blocks

- **Obsolete marking**
  - Mark code blocks as obsolete
  - Excluded from active context

---

## Intelligence Features

- **Iterative task execution**
  - Break complex goals into steps
  - Generate diffs
  - Optional auto-run via `/iterate`

- **Chain-of-thought reasoning**
  - `/think`

- **Assumption extraction**
  - `/assume`

- **Contradiction detection**
  - Warns about conflicting information

- **Duplicate question detection**
  - Detects repeated questions

- **Confidence scoring**
  - Requests confidence percentages from the model

---

## UX Helpers

- Proactive context warnings
- Command suggestions
- Similar message handling
- Feedback tracking
- LRU project cache

---

# Configuration (Valves)

All settings are configurable through the Open WebUI function interface.

---

## Core Settings

| Valve | Default | Description |
|------|---------|-------------|
| `priority` | `0` | Filter priority level |
| `max_turns` | `20` | Maximum active non-system messages |
| `debug` | `false` | Enable verbose logging |
| `project_id` | `default` | Shared project identifier |
| `context_window_tokens` | `8192` | LLM context size |
| `adaptive_trim` | `true` | Trim only when token limit is exceeded |

---

## Long-Term Memory

| Valve | Default |
|------|---------|
| `long_term_memory_dir` | `/app/backend/data/long_term_memory` |
| `long_term_memory_expiration_days` | `30` |
| `long_term_memory_top_k` | `10` |
| `long_term_memory_similarity_threshold` | `0.65` |
| `ltm_time_decay_hours` | `24.0` |

---

## Smart Context

| Valve | Default |
|------|---------|
| `smart_context_selection` | `false` |
| `smart_context_top_k` | `15` |
| `smart_context_include_last_user` | `true` |

---

## Iterative Mode

| Valve | Default |
|------|---------|
| `enable_iterative_mode` | `true` |
| `iterative_auto_continue` | `false` |
| `iterative_max_steps` | `10` |
| `iterative_diff_format` | `unified` |

---

## Code Awareness

| Valve | Default |
|------|---------|
| `enable_code_awareness` | `true` |
| `auto_detect_code_blocks` | `true` |
| `max_base_code_blocks` | `3` |
| `max_proposed_changes` | `5` |
| `max_committed_changes` | `10` |
| `max_code_block_tokens` | `20000` |

---

## Response Cache

| Valve | Default |
|------|---------|
| `enable_response_cache` | `true` |
| `response_cache_similarity_threshold` | `0.92` |
| `response_cache_ttl_hours` | `24.0` |
| `response_cache_max_entries` | `100` |

---

# Usage

## Installation

1. Copy the entire function code into:

   **Open WebUI → Workspace → Functions → +**

2. Adjust the valves as needed.

3. Activate the function:
   - globally: **Admin Panel → General Settings → Functions**
   - or per model

4. Start chatting.

The assistant will automatically:

- manage context
- remember code across chats
- retrieve relevant past information
- summarize old conversations
- track diffs and dependencies

---

# Commands

## Context Management

```text
/forget all
```

Clear all active context.

```text
/remember
```

Pin the last code block.

```text
/obsolete
```

Mark a block as obsolete.

---

## Iterative Workflows

```text
/iterate implement all missing features
```

Create a step-by-step implementation plan and generate diffs.

```text
/iterate resume
```

Resume interrupted iteration.

---

## Reasoning

```text
/think Why does this code produce an error?
```

Step-by-step reasoning.

```text
/assume "The system should always be available"
```

Extract hidden assumptions.

---

## Facts

```text
/fact add "The API uses port 3000"
```

Store a persistent fact.

```text
/fact list
```

List stored facts.

```text
/fact remove
```

Delete a fact.

---

## Recall

```text
/recall "What was the solution for the login bug?"
```

Retrieve a cached response.

---

# Natural Language Support

The system also understands plain English commands such as:

- "forget my last message"
- "remember this file"
- "mark this block as obsolete"
- "implement all features step by step"

---

# Requirements

The function automatically installs these Python packages:

- `aiohttp`
- `loguru`
- `orjson`
- `tiktoken`
- `sentence-transformers`
- `chromadb`
- `rapidfuzz`

> **Note:**  
> `sentence-transformers` and `chromadb` are heavy dependencies.  
> The first run may take several minutes while packages download.

---

# Recommended Settings

The most important valves to configure are:

- `project_id`
- `context_window_tokens`
- `openai_api_base`
- `openai_api_key`
- `llm_model`

---

# License

GPL-3.0
