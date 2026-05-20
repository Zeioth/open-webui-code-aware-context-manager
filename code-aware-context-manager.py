"""
title: Gestor de Contexto para Código con Memoria LTM y Resúmenes (v5.22)
description: Gestor completo de contexto para asistentes de código. Persiste estado por proyecto, detecta rangos de líneas, aplica diffs, comprime LTM, puntúa importancia, aprende de respuestas, resume código inactivo, admite marcadores manuales, comandos en lenguaje natural para olvidar/recordar, seguimiento de retroalimentación, memoria jerárquica, caché LRU, reranking opcional, detección de dependencias, manejo de bloques enormes, selección inteligente de contexto, compresión jerárquica, eliminación de duplicados, priorización por frecuencia, resúmenes selectivos, comandos para iterar, desduplicación de mensajes similares, detección de contradicciones, razonamiento paso a paso, extracción de supuestos, marcado de obsoleto, sugerencias proactivas, detección de preguntas repetidas y sugerencias de comandos por contexto.
author: zeioth
author_url: https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 5.22.0
license: GPL3
requirements: aiohttp, loguru, orjson, tiktoken, sentence-transformers, chromadb, rapidfuzz
"""

import os
import time
import re
import hashlib
import sqlite3
from collections import OrderedDict, defaultdict
import json
import asyncio
import difflib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Dependencias opcionales
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
        priority: int = Field(default=0, description="Nivel de prioridad del filtro.")
        max_turns: int = Field(
            default=20, description="Máximo de mensajes no‑sistema a conservar."
        )
        debug: bool = Field(
            default=False, description="Activa logs detallados de depuración."
        )
        state_db_path: str = Field(
            default="/app/backend/data/conversation_state.db",
            description="Ruta a la base de datos SQLite.",
        )
        track_line_numbers: bool = Field(
            default=True, description="Extraer rangos de línea de los archivos."
        )
        adaptive_trim: bool = Field(
            default=True, description="Recortar solo cuando se excedan los tokens."
        )
        context_window_tokens: int = Field(
            default=8192, description="Tamaño de la ventana de contexto del modelo."
        )
        use_tiktoken: bool = Field(
            default=True,
            description="Usar tiktoken para contar tokens (si está disponible).",
        )

        long_term_memory_dir: str = Field(
            default="/app/backend/data/long_term_memory",
            description="Directorio para ChromaDB.",
        )
        long_term_memory_expiration_days: int = Field(
            default=30, description="Días hasta que caduca una entrada de LTM."
        )
        long_term_memory_top_k: int = Field(
            default=10, description="Cantidad de resultados a recuperar de LTM."
        )
        long_term_memory_similarity_threshold: float = Field(
            default=0.65, description="Umbral mínimo de similitud coseno."
        )
        ltm_time_decay_hours: float = Field(
            default=24.0, description="Decaimiento temporal para la recuperación LTM."
        )
        enable_reranking: bool = Field(
            default=False,
            description="Usar reranking con cross‑encoder para resultados LTM.",
        )
        reranker_model: str = Field(
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            description="Modelo cross‑encoder para reranking.",
        )
        reranker_top_k: int = Field(
            default=5, description="Número de resultados después del reranking."
        )

        # Selección inteligente de contexto
        smart_context_selection: bool = Field(
            default=False,
            description="Reemplazar la ventana deslizante por recuperación semántica del historial.",
        )
        smart_context_top_k: int = Field(
            default=15, description="Número de mensajes pasados a recuperar."
        )
        smart_context_min_tokens: int = Field(
            default=1024, description="Tokens mínimos a intentar recuperar."
        )
        smart_context_include_last_user: bool = Field(
            default=True, description="Incluir siempre el último mensaje del usuario."
        )

        # Compresión jerárquica
        hierarchical_compression_enabled: bool = Field(
            default=False,
            description="Comprimir periódicamente segmentos viejos de conversación.",
        )
        hierarchical_compression_interval_messages: int = Field(
            default=100,
            description="Número de mensajes tras el cual se activa la compresión.",
        )
        hierarchical_summary_model: str = Field(
            default="", description="Modelo para resúmenes jerárquicos."
        )
        hierarchical_summary_max_tokens: int = Field(
            default=800, description="Máximo de tokens para el resumen jerárquico."
        )

        # Eliminación de duplicados y priorización por frecuencia
        auto_remove_duplicate_blocks: bool = Field(
            default=True,
            description="Eliminar automáticamente bloques de código duplicados más antiguos.",
        )
        max_duplicate_age_hours: float = Field(
            default=6.0,
            description="Diferencia máxima de edad para considerar duplicados.",
        )
        frequency_weight_factor: float = Field(
            default=0.3,
            description="Peso de la frecuencia de menciones en la importancia.",
        )
        min_mentions_for_boost: int = Field(
            default=3,
            description="Mínimo de menciones para aplicar el bonus de frecuencia.",
        )
        frequency_decay_hours: float = Field(
            default=12.0,
            description="Vida media para el decaimiento del bonus de frecuencia.",
        )

        # Características metacognitivas
        enable_confidence_scoring: bool = Field(
            default=True,
            description="Pedir al asistente que estime su confianza (0-100%).",
        )
        confidence_prompt: str = Field(
            default="\n\nDespués de tu respuesta, en una línea nueva, escribe '[Confianza: XX%]' donde XX es tu confianza estimada (0-100) en la corrección e integridad de tu respuesta, basada en el contexto disponible. Si te falta información, da una confianza baja y sugiere qué contexto ayudaría.",
            description="Sufijo que se añade al prompt del sistema para pedir confianza.",
        )
        proactive_context_warning_threshold: float = Field(
            default=0.85,
            description="Fracción de la ventana de tokens que activa una advertencia proactiva (0.0-1.0).",
        )
        proactive_context_warning_message: str = Field(
            default="\n\n⚠️ **Advertencia de contexto**: La conversación está usando más del {percent}% de la ventana de contexto disponible ({used_tokens}/{max_tokens} tokens). Considera usar `/forget` para eliminar partes irrelevantes, `/remember` para fijar contexto importante, o pídeme que resuma partes antiguas.",
            description="Mensaje de advertencia inyectado cuando el contexto está casi lleno.",
        )
        enable_facts: bool = Field(
            default=True,
            description="Permitir hechos explícitos con [FACT: ...] y almacenarlos persistentemente.",
        )
        fact_max_age_days: int = Field(
            default=90, description="Días hasta que un hecho expira (0 = nunca)."
        )
        inject_facts_in_context: bool = Field(
            default=True,
            description="Inyectar siempre los hechos almacenados en el prompt del sistema.",
        )
        fact_importance_boost: float = Field(
            default=1.5,
            description="Multiplicador para la puntuación de importancia de los hechos.",
        )
        fact_command_prefix: str = Field(
            default="/fact",
            description="Prefijo de comando para la gestión de hechos (ej. /fact add, /fact list, /fact remove).",
        )
        enable_auto_fact_detection: bool = Field(
            default=False,
            description="Detectar automáticamente hechos potenciales a partir de mensajes del usuario (experimental).",
        )

        # Ejecución iterativa de tareas (/iterate)
        enable_iterative_mode: bool = Field(
            default=True,
            description="Permitir el comando /iterate para tareas multi‑paso.",
        )
        iterative_auto_continue: bool = Field(
            default=False,
            description="Si es True, /iterate ejecuta todos los pasos sin esperar confirmación del usuario (usar con precaución).",
        )
        iterative_max_steps: int = Field(
            default=10,
            description="Número máximo de pasos por iteración para evitar desbordamiento de contexto.",
        )
        iterative_diff_format: str = Field(
            default="unified",
            description="Formato del diff: 'unified' (por defecto) o 'context'.",
        )
        iterative_planning_model: str = Field(
            default="",
            description="Modelo para la planificación (si está vacío, usa llm_model o summarization_model).",
        )
        iterative_execution_model: str = Field(
            default="",
            description="Modelo para la ejecución de pasos (si está vacío, usa el mismo que planificación).",
        )
        iterative_resume_command: str = Field(
            default="/iterate resume",
            description="Comando para reanudar una iteración interrumpida.",
        )
        natural_language_iterate: bool = Field(
            default=True,
            description="Permitir lenguaje natural para iniciar iteraciones (ej. 'implementa todas las características paso a paso').",
        )

        # Desduplicación de mensajes similares consecutivos
        similar_message_handling: str = Field(
            default="replace",
            description="Acción para mensajes consecutivos muy similares: 'replace' (conserva el último), 'summarize_diff' (genera resumen de diferencias), 'mark_obsolete' o 'none'.",
        )
        similar_message_threshold: float = Field(
            default=0.85,
            description="Umbral de similitud (0-1) para considerar mensajes como duplicados.",
        )
        similar_message_check_code_only: bool = Field(
            default=True,
            description="Aplicar solo a mensajes que contengan bloques de código.",
        )

        # Marcado de obsoleto (/obsolete)
        enable_obsolete_marking: bool = Field(
            default=True,
            description="Permitir marcar bloques de código como obsoletos con /obsolete.",
        )

        # Sugerencias proactivas de resumen (predicción #1)
        proactive_summary_threshold: float = Field(
            default=0.75,
            description="Uso de tokens que activa una sugerencia de resumen.",
        )
        proactive_summary_growth_window: int = Field(
            default=3,
            description="Número de mensajes recientes para estimar la tasa de crecimiento de tokens.",
        )

        # Detección de preguntas repetidas (#2)
        duplicate_question_threshold: float = Field(
            default=0.92,
            description="Umbral de similitud para considerar una pregunta como duplicada.",
        )
        duplicate_question_lookback: int = Field(
            default=20,
            description="Número de mensajes de usuario anteriores a revisar.",
        )

        # Sugerencias de comandos por contexto (#5)
        enable_command_suggestions: bool = Field(
            default=True,
            description="Inyectar mensajes de sistema con sugerencias de comandos útiles según el estado actual.",
        )
        command_suggestion_cooldown_minutes: int = Field(
            default=10,
            description="Minutos que esperar antes de mostrar la misma sugerencia otra vez.",
        )

        # Resúmenes selectivos por tipo de contenido
        selective_summarization: bool = Field(
            default=True,
            description="Aplicar estrategias de resumen distintas según el tipo de contenido.",
        )
        error_preserve_verbatim: bool = Field(
            default=True,
            description="Nunca resumir mensajes de error; mantenerlos textualmente.",
        )
        error_max_age_hours: float = Field(
            default=48.0,
            description="Edad máxima tras la cual los errores pueden ser resumidos (si error_preserve_verbatim es False).",
        )
        code_summary_level: str = Field(
            default="balanced",
            description="Nivel de detalle para resúmenes de código: 'minimal' (solo firmas), 'balanced' (firmas + lógica clave), 'detailed' (estructura completa).",
        )
        general_summary_max_tokens: int = Field(
            default=200,
            description="Máximo de tokens para resumir conversación general.",
        )
        tool_call_preserve: bool = Field(
            default=True,
            description="Preservar cadenas de llamadas a herramientas sin resumir.",
        )
        code_always_keep_signature: bool = Field(
            default=True,
            description="Siempre extraer y conservar firmas de funciones/clases aunque se resuma el código.",
        )
        summary_fallback_model: str = Field(
            default="",
            description="Modelo a usar para resúmenes selectivos (si está vacío, usa summarization_model).",
        )
        summary_include_metadata: bool = Field(
            default=True,
            description="Incluir metadatos (tipo de contenido, rango de tiempo) en los resúmenes.",
        )

        summarize_old_messages: bool = Field(
            default=True, description="Resumir bloques de mensajes descartados."
        )
        summarization_model: str = Field(
            default="gpt-3.5-turbo", description="Modelo de resumen por defecto."
        )
        openai_api_base: str = Field(
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1"),
            description="Base URL de la API compatible con OpenAI.",
        )
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", "dummy"), description="Clave de la API."
        )

        enable_code_awareness: bool = Field(
            default=True,
            description="Activar todas las funcionalidades de análisis de código.",
        )
        code_similarity_threshold: float = Field(
            default=0.85, description="Umbral de similitud para detectar duplicados."
        )
        max_base_code_blocks: int = Field(
            default=3,
            description="Máximo de bloques de código base a mantener en contexto.",
        )

        project_id: str = Field(
            default="default",
            description="Identificador del proyecto (memoria compartida).",
        )

        max_proposed_changes: int = Field(
            default=5, description="Máximo de cambios propuestos a conservar."
        )
        max_committed_changes: int = Field(
            default=10, description="Máximo de cambios aplicados a conservar."
        )
        prioritize_recent_code: bool = Field(
            default=True,
            description="Conservar la versión más reciente del código similar.",
        )
        auto_detect_code_blocks: bool = Field(
            default=True,
            description="Detectar bloques de código cercados e indentados.",
        )
        max_cached_projects: int = Field(
            default=10, description="Máximo de proyectos a mantener en caché LRU."
        )
        track_file_paths: bool = Field(
            default=True, description="Extraer rutas de archivo de los mensajes."
        )
        max_active_blocks: int = Field(
            default=50, description="Máximo de bloques activos por conversación."
        )
        file_path_pattern: str = Field(
            default=r"\b([a-zA-Z0-9_\-\./]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|hpp))\b",
            description="Expresión regular para detectar rutas de archivo.",
        )

        # Manejo de bloques de código demasiado grandes
        max_code_block_tokens: int = Field(
            default=20000,
            description="Tamaño máximo en tokens para un bloque de código (0 = sin límite).",
        )
        code_block_overflow_action: str = Field(
            default="summarize",
            description="Acción para bloques que exceden el límite: 'truncate', 'summarize' o 'warn'.",
        )
        code_block_summary_model: str = Field(
            default="", description="Modelo para resumir bloques demasiado grandes."
        )
        code_block_truncate_keep_head: int = Field(
            default=50,
            description="Número de líneas a conservar del principio al truncar.",
        )
        code_block_truncate_keep_tail: int = Field(
            default=50, description="Número de líneas a conservar del final al truncar."
        )
        code_block_warn_message: str = Field(
            default="[Bloque de código demasiado grande - truncado por el sistema]",
            description="Texto de reemplazo cuando la acción es 'warn'.",
        )

        importance_mention_boost: float = Field(
            default=0.2, description="Bonus adicional por mención (0-1)."
        )
        importance_recency_half_life_hours: float = Field(
            default=2.0, description="Vida media de la recencia para la importancia."
        )

        ltm_compress_after_messages: int = Field(
            default=50,
            description="Número de mensajes tras los cuales comprimir entradas antiguas de LTM.",
        )
        ltm_summarization_trigger_similarity: float = Field(
            default=0.85,
            description="Umbral de similitud para activar resúmenes de duplicados en LTM.",
        )

        enable_diff_application: bool = Field(
            default=True, description="Aplicar diffs unificados al código base."
        )
        preserve_error_context: bool = Field(
            default=True, description="Nunca descartar mensajes de error."
        )
        error_retention_turns: int = Field(
            default=15, description="Número de turnos que mantener los errores vivos."
        )
        block_expiration_hours: float = Field(
            default=24.0,
            description="Horas tras las cuales los bloques inactivos pueden expirar.",
        )
        proposed_change_retention_turns: int = Field(
            default=20, description="Turnos que mantener los cambios propuestos."
        )
        preserve_tool_calls: bool = Field(
            default=True, description="Mantener cadenas de tool calls intactas."
        )

        enable_feedback_tracking: bool = Field(
            default=True,
            description="Registrar retroalimentación sobre cambios aplicados.",
        )
        feedback_history_limit: int = Field(
            default=10,
            description="Máximo de entradas de retroalimentación por proyecto.",
        )
        inject_feedback_context: bool = Field(
            default=True,
            description="Inyectar retroalimentación reciente en el prompt del sistema.",
        )
        feedback_importance_penalty_for_failure: float = Field(
            default=2.0, description="Reducción de importancia para cambios fallidos."
        )

        code_block_pattern: str = Field(
            default="```(\\w*)\\n(.*?)```",
            description="Expresión regular para bloques de código cercados.",
        )
        diff_pattern: str = Field(
            default="@@\\s*-([0-9]+),([0-9]+)\\s*\\+([0-9]+),([0-9]+)\\s*@@",
            description="Expresión regular para hunks de diff.",
        )
        commit_pattern: str = Field(
            default="commit\\s+([a-f0-9]{7,40})",
            description="Expresión regular para hashes de commit.",
        )

        enable_dependency_tracking: bool = Field(
            default=False,
            description="Extraer dependencias y marcar bloques afectados.",
        )
        dependency_extraction_model: str = Field(
            default="", description="Modelo para extraer dependencias."
        )
        dependency_refresh_on_update: bool = Field(
            default=True,
            description="Volver a extraer dependencias cuando se actualiza un bloque.",
        )
        affected_importance_penalty: float = Field(
            default=0.7,
            description="Multiplicador de importancia para bloques afectados por cambios en dependencias.",
        )
        affected_decay_hours: float = Field(
            default=4.0,
            description="Horas hasta que se elimina la marca de 'afectado'.",
        )

        track_active_code_age: bool = Field(
            default=True,
            description="Marcar código como inactivo tras un tiempo de espera.",
        )
        active_code_timeout_minutes: int = Field(
            default=30,
            description="Minutos tras los cuales el código inactivo puede ser resumido.",
        )

        summarize_inactive_code: bool = Field(
            default=True, description="Resumir bloques de código inactivos."
        )
        inactive_code_summary_model: str = Field(
            default="gpt-3.5-turbo", description="Modelo para resumir código inactivo."
        )

        llm_model: str = Field(
            default="",
            description="Modelo preferido (ej. 'ollama/llama3.2:3b'). Si falla, usa summarization_model.",
        )

        enable_forget_command: bool = Field(
            default=True, description="Permitir comandos /forget."
        )
        enable_natural_language_forget: bool = Field(
            default=True, description="Interpretar olvido en lenguaje natural."
        )
        natural_language_forget_model: str = Field(
            default="", description="Modelo para interpretar intenciones de olvido."
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
        }
        self.code_pattern = re.compile(self.valves.code_block_pattern, re.DOTALL)
        self.diff_pattern = re.compile(self.valves.diff_pattern)
        self.commit_pattern = re.compile(self.valves.commit_pattern, re.IGNORECASE)

        if HAS_TIKTOKEN and self.valves.use_tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self._log_debug("Tiktoken inicializado")
            except Exception as e:
                logger.warning(f"Fallo al cargar tiktoken: {e}")

        if HAS_SENTENCE and HAS_CHROMA and self.valves.enable_code_awareness:
            self._init_long_term_memory()
        else:
            logger.warning("Memoria a largo plazo o análisis de código desactivado")

        if self.valves.enable_reranking and HAS_CROSS_ENCODER:
            self._load_reranker()

        if self.valves.enable_facts:
            self._log_debug("Almacenamiento de hechos activado.")

    # --------------------------------------------------------------------------
    # Caché LRU
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
            self._log_debug(f"Expulsando proyecto {oldest} de la caché")
            del self._conversation_state[oldest]
        return state

    def _set_state(self, project_id: str, state: Dict):
        self._conversation_state[project_id] = state
        self._conversation_state.move_to_end(project_id)
        self._save_state_to_db(project_id, state)

    # --------------------------------------------------------------------------
    # Base de datos
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
        self._log_debug(f"Base de datos de estado inicializada en {db_path}")

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
            "last_suggestion_timestamp": state.get("last_suggestion_timestamp", 0),
        }
        self._db_conn.execute(
            "REPLACE INTO conversation_state (project_id, state_json, updated_at) VALUES (?, ?, ?)",
            (project_id, json.dumps(serializable), time.time()),
        )
        self._db_conn.commit()

    # --------------------------------------------------------------------------
    # Depuración
    # --------------------------------------------------------------------------
    def _log_debug(self, msg: str):
        if self.valves.debug:
            logger.debug(msg)

    # --------------------------------------------------------------------------
    # Inicialización de LTM
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
        self._log_debug("LTM lista")

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
                self._log_debug(f"Purgadas {len(expired['ids'])} memorias expiradas")
        except Exception as e:
            logger.warning(f"Fallo al purgar memorias: {e}")

    # --------------------------------------------------------------------------
    # Advertencia proactiva de contexto
    # --------------------------------------------------------------------------
    def _check_context_usage_and_warn(
        self, system_msgs: List[dict], history_msgs: List[dict]
    ) -> Optional[str]:
        if self.valves.proactive_context_warning_threshold <= 0:
            return None
        total_tokens = self._estimate_tokens(system_msgs + history_msgs)
        max_tokens = self.valves.context_window_tokens
        if max_tokens <= 0:
            return None
        usage_ratio = total_tokens / max_tokens
        if usage_ratio >= self.valves.proactive_context_warning_threshold:
            percent = int(usage_ratio * 100)
            return self.valves.proactive_context_warning_message.format(
                percent=percent, used_tokens=total_tokens, max_tokens=max_tokens
            )
        return None

    # --------------------------------------------------------------------------
    # Gestión de hechos
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
        self._log_debug(f"Hecho añadido: {fact_text[:50]}...")

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
            self._log_debug(f"Hecho eliminado: {fact_text_or_index}")

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
        return "## Hechos acordados explícitamente\n" + "\n".join(
            [f"- {fact}" for fact in active_facts]
        )

    # --------------------------------------------------------------------------
    # Manejo de bloques de código demasiado grandes
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
            f"Bloque de código de {estimated} tokens excede el límite ({max_tokens}). Acción: {action}"
        )

        if action == "truncate":
            lines = code.splitlines()
            head = self.valves.code_block_truncate_keep_head
            tail = self.valves.code_block_truncate_keep_tail
            if len(lines) <= head + tail:
                return code
            truncated = "\n".join(
                lines[:head]
                + [f"... [{len(lines) - head - tail} líneas truncadas] ..."]
                + lines[-tail:]
            )
            return truncated

        elif action == "summarize":
            model = (
                self.valves.code_block_summary_model
                or self.valves.llm_model
                or self.valves.summarization_model
            )
            prompt = f"""Resume el siguiente bloque de código {language}. Céntrate en:
- Qué hace el código (propósito)
- Funciones/clases principales y sus firmas
- Lógica o algoritmos importantes
- Dependencias externas relevantes

Mantén el resumen conciso (máx. 300 palabras).

Código:
```{language}
{code[:8000]}
```"""
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="Eres un asistente de resumen de código. Solo genera el resumen, sin texto extra.",
                model_override=model,
                max_tokens=500,
                temperature=0.2,
            )
            if summary:
                return f"[Resumen automático de un bloque de código de {estimated} tokens]\n{summary}"
            else:
                return f"[Bloque de código demasiado grande, no se pudo resumir] Tamaño original: {estimated} tokens."

        elif action == "warn":
            return self.valves.code_block_warn_message

        else:
            return code

    # --------------------------------------------------------------------------
    # Extracción de dependencias
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
        prompt = f"""Analiza el siguiente código y extrae sus dependencias:
- Sentencias import
- Llamadas a funciones externas o definidas por el usuario (por nombre)
- Instanciaciones o referencias a clases
- Rutas de archivo (ej. './utils.py')

Devuelve un array JSON de cadenas, cada cadena un identificador o ruta simple.
Si no hay dependencias, devuelve [].

Código:
```{code[:1500]}```

Ejemplo de salida: ["os", "Path", "utils.py", "calculate_total"]
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Solo genera JSON arrays.",
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
        self._log_debug(f"Dependencias actualizadas para bloque {block_hash}: {deps}")

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
                    f"Bloque {h} marcado como potencialmente afectado por dependencia con {changed_hash}"
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
                self._log_debug(f"Eliminada marca de afectado para bloque {block.hash}")
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
                self._log_debug(f"Cargado modelo reranker {self.valves.reranker_model}")
            except Exception as e:
                logger.warning(f"Fallo al cargar el reranker: {e}")
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
    # Extracción y clasificación de código
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
    # Prompts de resumen según tipo de contenido
    # --------------------------------------------------------------------------
    def _get_summary_prompt_for_content(
        self, content_type: ContentType, text: str, max_tokens: int
    ) -> str:
        if not self.valves.selective_summarization:
            return f"Resume la siguiente conversación. Conserva las decisiones clave, acciones y contexto importante. Sé conciso.\n\n{text}"

        if content_type == ContentType.ERROR:
            if self.valves.error_preserve_verbatim:
                return text
            else:
                return f"Resume el siguiente mensaje de error, conservando el tipo de error, ubicación y causa raíz. No omitas detalles técnicos.\n\n{text}"
        elif content_type in (
            ContentType.BASE_CODE,
            ContentType.PROPOSED_CHANGE,
            ContentType.COMMITTED_CHANGE,
        ):
            level = self.valves.code_summary_level
            if level == "minimal":
                instruction = "Extrae solo las firmas de funciones/clases y el propósito general. No incluyas detalles de implementación."
            elif level == "detailed":
                instruction = "Resume el código, conservando funciones clave, clases, lógica importante y comentarios. Un nivel de detalle medio."
            else:
                instruction = "Resume el código, centrándote en lo que hace, sus funciones/clases principales y cualquier lógica no trivial."
            return f"{instruction}\n\n```\n{text[:3000]}\n```"
        elif content_type == ContentType.TOOL_CALL:
            if self.valves.tool_call_preserve:
                return text
            else:
                return f"Resume la siguiente secuencia de llamadas a herramientas, conservando los nombres de las herramientas y los parámetros principales.\n\n{text}"
        else:
            return f"Resume la siguiente conversación, conservando las decisiones clave, elementos de acción y problemas pendientes. Sé conciso (objetivo {max_tokens} tokens).\n\n{text}"

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
                                f"LLM falló con modelo {model}, estado {resp.status}"
                            )
            except Exception as e:
                self._log_debug(f"Error con modelo {model}: {e}")
                continue
        logger.warning(f"Todos los modelos LLM fallaron para prompt: {prompt[:100]}...")
        return None

    # --------------------------------------------------------------------------
    # Funciones auxiliares (nombres, firmas, similitud)
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
                f"Función `{name}({params})` - {docstring}"
                if docstring
                else f"Función `{name}({params})`"
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
            return f"Clase `{name}` - {docstring}" if docstring else f"Clase `{name}`"
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
                        f"Importancia aumentada para {block.hash} por mención de '{name}'"
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
    # Aplicación de diffs
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
                self._log_debug(
                    "El hunk del diff no coincide con el código actual; se omite."
                )
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
            self._log_debug(f"Diff aplicado al bloque base {base_block.hash}")
            return True
        return False

    # --------------------------------------------------------------------------
    # Detección de conflictos
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
                    f"Conflicto detectado entre cambios propuestos {existing.hash} y {new_block.hash}"
                )
                return True
        return False

    # --------------------------------------------------------------------------
    # Expiración de bloques (ignorando los anclados y obsoletos)
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
            self._log_debug(f"Bloque expirado: {h}")
        if to_remove:
            self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Resumen de bloques inactivos
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
                    content=f"[Resumen de código inactivo]\n{summary}",
                    content_type=ContentType.GENERAL,
                    timestamp=time.time(),
                    is_active=False,
                    importance_score=block.importance_score * 0.5,
                )
                state["active_blocks"][h] = summary_block
                self._log_debug(f"Bloque inactivo resumido: {h}")
        self._set_state(project_id, state)

    async def _summarize_code_block(self, block: CodeBlock) -> Optional[str]:
        if not self.valves.summarize_inactive_code or not HAS_AIOHTTP:
            return None
        sig = self._extract_signature(block.content)
        if sig:
            prompt = f"""El bloque de código tiene la firma: {sig}
Proporciona una descripción muy breve (máx. 50 palabras) de lo que hace este código.
Código:
```{block.content[:1000]}```
"""
        else:
            prompt = f"""Resume el siguiente bloque de código. Incluye:
1. Qué hace el código (propósito)
2. Funciones/clases/variables clave
3. Lógica importante o casos extremos
Mantén el resumen por debajo de 150 palabras.

```{block.content[:1500]}```
"""
        return await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente de resumen de código.",
            model_override=self.valves.inactive_code_summary_model,
            max_tokens=200,
            temperature=0.2,
        )

    # --------------------------------------------------------------------------
    # Compresión jerárquica
    # --------------------------------------------------------------------------
    async def _hierarchical_compress(self, project_id: str, state: Dict):
        if not self.valves.hierarchical_compression_enabled:
            return
        last_ts = state.get("last_compression_timestamp", 0)
        if time.time() - last_ts < 3600:
            return

        try:
            where_filter = {
                "project_id": project_id,
                "$and": [
                    {"is_hierarchical_summary": {"$ne": True}},
                    {"timestamp": {"$lt": datetime.utcnow().isoformat()}},
                ],
            }
            results = self.memory_collection.get(
                where=where_filter,
                include=["documents", "metadatas", "ids"],
                limit=self.valves.hierarchical_compression_interval_messages * 2,
            )
        except Exception as e:
            self._log_debug(
                f"Fallo al recuperar mensajes para compresión jerárquica: {e}"
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
            key=lambda x: x[2].get("timestamp", ""),
        )
        to_compress = pairs[: self.valves.hierarchical_compression_interval_messages]

        texts = "\n---\n".join([doc for _, doc, _ in to_compress])
        model = (
            self.valves.hierarchical_summary_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Resume el siguiente segmento de conversación, centrándote en:
- Decisiones técnicas clave
- Cambios de código y sus resultados
- Problemas resueltos o aún abiertos
- Contexto importante para interacciones futuras

Mantén el resumen conciso (máx. {self.valves.hierarchical_summary_max_tokens // 4} palabras).

Segmento:
{texts[:4000]}
"""
        summary = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente de resúmenes jerárquicos.",
            model_override=model,
            max_tokens=self.valves.hierarchical_summary_max_tokens,
            temperature=0.2,
        )
        if not summary:
            self._log_debug("Compresión jerárquica: falló la generación del resumen.")
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
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_hierarchical_summary": True,
                    "summary_level": 1,
                    "source_message_ids": ",".join(
                        [id for id, _, _ in to_compress][:5]
                    ),
                }
            ],
            documents=[
                f"[Resumen jerárquico de {len(to_compress)} mensajes]\n{summary}"
            ],
        )
        self._log_debug(
            f"Compresión jerárquica: creado resumen para {len(to_compress)} mensajes"
        )
        state["last_compression_timestamp"] = time.time()
        self._set_state(project_id, state)

    # --------------------------------------------------------------------------
    # Marcado de obsoleto
    # --------------------------------------------------------------------------
    async def _parse_obsolete_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_obsolete_marking:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Eres un intérprete de comandos. El usuario quiere marcar algo como obsoleto (ya no relevante).
Acciones posibles:
- "obsolete_last": marcar el último bloque de código como obsoleto.
- "obsolete_n": marcar los últimos N bloques de código como obsoletos.
- "obsolete_file": marcar todos los bloques relacionados con un archivo como obsoletos.
- "obsolete_block": marcar un bloque específico (por hash, nombre de función o línea) como obsoleto.
- "obsolete_all": marcar todos los bloques activos como obsoletos.
- "revive_last": quitar la marca de obsoleto del último bloque.
- "revive_file": quitar la marca de obsoleto de bloques relacionados con un archivo.
- "revive_all": quitar todas las marcas de obsoleto.

Mensaje del usuario: "{user_message}"

Si el usuario quiere claramente marcar/desmarcar como obsoleto, genera JSON con la acción y los parámetros.
Si no, genera {{"action": "none"}}

Ejemplos:
- "marca este bloque como obsoleto" -> {{"action": "obsolete_last"}}
- "los últimos dos cambios ya no sirven" -> {{"action": "obsolete_n", "n": 2}}
- "el archivo utils.py está obsoleto" -> {{"action": "obsolete_file", "file": "utils.py"}}
- "revive la función calcular_total" -> {{"action": "revive_block", "description": "calcular_total"}}

Devuelve solo JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Solo devuelves JSON.",
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
                self._log_debug(f"Intención de obsoleto interpretada: {data}")
                return data
        except:
            pass
        return None

    def _set_obsolete_flag(self, blocks: List[CodeBlock], obsolete_value: bool):
        for blk in blocks:
            blk.obsolete = obsolete_value
            blk._update_importance()
            self._log_debug(f"Bloque {blk.hash} marcado como obsoleto={obsolete_value}")
        return len(blocks)

    async def _execute_obsolete_intent(self, project_id: str, intent: Dict) -> str:
        state = self._get_state(project_id)
        if not state:
            return "No hay contexto activo para marcar como obsoleto."
        action = intent.get("action", "")
        blocks = list(state["active_blocks"].values())
        if not blocks:
            return "No hay bloques disponibles."
        if action == "obsolete_last":
            last_block = max(blocks, key=lambda b: b.timestamp)
            self._set_obsolete_flag([last_block], True)
            return "Último bloque marcado como obsoleto."
        elif action == "obsolete_n":
            n = intent.get("n", 1)
            blocks_by_time = sorted(blocks, key=lambda b: b.timestamp, reverse=True)
            to_obsolete = blocks_by_time[:n]
            count = self._set_obsolete_flag(to_obsolete, True)
            return f"{count} bloque(s) marcado(s) como obsoleto(s)."
        elif action == "obsolete_file":
            file_path = intent.get("file", "")
            if not file_path:
                return "No se especificó archivo."
            to_obsolete = [
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = self._set_obsolete_flag(to_obsolete, True)
            return f"{count} bloque(s) relacionado(s) con {file_path} marcado(s) como obsoleto(s)."
        elif action == "obsolete_block":
            desc = intent.get("description", "") or intent.get("hash", "")
            if not desc:
                return "No se especificó identificador del bloque."
            matches = [
                blk
                for blk in blocks
                if desc in blk.content
                or (blk.hash and desc in blk.hash)
                or (blk.file_path and desc in blk.file_path)
            ]
            count = self._set_obsolete_flag(matches, True)
            return f"{count} bloque(s) que coinciden con '{desc}' marcado(s) como obsoleto(s)."
        elif action == "obsolete_all":
            count = self._set_obsolete_flag(blocks, True)
            return f"Todos los {count} bloque(s) marcados como obsoletos."
        elif action == "revive_last":
            last_block = max(blocks, key=lambda b: b.timestamp)
            self._set_obsolete_flag([last_block], False)
            return "Marca de obsoleto eliminada del último bloque."
        elif action == "revive_file":
            file_path = intent.get("file", "")
            if not file_path:
                return "No se especificó archivo."
            to_revive = [
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = self._set_obsolete_flag(to_revive, False)
            return f"Marca de obsoleto eliminada de {count} bloque(s) relacionado(s) con {file_path}."
        elif action == "revive_all":
            count = self._set_obsolete_flag(blocks, False)
            return f"Marca de obsoleto eliminada de todos los {count} bloque(s)."
        else:
            return "Acción de obsoleto no reconocida."

    # --------------------------------------------------------------------------
    # Sugerencia proactiva de resumen (predicción #1)
    # --------------------------------------------------------------------------
    async def _check_and_suggest_summarization(
        self, project_id: str, current_tokens: int, max_tokens: int
    ) -> Optional[str]:
        if not self.valves.proactive_summary_threshold or max_tokens <= 0:
            return None
        usage_ratio = current_tokens / max_tokens
        if usage_ratio < self.valves.proactive_summary_threshold:
            return None
        state = self._get_state(project_id)
        if not state:
            return None
        obsolete_count = sum(1 for b in state["active_blocks"].values() if b.obsolete)
        inactive_count = sum(
            1
            for b in state["active_blocks"].values()
            if not b.is_active and not b.obsolete
        )
        suggestion_parts = []
        if obsolete_count > 2:
            suggestion_parts.append(
                f"- Tienes {obsolete_count} bloque(s) obsoleto(s). Usa `/obsolete revive` si los necesitas, o se ignorarán."
            )
        if inactive_count > 5:
            suggestion_parts.append(
                f"- Hay {inactive_count} bloque(s) de código inactivos. Usa `/forget` para eliminarlos o `/remember` para anclar los importantes."
            )
        if not suggestion_parts:
            suggestion_parts.append(
                f"- El contexto está al {int(usage_ratio*100)}% de capacidad. Considera resumir conversaciones antiguas con `/summarize` o usar `/forget` para liberar espacio."
            )
        return "⚠️ **Contexto casi lleno**\n" + "\n".join(suggestion_parts)

    # --------------------------------------------------------------------------
    # Detección de preguntas repetidas (#2)
    # --------------------------------------------------------------------------
    async def _find_duplicate_question(
        self, query: str, project_id: str
    ) -> Optional[Dict]:
        if not self.valves.duplicate_question_threshold or not HAS_SENTENCE:
            return None
        results = self.memory_collection.get(
            where={"project_id": project_id, "role": "user"},
            limit=self.valves.duplicate_question_lookback,
            include=["documents", "metadatas"],
        )
        if not results or not results["documents"]:
            return None
        query_embedding = self.embedder.encode(query).tolist()
        best_sim = 0.0
        best_entry = None
        for i, doc in enumerate(results["documents"]):
            doc_embedding = self.embedder.encode(doc).tolist()
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            if sim > best_sim and sim >= self.valves.duplicate_question_threshold:
                best_sim = sim
                best_entry = {
                    "content": doc,
                    "sim": sim,
                    "timestamp": results["metadatas"][0][i].get("timestamp"),
                }
        return best_entry

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # --------------------------------------------------------------------------
    # Sugerencias de comandos por contexto (#5)
    # --------------------------------------------------------------------------
    async def _suggest_commands(self, project_id: str, state: Dict) -> Optional[str]:
        if not self.valves.enable_command_suggestions:
            return None
        last_suggestion = state.get("last_suggestion_timestamp", 0)
        if (
            time.time() - last_suggestion
            < self.valves.command_suggestion_cooldown_minutes * 60
        ):
            return None
        suggestions = []
        obsolete_count = sum(1 for b in state["active_blocks"].values() if b.obsolete)
        if obsolete_count > 3:
            suggestions.append(
                "`/forget all` o `/obsolete revive` para gestionar bloques obsoletos."
            )
        high_importance_unpinned = [
            b
            for b in state["active_blocks"].values()
            if b.importance_score > 6.0 and not b.pinned
        ]
        if len(high_importance_unpinned) > 2:
            suggestions.append(
                "`/remember` para anclar bloques importantes y evitar que se resuman o expiren."
            )
        if state.get("recent_changes"):
            suggestions.append(
                "`/iterate` para aplicar automáticamente los cambios pendientes."
            )
        if len(state["active_blocks"]) > 30:
            suggestions.append(
                "Considera usar `/summarize` para comprimir conversaciones antiguas (si tienes la funcionalidad)."
            )
        if suggestions:
            state["last_suggestion_timestamp"] = time.time()
            self._set_state(project_id, state)
            return "💡 **Tip**: " + " ".join(suggestions)
        return None

    # --------------------------------------------------------------------------
    # Resumen de mensajes (ya existente, adaptado)
    # --------------------------------------------------------------------------
    async def _summarize_messages(
        self, messages: List[dict], is_code_context: bool = False
    ) -> Optional[str]:
        if not HAS_AIOHTTP or not messages:
            return None

        content_type_counts = defaultdict(int)
        for m in messages:
            content = m.get("content", "")
            extracted = await self._extract_code_blocks(content)
            ctype = self._classify_content(content, extracted)
            content_type_counts[ctype] += 1

        if (
            self.valves.selective_summarization
            and self.valves.error_preserve_verbatim
            and content_type_counts.get(ContentType.ERROR, 0) > 0
        ):
            return "\n".join(
                [f"{m.get('role')}: {m.get('content', '')}" for m in messages]
            )

        conv_text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in messages
        )
        if content_type_counts:
            dominant_type = max(content_type_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_type = ContentType.GENERAL
        prompt = self._get_summary_prompt_for_content(
            dominant_type, conv_text, self.valves.general_summary_max_tokens
        )
        model_override = (
            self.valves.summary_fallback_model
            if self.valves.summary_fallback_model
            else None
        )
        max_tokens = (
            self.valves.general_summary_max_tokens
            if dominant_type == ContentType.GENERAL
            else 500
        )
        summary = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente de resúmenes concisos.",
            model_override=model_override,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        if not summary:
            return None
        if self.valves.selective_summarization and self.valves.summary_include_metadata:
            summary = f"[Resumen de {len(messages)} mensajes, tipo: {dominant_type.value}]\n{summary}"
        return summary

    # --------------------------------------------------------------------------
    # Duplicación de mensajes consecutivos similares
    # --------------------------------------------------------------------------
    async def _summarize_diff_between_messages(self, msg1: dict, msg2: dict) -> str:
        content1 = msg1.get("content", "")
        content2 = msg2.get("content", "")
        prompt = f"""El usuario envió dos mensajes consecutivos muy similares. El segundo es probablemente una actualización/corrección del primero.

PRIMER MENSAJE:
{content1[:1500]}

SEGUNDO MENSAJE (más reciente):
{content2[:1500]}

Proporciona un resumen muy conciso (máx. 150 palabras) de lo que cambió entre ellos. Céntrate solo en las diferencias. Si el segundo mensaje reemplaza completamente al primero, di "Versión actualizada" e incluye el contenido nuevo.
"""
        diff_summary = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente que resume diferencias entre mensajes similares.",
            max_tokens=300,
            temperature=0.2,
        )
        if not diff_summary:
            return f"[Versión actualizada]\n{content2}"
        return f"[Resumen de cambios]\n{diff_summary}\n\n[Versión final]\n{content2}"

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
                first = history_msgs[i]
                last = history_msgs[j - 1]
                contains_code = False
                if self.valves.similar_message_check_code_only:
                    for k in range(i, j):
                        if "```" in history_msgs[k].get("content", ""):
                            contains_code = True
                            break
                else:
                    contains_code = True
                if contains_code:
                    sim = self._calculate_code_similarity(
                        first.get("content", ""), last.get("content", "")
                    )
                    if sim >= self.valves.similar_message_threshold:
                        action = self.valves.similar_message_handling
                        if action == "replace":
                            new_history.append(last)
                            i = j
                            continue
                        elif action == "summarize_diff" and j - i == 2:
                            diff_summary = await self._summarize_diff_between_messages(
                                first, last
                            )
                            new_history.append(
                                {"role": first.get("role"), "content": diff_summary}
                            )
                            i = j
                            continue
            new_history.append(current)
            i += 1
        return new_history

    # --------------------------------------------------------------------------
    # Razonamiento paso a paso (/think)
    # --------------------------------------------------------------------------
    async def _parse_cot_intent(self, user_message: str) -> Optional[str]:
        if not self.valves.enable_cot_on_demand:
            return None
        if user_message.strip().startswith("/think"):
            parts = user_message.split(maxsplit=1)
            if len(parts) > 1:
                return parts[1]
            return "¿Sobre qué te gustaría que piense paso a paso?"
        if re.search(
            r"\b(think step by step|razona paso a paso|piensa paso a paso)\b",
            user_message,
            re.IGNORECASE,
        ):
            return user_message
        return None

    async def _generate_cot(self, question: str, context: str) -> str:
        model = (
            self.valves.cot_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Eres un asistente que razona paso a paso. Por favor, piensa paso a paso para responder la siguiente pregunta. Muestra tu razonamiento claramente, luego proporciona una respuesta final.

Contexto:
{context[:2000]}

Pregunta: {question}

Procede paso a paso.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un razonador meticuloso. Genera primero el razonamiento paso a paso, luego la respuesta final.",
            model_override=model,
            max_tokens=self.valves.cot_max_tokens,
            temperature=0.3,
        )
        return response or "No se pudo generar el razonamiento."

    # --------------------------------------------------------------------------
    # Extracción de supuestos (/assume)
    # --------------------------------------------------------------------------
    async def _parse_assumption_intent(self, user_message: str) -> Optional[str]:
        if not self.valves.enable_assumption_extraction:
            return None
        if user_message.strip().startswith("/assume"):
            parts = user_message.split(maxsplit=1)
            if len(parts) > 1:
                return parts[1]
            return None
        if re.search(
            r"\b(qué supuestos|qué asunciones|underlying assumptions)\b",
            user_message,
            re.IGNORECASE,
        ):
            return user_message
        return None

    async def _extract_assumptions(self, text: str) -> str:
        model = (
            self.valves.assumption_extraction_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Analiza la siguiente declaración o código y extrae los supuestos subyacentes. Enumera claramente cada supuesto. También señala premisas implícitas o sesgos.

Declaración/Código:
{text[:2000]}

Da como resultado una lista estructurada de supuestos y un breve comentario sobre su validez o impacto.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente analítico que extrae supuestos ocultos.",
            model_override=model,
            max_tokens=800,
            temperature=0.2,
        )
        return response or "No se pudieron extraer supuestos."

    # --------------------------------------------------------------------------
    # Detección de contradicciones
    # --------------------------------------------------------------------------
    async def _detect_contradictions(
        self, conversation_messages: List[dict]
    ) -> Optional[str]:
        if (
            not self.valves.enable_contradiction_detection
            or len(conversation_messages) < 4
        ):
            return None
        model = (
            self.valves.contradiction_detection_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        recent = conversation_messages[-10:]
        conv_text = "\n".join(
            [f"{m.get('role')}: {m.get('content', '')[:500]}" for m in recent]
        )
        prompt = f"""Analiza la siguiente conversación en busca de contradicciones. Busca afirmaciones que entren en conflicto entre sí, como:
- El usuario dice A y luego dice no A
- El asistente proporciona información inconsistente
- Requisitos de código que se contradicen

Si encuentras una contradicción, genera un JSON con "contradiction": true, "explanation": "..." y "suggestion": "...".
Si no hay contradicción, genera {{"contradiction": false}}.

Conversación:
{conv_text}

Devuelve solo JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un asistente detector de contradicciones. Devuelves solo JSON.",
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
                return f"⚠️ **Contradicción detectada**: {data.get('explanation', '')}\n\nSugerencia: {data.get('suggestion', '')}"
        except:
            pass
        return None

    # --------------------------------------------------------------------------
    # Comandos de olvido (forget)
    # --------------------------------------------------------------------------
    async def _parse_forget_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_natural_language_forget:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Interpreta la intención de olvido. Acciones posibles: forget_last, forget_n (con n), forget_file (con file), forget_block (con hash), forget_all.
Mensaje: "{user_message}"
Salida JSON: {{"action": "...", "n": N, "file": "...", "hash": "..."}} o {{"action": "none"}}
Solo JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Solo JSON.",
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
                self._log_debug(f"Intención de olvido interpretada: {data}")
                return data
        except:
            pass
        return None

    async def _execute_forget_intent(self, project_id: str, intent: Dict) -> str:
        state = self._get_state(project_id)
        if not state:
            return "No hay contexto activo para olvidar."
        action = intent.get("action")
        if action == "forget_last":
            if state["active_blocks"]:
                last_hash = max(
                    state["active_blocks"].keys(),
                    key=lambda h: state["active_blocks"][h].timestamp,
                )
                del state["active_blocks"][last_hash]
                self._log_debug(f"Olvidado último bloque: {last_hash}")
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
            return "Intención de olvido no reconocida."

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
    # Recordar / anclar (pin) con lenguaje natural
    # --------------------------------------------------------------------------
    async def _parse_remember_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_natural_language_forget:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Eres un intérprete de comandos. El usuario quiere anclar o recordar contexto.
Acciones posibles: pin_last, pin_n (con n), pin_file (con file), pin_block (con description), pin_all, unpin_last, unpin_file, unpin_all, etc.
Mensaje: "{user_message}"
Salida JSON con acción y parámetros. Si no es intención de recordar: {{"action": "none"}}
Solo JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Solo JSON.",
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
                self._log_debug(f"Intención de recordar interpretada: {data}")
                return data
        except:
            pass
        return None

    async def _execute_remember_intent(self, project_id: str, intent: Dict) -> str:
        state = self._get_state(project_id)
        if not state:
            return "No hay contexto activo para anclar."

        def set_pinned(blocks, pinned_value):
            count = 0
            for blk in blocks:
                blk.pinned = pinned_value
                if pinned_value:
                    blk.importance_score = 10.0
                else:
                    blk._update_importance()
                count += 1
            return count

        action = intent.get("action", "")
        blocks = list(state["active_blocks"].values())
        if not blocks:
            return "No hay bloques disponibles."

        if action == "pin_last":
            last_block = max(blocks, key=lambda b: b.timestamp)
            set_pinned([last_block], True)
            return "Anclado el último bloque de código."
        elif action == "pin_n":
            n = intent.get("n", 1)
            blocks_by_time = sorted(blocks, key=lambda b: b.timestamp, reverse=True)
            to_pin = blocks_by_time[:n]
            count = set_pinned(to_pin, True)
            return f"Anclados {count} bloque(s)."
        elif action == "pin_file":
            file_path = intent.get("file", "")
            if not file_path:
                return "No se especificó archivo."
            to_pin = [
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = set_pinned(to_pin, True)
            return f"Anclados {count} bloque(s) relacionados con {file_path}."
        elif action == "pin_block":
            desc = intent.get("description", "") or intent.get("hash", "")
            if not desc:
                return "No se especificó identificador."
            matches = [
                blk
                for blk in blocks
                if desc in blk.content
                or (blk.hash and desc in blk.hash)
                or (blk.file_path and desc in blk.file_path)
            ]
            count = set_pinned(matches, True)
            return f"Anclados {count} bloque(s) que coinciden con '{desc}'."
        elif action == "pin_all":
            count = set_pinned(blocks, True)
            return f"Anclados todos los {count} bloque(s) activos."
        elif action == "unpin_last":
            last_block = max(blocks, key=lambda b: b.timestamp)
            set_pinned([last_block], False)
            return "Desanclado el último bloque."
        elif action == "unpin_file":
            file_path = intent.get("file", "")
            if not file_path:
                return "No se especificó archivo."
            to_unpin = [
                blk for blk in blocks if blk.file_path and file_path in blk.file_path
            ]
            count = set_pinned(to_unpin, False)
            return f"Desanclados {count} bloque(s) relacionados con {file_path}."
        elif action == "unpin_all":
            count = set_pinned(blocks, False)
            return f"Desanclados todos los {count} bloque(s)."
        else:
            return "Acción de anclaje no reconocida."

    # --------------------------------------------------------------------------
    # Iteración (/iterate) - planificación y ejecución
    # --------------------------------------------------------------------------
    async def _generate_plan(self, goal: str, context: str) -> List[Dict]:
        model = (
            self.valves.iterative_planning_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Eres un experto desarrollador Python. El usuario quiere realizar la siguiente tarea:

{goal}

Contexto actual (archivos, funciones, etc.):
{context[:3000]}

Descompón la tarea en una secuencia de pasos concretos. Cada paso debe ser un cambio pequeño y accionable que pueda implementarse como un diff unificado (parche) sobre el código.
Devuelve un array JSON de pasos, cada paso con:
- "description": descripción corta de lo que hace el paso
- "file": la ruta del archivo a modificar (si se conoce, si no "unknown")
- "changes": un resumen breve de los cambios (se usará para generar el diff después)

Ejemplo:
[
  {{
    "description": "Añadir función calculate_average",
    "file": "utils/math.py",
    "changes": "Añadir función que toma una lista y devuelve la media"
  }}
]

El plan debe tener como máximo {self.valves.iterative_max_steps} pasos. Devuelve solo JSON, sin texto extra.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Eres un planificador preciso. Solo JSON.",
            model_override=model,
            max_tokens=1500,
            temperature=0.2,
        )
        if not response:
            return []
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            plan = json.loads(response)
            if isinstance(plan, list):
                return plan[: self.valves.iterative_max_steps]
        except:
            pass
        return []

    async def _generate_diff_for_step(self, step: Dict, project_id: str) -> str:
        model = (
            self.valves.iterative_execution_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        state = self._get_state(project_id)
        active_code = "\n".join(
            [
                b.content
                for b in state["active_blocks"].values()
                if b.content_type == ContentType.BASE_CODE
            ]
        )
        prompt = f"""Genera un diff unificado (parche) para implementar un cambio específico.

Archivo a modificar: {step.get('file', 'unknown')}
Descripción del cambio: {step.get('changes', '')}
Objetivo: {step.get('description', '')}

Código relevante actual (puede incluir varios archivos):
{active_code[:3000]}

Genera un diff unificado que aplique este cambio. Usa el formato:

```diff
--- a/file.py
+++ b/file.py
@@ -line,old +line,new @@
 -línea antigua
 +línea nueva
 ```
 Si el archivo no existe en el contexto proporcionado, supón que es un archivo nuevo y crea un diff desde vacío (ej. usa /dev/null como original).
Devuelve solo el diff, encerrado entre `diff ...` .
"""

        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Generas diffs unificados correctos. Solo el diff con formato.",
            model_override=model,
            max_tokens=2000,
            temperature=0.2,
        )
        if not response:
            return f"# Fallo al generar diff para el paso: {step.get('description')}"
        diff_match = re.search(r"```diff\n(.*?)\n```", response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        if response.strip().startswith("--- "):
            return response.strip()
        return response

    async def _start_new_iteration(
        self, project_id: str, goal: str, auto_continue: bool
    ) -> bool:
        state = self._get_state(project_id)
        active_ctx = self._get_active_code_context(project_id)
        facts_ctx = self._get_facts_context(project_id)
        context = f"Código activo:\n{active_ctx}\n\nHechos:\n{facts_ctx}"
        plan = await self._generate_plan(goal, context)
        if not plan:
            return False
        state["iterative_state"] = {
            "goal": goal,
            "plan": plan,
            "current_step": 0,
            "results": [],
            "auto_continue": auto_continue,
            "created_at": time.time(),
        }
        self._set_state(project_id, state)
        self._log_debug(
            f"Iniciada iteración para objetivo: {goal} con {len(plan)} pasos"
        )
        return True

    async def _execute_next_step(self, project_id: str) -> str:
        state = self._get_state(project_id)
        iter_state = state.get("iterative_state")
        if not iter_state:
            return "No hay ninguna iteración en curso. Usa `/iterate <objetivo>` para iniciar una tarea."
        plan = iter_state["plan"]
        step_idx = iter_state["current_step"]
        if step_idx >= len(plan):
            state["iterative_state"] = None
            self._set_state(project_id, state)
            return f"✅ **Todos los pasos completados!**\n\nObjetivo: {iter_state['goal']}\nYa puedes aplicar los diffs manualmente. Usa `/iterate` para iniciar una nueva tarea."

        step = plan[step_idx]
        pre_reflection = f"**Paso {step_idx+1}/{len(plan)}: {step.get('description')}**\n- Archivo: {step.get('file', 'unknown')}\n- Cambios: {step.get('changes', '')}"
        diff = await self._generate_diff_for_step(step, project_id)
        post_reflection = f"\n**Diff generado**\n```diff\n{diff[:1000]}\n```"
        full_output = pre_reflection + post_reflection

        iter_state["results"].append((step_idx, diff, full_output))
        iter_state["current_step"] = step_idx + 1
        self._set_state(project_id, state)

        if iter_state.get("auto_continue"):
            next_output = (
                await self._execute_next_step(project_id)
                if iter_state["current_step"] < len(plan)
                else ""
            )
            return full_output + "\n\n" + next_output
        else:
            return (
                full_output
                + "\n\n---\nResponde con **next** / **siguiente** para continuar."
            )

    async def _run_iteration(self, project_id: str, command: str) -> Tuple[str, bool]:
        if not self.valves.enable_iterative_mode:
            return "", False
        state = self._get_state(project_id)
        current_iter = state.get("iterative_state")
        if command.startswith("/iterate"):
            parts = command.split(maxsplit=1)
            if len(parts) == 1:
                return (
                    "**Comandos iterativos:**\n- `/iterate <objetivo>` – iniciar un plan de varios pasos.\n- `/iterate --auto <objetivo>` – ejecutar todos los pasos automáticamente.\n- `/iterate resume` – reanudar una iteración interrumpida.\n- También puedes usar lenguaje natural: 'implementa todas las características paso a paso'.",
                    True,
                )
            action = parts[1].lower()
            if action == "resume":
                if not current_iter:
                    return (
                        "No hay iteración en curso. Inicia una con `/iterate <objetivo>`.",
                        True,
                    )
                return await self._execute_next_step(project_id), True
            elif action.startswith("--auto"):
                goal = " ".join(parts[1:]).replace("--auto", "").strip()
                if not goal:
                    return "Debes especificar un objetivo para la iteración.", True
                if await self._start_new_iteration(
                    project_id, goal, auto_continue=True
                ):
                    return await self._execute_next_step(project_id), True
                return "Fallo al iniciar la iteración.", True
            else:
                goal = parts[1]
                if await self._start_new_iteration(
                    project_id, goal, auto_continue=False
                ):
                    return await self._execute_next_step(project_id), True
                return "Fallo al iniciar la iteración.", True
        # Lenguaje natural para continuar
        if current_iter and command.lower() in [
            "siguiente",
            "continue",
            "next",
            "yes",
            "si",
            "ok",
            "aplicar",
        ]:
            return await self._execute_next_step(project_id), True
        # Lenguaje natural para iniciar iteración (detección simple)
        if re.search(
            r"\b(implementa|haz|ejecuta) (todas|paso a paso|iterativamente)\b",
            command,
            re.IGNORECASE,
        ):
            goal = command
            if await self._start_new_iteration(project_id, goal, auto_continue=False):
                return await self._execute_next_step(project_id), True
        return "", False

    # --------------------------------------------------------------------------
    # LTM retrieval (para selección inteligente y contexto)
    # --------------------------------------------------------------------------
    async def _retrieve_historical_messages(
        self, query: str, project_id: str, limit: int
    ) -> List[Dict]:
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return []
        try:
            q_emb = self.embedder.encode(query[:1000]).tolist()
            where_filter = {
                "project_id": project_id,
                "is_hierarchical_summary": {"$ne": True},
            }
            if self.valves.long_term_memory_expiration_days > 0:
                where_filter["expires_at"] = {"$gt": datetime.utcnow().isoformat()}
            results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            messages = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    role = meta.get("role", "user")
                    messages.append({"role": role, "content": doc})
            summary_filter = {"project_id": project_id, "is_hierarchical_summary": True}
            summary_results = self.memory_collection.query(
                query_embeddings=[q_emb],
                n_results=limit // 2,
                where=summary_filter,
                include=["documents", "metadatas"],
            )
            if summary_results and summary_results["documents"]:
                for i, doc in enumerate(summary_results["documents"][0]):
                    meta = summary_results["metadatas"][0][i]
                    role = meta.get("role", "assistant")
                    messages.append({"role": role, "content": doc})
            if (
                self.valves.enable_reranking
                and self._cross_encoder
                and len(messages) > 1
            ):
                docs = [m["content"] for m in messages]
                reranked = await self._rerank_results(query, docs, limit)
                doc_to_msg = {m["content"]: m for m in messages}
                messages = [doc_to_msg[doc] for doc in reranked if doc in doc_to_msg]
            return messages[:limit]
        except Exception as e:
            logger.warning(f"Fallo recuperación de mensajes históricos: {e}")
            return []

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
            logger.warning(f"Fallo en recuperación de memorias: {e}")
            return []

    # --------------------------------------------------------------------------
    # Almacenamiento en LTM
    # --------------------------------------------------------------------------
    def _store_message_in_memory(self, message: dict, project_id: str):
        if not HAS_SENTENCE or not HAS_CHROMA or self.memory_collection is None:
            return
        content = message.get("content", "")
        if not content or len(content.strip()) < 15:
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        extracted = loop.run_until_complete(self._extract_code_blocks(content))
        loop.close()
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
        self._log_debug(f"Mensaje almacenado en LTM: {msg_id}")

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
            prompt = f"Resume el siguiente segmento de conversación, manteniendo las decisiones técnicas clave y los cambios de código:\n\n{texts[:3000]}"
            summary = await self._call_llm(
                prompt=prompt,
                system_prompt="Eres un asistente que produce resúmenes concisos y densos en información.",
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
                    f"Comprimidos {len(to_compress)} mensajes en resumen para {project_id}"
                )
        except Exception as e:
            logger.warning(f"Fallo en compresión LTM: {e}")

    # --------------------------------------------------------------------------
    # Feedback
    # --------------------------------------------------------------------------
    async def _parse_feedback_intent(self, user_message: str) -> Optional[Dict]:
        if not self.valves.enable_feedback_tracking:
            return None
        model = (
            self.valves.natural_language_forget_model
            or self.valves.llm_model
            or self.valves.summarization_model
        )
        prompt = f"""Eres un intérprete de retroalimentación. El usuario está dando feedback sobre un cambio aplicado.
Resultados posibles:
- success: el cambio funcionó, problema resuelto
- failure: el cambio no funcionó, el error persiste
- neutral: no está claro o no es feedback.

Mensaje: "{user_message}"

Si es feedback, genera JSON: {{"action": "feedback", "outcome": "success/failure", "comment": "..."}}
Si no, {{"action": "none"}}

Solo JSON.
"""
        response = await self._call_llm(
            prompt=prompt,
            system_prompt="Solo JSON.",
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
                self._log_debug(f"Feedback interpretado: {data}")
                return data
        except:
            pass
        return None

    async def _record_feedback(self, project_id: str, outcome: str, comment: str):
        state = self._get_state(project_id)
        if not state or not state["committed_changes"]:
            self._log_debug("No hay cambios aplicados con los que asociar el feedback.")
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
                f"Importancia reducida para cambio fallido {last_commit.hash} a {last_commit.importance_score:.1f}"
            )
        else:
            last_commit.importance_score = min(10.0, last_commit.importance_score + 1.0)
        self._set_state(project_id, state)
        self._log_debug(
            f"Feedback registrado para cambio {last_commit.hash}: {outcome}"
        )

    def _get_feedback_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["feedback_history"]:
            return ""
        lines = ["## Retroalimentación sobre cambios recientes\n"]
        for fb in state["feedback_history"][-5:]:
            status = "✅ ÉXITO" if fb.success else "❌ FALLO"
            desc = fb.change_description.replace("\n", " ")[:100]
            lines.append(f'- {status}: `{desc}` - Usuario: "{fb.user_comment}"')
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Token estimation para mensajes
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
    # Contexto de código activo (excluyendo obsoletos)
    # --------------------------------------------------------------------------
    def _get_active_code_context(self, project_id: str) -> str:
        state = self._get_state(project_id)
        if not state or not state["active_blocks"]:
            return ""
        now = time.time()
        active = []
        for block in state["active_blocks"].values():
            if block.obsolete:
                continue
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

        parts = ["## Contexto de código activo (por importancia)\n"]
        if base_codes:
            parts.append("### Código base (trabajo actual):")
            for b in base_codes:
                loc = (
                    f" (archivo: {b.file_path}"
                    + (
                        f", líneas {b.line_range[0]}-{b.line_range[1]}"
                        if b.line_range
                        else ""
                    )
                    + ")"
                    if b.file_path
                    else ""
                )
                pin = " [ANCLADO]" if b.pinned else ""
                aff = (
                    " [AFECTADO POR CAMBIO EN DEPENDENCIA]"
                    if b.potentially_affected
                    else ""
                )
                parts.append(
                    f"```\n{b.content[:600]}\n```{loc}  (importancia: {b.importance_score:.1f}){aff}{pin}"
                )
        if proposed:
            parts.append("### Cambios propuestos (pendientes de revisión):")
            for b in proposed:
                parts.append(f"```diff\n{b.content[:500]}\n```")
        if committed:
            parts.append("### Cambios aplicados recientemente:")
            for b in committed:
                parts.append(f"```\n{b.content[:300]}\n```")
        if errors:
            parts.append("### Errores recientes:")
            for b in errors:
                parts.append(f"```\n{b.content[:500]}\n```")
        return "\n".join(parts)

    # --------------------------------------------------------------------------
    # Actualización del código activo
    # --------------------------------------------------------------------------
    def _update_active_code(self, message: dict, project_id: str):
        if not self.valves.enable_code_awareness:
            return
        state = self._get_state(project_id)
        if self.valves.auto_remove_duplicate_blocks:
            self._remove_duplicate_blocks(state)

        asyncio.create_task(self._summarize_inactive_blocks_safely(project_id))
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

        # Extracción asíncrona de bloques
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        extracted = loop.run_until_complete(self._extract_code_blocks(content))
        loop.close()
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
                pinned=False,
                obsolete=False,
            )

            if "[KEEP]" in content or "#important" in content.lower():
                new_block.importance_score = 10.0
                new_block.pinned = True
                self._log_debug(
                    f"Marcador manual de importancia detectado para bloque {new_block.hash}, anclado automáticamente"
                )

            is_dup, existing = self._is_duplicate_code(
                new_block, list(state["active_blocks"].values())
            )
            if is_dup and existing:
                if self.valves.prioritize_recent_code:
                    existing.content = new_block.content
                    existing.hash = new_block.hash
                    if new_block.file_path:
                        existing.file_path = new_block.file_path
                    existing.line_range = new_block.line_range
                    existing.timestamp = time.time()
                    existing.mention_count += 1
                    existing.last_mentioned = time.time()
                    existing._update_importance()
                    self._log_debug(f"Actualizado bloque existente {existing.hash}")
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
                    f"Cambio propuesto {new_block.hash} marcado como conflictivo"
                )

            state["active_blocks"][new_block.hash] = new_block
            self._log_debug(f"Nuevo bloque {content_type.value}: {new_block.hash}")

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
                        f"Auto-aplicación omitida para cambio propuesto conflictivo {new_block.hash}"
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

            if self.valves.enable_dependency_tracking and content_type in (
                ContentType.BASE_CODE,
                ContentType.PROPOSED_CHANGE,
                ContentType.COMMITTED_CHANGE,
            ):
                asyncio.create_task(
                    self._refresh_dependencies_for_block(new_block.hash, project_id)
                )

        # Aprendizaje del asistente: actualizar mejor bloque base coincidente
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
                        f"Asistente actualizó bloque base {best_base.hash} (sim={best_sim:.2f})"
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

        if self.valves.auto_remove_duplicate_blocks:
            self._remove_duplicate_blocks(state)

        if self.valves.hierarchical_compression_enabled:
            if (
                state["message_count"]
                % self.valves.hierarchical_compression_interval_messages
                == 0
            ):
                asyncio.create_task(self._hierarchical_compress(project_id, state))

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
        if to_remove:
            for h in to_remove:
                if h in state["active_blocks"]:
                    self._log_debug(
                        f"Eliminado bloque duplicado {h} (importancia: {state['active_blocks'][h].importance_score:.1f})"
                    )
                    del state["active_blocks"][h]
            state["recent_changes"] = [
                b for b in state["recent_changes"] if b.hash not in to_remove
            ]
            state["committed_changes"] = [
                b for b in state["committed_changes"] if b.hash not in to_remove
            ]

    # --------------------------------------------------------------------------
    # Inlet (punto de entrada principal)
    # --------------------------------------------------------------------------
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self._log_debug("inlet llamado")
        messages = body.get("messages", [])
        project_id = self._get_project_id()
        self._log_debug(f"ID del proyecto: {project_id}")
        if not messages:
            return body

        state = self._get_state(project_id)

        # 1. Comandos de olvido
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

        # 2. Comandos de recordar (anclar)
        if self.valves.enable_natural_language_forget:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
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
                    new_messages = messages + [
                        {"role": "assistant", "content": confirmation}
                    ]
                    body["messages"] = new_messages
                    return body

        # 3. Comandos de obsoleto
        if self.valves.enable_obsolete_marking:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg and (
                last_user_msg.get("content", "").startswith("/obsolete")
                or re.search(
                    r"\b(marca|obsoleto|ya no sirve|revive)\b",
                    last_user_msg.get("content", ""),
                    re.IGNORECASE,
                )
            ):
                intent = await self._parse_obsolete_intent(last_user_msg["content"])
                if intent and intent.get("action") != "none":
                    confirmation = await self._execute_obsolete_intent(
                        project_id, intent
                    )
                    messages.pop()
                    messages.append({"role": "assistant", "content": confirmation})
                    body["messages"] = messages
                    return body

        # 4. Razonamiento paso a paso (/think)
        if self.valves.enable_cot_on_demand:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
                cot_question = await self._parse_cot_intent(
                    last_user_msg.get("content", "")
                )
                if cot_question:
                    active_ctx = self._get_active_code_context(project_id)
                    facts_ctx = self._get_facts_context(project_id)
                    context = f"Código activo:\n{active_ctx}\n\nHechos:\n{facts_ctx}"
                    reasoning = await self._generate_cot(cot_question, context)
                    messages.pop()
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Razonamiento paso a paso**\n{reasoning}",
                        }
                    )
                    body["messages"] = messages
                    return body

        # 5. Extracción de supuestos (/assume)
        if self.valves.enable_assumption_extraction:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
                assumption_target = await self._parse_assumption_intent(
                    last_user_msg.get("content", "")
                )
                if assumption_target:
                    analysis = await self._extract_assumptions(assumption_target)
                    messages.pop()
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Análisis de supuestos**\n{analysis}",
                        }
                    )
                    body["messages"] = messages
                    return body

        # 6. Iteración (/iterate) - natural language y /iterate
        if self.valves.enable_iterative_mode:
            last_user_msg = (
                messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None
            )
            if last_user_msg:
                result, consumed = await self._run_iteration(
                    project_id, last_user_msg.get("content", "")
                )
                if consumed:
                    messages.pop()
                    messages.append({"role": "assistant", "content": result})
                    body["messages"] = messages
                    return body

        # 7. Selección inteligente de contexto (si está activada)
        if self.valves.smart_context_selection and len(messages) > 0:
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                query = messages[last_user_idx].get("content", "")
                if query:
                    top_k = self.valves.smart_context_top_k
                    historical = await self._retrieve_historical_messages(
                        query, project_id, top_k
                    )
                    new_history = []
                    if self.valves.smart_context_include_last_user:
                        new_history.append(messages[last_user_idx])
                        if (
                            last_user_idx + 1 < len(messages)
                            and messages[last_user_idx + 1].get("role") == "assistant"
                        ):
                            new_history.append(messages[last_user_idx + 1])
                    for msg in historical:
                        if msg["content"] != query:
                            new_history.append(msg)
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    messages = system_msgs + new_history
                    body["messages"] = messages
                    self._log_debug(
                        f"Selección inteligente reemplazó historial con {len(new_history)} mensajes"
                    )

        # 8. Detección de contradicciones
        if (
            self.valves.enable_contradiction_detection
            and self.valves.contradiction_inject_warning
        ):
            contradiction_warning = await self._detect_contradictions(messages)
            if contradiction_warning:
                messages.insert(0, {"role": "system", "content": contradiction_warning})
                body["messages"] = messages
                self._log_debug("Advertencia de contradicción inyectada")

        # 9. Actualizar código activo desde mensajes recientes
        if self.valves.enable_code_awareness:
            for msg in messages[-5:]:
                self._update_active_code(msg, project_id)

        # 10. Inyectar contexto LTM (si no se usó smart context)
        if not self.valves.smart_context_selection:
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
                    ctx = "## Contexto relevante del pasado\n\n" + "\n---\n".join(
                        all_mem
                    )
                    sys_msgs = [m for m in messages if m.get("role") == "system"]
                    if sys_msgs:
                        sys_msgs[0]["content"] = ctx + "\n\n" + sys_msgs[0]["content"]
                    else:
                        messages.insert(0, {"role": "system", "content": ctx})
                    body["messages"] = messages

        # 11. Inyectar contexto de código activo
        active_ctx = self._get_active_code_context(project_id)
        if active_ctx:
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            if sys_msgs:
                sys_msgs[0]["content"] = active_ctx + "\n\n" + sys_msgs[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": active_ctx})
            body["messages"] = messages

        # 12. Inyectar hechos
        if self.valves.enable_facts and self.valves.inject_facts_in_context:
            facts_ctx = self._get_facts_context(project_id)
            if facts_ctx:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_msgs[0]["content"] = facts_ctx + "\n\n" + sys_msgs[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": facts_ctx})
                body["messages"] = messages

        # 13. Inyectar instrucción de confianza
        if self.valves.enable_confidence_scoring:
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            if sys_msgs:
                sys_msgs[0]["content"] += self.valves.confidence_prompt
            else:
                messages.insert(
                    0, {"role": "system", "content": self.valves.confidence_prompt}
                )
                body["messages"] = messages

        # 14. Inyectar retroalimentación
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

        # 15. Advertencia proactiva de contexto y sugerencias de comandos
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") != "system"]
        total_tokens = self._estimate_tokens(system_msgs + history_msgs)
        max_tokens = self.valves.context_window_tokens
        if max_tokens > 0:
            suggestion = await self._check_and_suggest_summarization(
                project_id, total_tokens, max_tokens
            )
            if suggestion:
                messages.insert(0, {"role": "system", "content": suggestion})
                body["messages"] = messages

        cmd_suggestion = await self._suggest_commands(project_id, state)
        if cmd_suggestion:
            messages.insert(0, {"role": "system", "content": cmd_suggestion})
            body["messages"] = messages

        # 16. Detección de preguntas repetidas
        if self.valves.duplicate_question_threshold and HAS_SENTENCE:
            last_user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user_msg:
                duplicate = await self._find_duplicate_question(
                    last_user_msg.get("content", ""), project_id
                )
                if duplicate:
                    warn_msg = f"⚠️ **Nota**: Esta pregunta es muy similar a una que hiciste antes (similitud {duplicate['sim']:.2f}). La respuesta anterior podría seguir siendo relevante. Si no es así, por favor aclara qué ha cambiado."
                    messages.insert(0, {"role": "system", "content": warn_msg})
                    body["messages"] = messages

        # 17. Recorte adaptativo (adaptative trim) si es necesario
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
                            "content": f"[Resumen de conversación anterior]\n{summary}",
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
        self._log_debug("outlet llamado")
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
