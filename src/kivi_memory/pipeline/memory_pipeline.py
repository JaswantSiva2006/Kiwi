"""Production write pipeline through retrieval, reconciliation, and mutation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from kivi_memory.common.config import (
    PIPELINE_FINAL_TOP_K,
    PIPELINE_SEMANTIC_COMPILER_MODEL,
    PIPELINE_SEMANTIC_COMPILER_THINK,
    KiviCompilerConfig,
    load_config_from_env,
)
from kivi_memory.common.schemas import CandidateSemanticAssertion, MemoryEpisode, SemanticArgument
from kivi_memory.embeddings import retrieve_vector_candidates
from kivi_memory.enrichment.temporal import TemporalNormalizer
from kivi_memory.enrichment.temporal_gate import TemporalRoute, TemporalRoutingGate, route_temporal_assertions
from kivi_memory.enrichment.validator import validate_assertions
from kivi_memory.entity_resolution import resolve_entity_mention, resolve_or_create_entity
from kivi_memory.ledger import execute_reconciliation
from kivi_memory.reconciliation.config import RECONCILIATION_MODEL
from kivi_memory.reconciliation import reconcile_memory
from kivi_memory.retrieval import merge_and_rerank_candidates, retrieve_graph_candidates, retrieve_structured_candidates
from kivi_memory.semantic_compiler.compiler import SemanticCompiler

SEMANTIC_COMPILER_MODEL = PIPELINE_SEMANTIC_COMPILER_MODEL
SEMANTIC_COMPILER_THINK = PIPELINE_SEMANTIC_COMPILER_THINK
FINAL_TOP_K = PIPELINE_FINAL_TOP_K


@dataclass(frozen=True)
class MemoryPipelineResult:
    episode_id: str
    output: dict[str, Any]


@dataclass(frozen=True)
class RetrievalStageResult:
    vector_result: Any
    structured_result: Any
    graph_result: Any
    final_candidates: list[Any]


class MemoryPipeline:
    """Complete v1 path from episode to canonical ledger mutation."""

    def __init__(
        self,
        config: KiviCompilerConfig | None = None,
        compiler: SemanticCompiler | None = None,
        temporal_normalizer: TemporalNormalizer | None = None,
        temporal_gate: TemporalRoutingGate | None = None,
    ) -> None:
        base_config = config or load_config_from_env()
        self.config = KiviCompilerConfig(
            ollama_base_url=base_config.ollama_base_url,
            model=SEMANTIC_COMPILER_MODEL,
            temporal_model=base_config.temporal_model,
            temporal_reasoning_model=base_config.temporal_reasoning_model,
            temporal_normalization_model=base_config.temporal_normalization_model,
            temperature=base_config.temperature,
            max_attempts=base_config.max_attempts,
            timeout_seconds=base_config.timeout_seconds,
            think=SEMANTIC_COMPILER_THINK,
        )
        self.compiler = compiler or SemanticCompiler(config=self.config)
        self.temporal_normalizer = temporal_normalizer or TemporalNormalizer(config=self.config)
        self.temporal_gate = temporal_gate or TemporalRoutingGate()

    def process(self, episode: MemoryEpisode) -> MemoryPipelineResult:
        total_started = time.perf_counter()
        compiler_result = self.compiler.compile(episode)
        validated = validate_assertions(compiler_result.output.assertions, episode)
        valid_results = [result for result in validated if result.validation_report.valid]
        temporal_gate_result = route_temporal_assertions(
            [result.assertion for result in valid_results],
            gate=self.temporal_gate,
        )
        temporal_metrics = _initial_temporal_metrics(temporal_gate_result)
        route_ids_by_index = {
            id(result): f"A{position + 1}"
            for position, result in enumerate(valid_results)
        }
        assertion_outputs = []

        for index, result in enumerate(validated):
            assertion_started = time.perf_counter()
            assertion = result.assertion
            item = _initial_assertion_item(index, assertion, result.validation_report)
            if not result.validation_report.valid:
                item["latency"]["total_ms"] = _elapsed_ms(assertion_started)
                assertion_outputs.append(item)
                continue

            try:
                temporal_started = time.perf_counter()
                route_id = route_ids_by_index[id(result)]
                route = temporal_gate_result.routes.get(route_id, TemporalRoute.BOTH)
                item["temporal_route"] = route.value
                temporal_result = self._run_temporal_route(assertion, episode, route, temporal_metrics)
                temporal_metadata = temporal_result.temporal_metadata
                item["temporal_metadata"] = _to_jsonable(temporal_metadata)
                item["latency"]["temporal_ms"] = _elapsed_ms(temporal_started)

                entity_started = time.perf_counter()
                entity_resolution = _resolve_entities(assertion, episode.episode_id)
                entity_resolution_payload = _to_jsonable(entity_resolution)
                item["entity_resolution"] = entity_resolution_payload
                item["latency"]["entity_resolution_ms"] = _elapsed_ms(entity_started)

                retrieval = self._run_retrieval(assertion, entity_resolution_payload, item)

                reconcile_started = time.perf_counter()
                wrapped_assertion = {
                    "semantic_assertion": _to_jsonable(assertion),
                    "validation_report": _to_jsonable(result.validation_report),
                    "temporal_metadata": _to_jsonable(temporal_metadata),
                    "entity_resolution": entity_resolution_payload,
                }
                decision = reconcile_memory(wrapped_assertion, retrieval.final_candidates)
                item["reconciliation_decision"] = _to_jsonable(decision)
                item["latency"]["reconciliation_ms"] = _elapsed_ms(reconcile_started)

                mutation_started = time.perf_counter()
                mutation = execute_reconciliation(
                    decision,
                    wrapped_assertion,
                    retrieval.final_candidates,
                    temporal_metadata,
                    entity_resolution_payload,
                    episode,
                    validation_report=result.validation_report,
                )
                item["mutation_result"] = _to_jsonable(mutation)
                item["latency"]["mutation_ms"] = _elapsed_ms(mutation_started)
            except Exception as exc:
                item["errors"].append({"stage": "memory_pipeline", "error": str(exc)})

            item["latency"]["total_ms"] = _elapsed_ms(assertion_started)
            assertion_outputs.append(item)

        output = {
            "episode_id": episode.episode_id,
            "models": {
                "semantic_compiler": compiler_result.model_name,
                "semantic_compiler_think": SEMANTIC_COMPILER_THINK,
                "temporal_reasoning": self.config.temporal_reasoning_model,
                "temporal_normalization": self.config.temporal_normalization_model,
                "reconciliation": RECONCILIATION_MODEL,
                "sensitivity": None,
            },
            "pipeline_metadata": {
                "compiler_attempt_count": compiler_result.attempt_count,
                "compiler_inference_duration_ms": compiler_result.inference_duration_ms,
                "assertion_count": len(compiler_result.output.assertions),
                "temporal": temporal_metrics,
                "total_ms": _elapsed_ms(total_started),
            },
            "assertions": assertion_outputs,
        }
        return MemoryPipelineResult(episode_id=episode.episode_id, output=_to_jsonable(output))

    def _run_temporal_route(
        self,
        assertion: CandidateSemanticAssertion,
        episode: MemoryEpisode,
        route: TemporalRoute,
        temporal_metrics: dict[str, Any],
    ):
        route_key = f"temporal_route_{route.value.lower()}_count"
        temporal_metrics[route_key] += 1
        started = time.perf_counter()
        if route == TemporalRoute.NONE:
            result = self.temporal_normalizer.normalize_none(assertion)
        elif route == TemporalRoute.REASON:
            result = self.temporal_normalizer.normalize_reason_only(assertion, episode)
            temporal_metrics["temporal_reasoning_call_count"] += 1
            temporal_metrics["temporal_reasoning_ms"] += result.reasoning_duration_ms
        else:
            result = self.temporal_normalizer.normalize_both(assertion, episode)
            temporal_metrics["temporal_reasoning_call_count"] += 1
            temporal_metrics["temporal_normalization_call_count"] += 1
            temporal_metrics["temporal_reasoning_ms"] += result.reasoning_duration_ms
            temporal_metrics["temporal_normalization_ms"] += result.normalization_duration_ms
        temporal_metrics["total_temporal_ms"] += _elapsed_ms(started)
        return result

    def _run_retrieval(
        self,
        assertion: CandidateSemanticAssertion,
        entity_resolution_payload: dict[str, Any],
        item: dict[str, Any],
    ) -> RetrievalStageResult:
        """Run vector, structured, and graph retrieval before deterministic reranking."""

        vector_result = retrieve_vector_candidates(assertion.canonical_text, top_k=20)
        item["vector_candidates"] = _to_jsonable(vector_result.candidates)
        item["latency"]["vector_retrieval_ms"] = vector_result.total_ms
        item["vector_retrieval"] = _vector_metrics(vector_result)

        structured_result = retrieve_structured_candidates(assertion, entity_resolution_payload, top_k=20)
        item["structured_candidates"] = _to_jsonable(structured_result.candidates)
        item["latency"]["structured_retrieval_ms"] = structured_result.total_ms

        graph_result = retrieve_graph_candidates(assertion, entity_resolution_payload)
        item["graph_candidates"] = _to_jsonable(graph_result.candidates)
        item["latency"]["graph_retrieval_ms"] = graph_result.graph_retrieval_ms
        item["graph_retrieval"] = _graph_metrics(graph_result)

        rerank_started = time.perf_counter()
        final_candidates = merge_and_rerank_candidates(
            vector_result.candidates,
            structured_result.candidates,
            graph_result.candidates,
            final_top_k=FINAL_TOP_K,
        )
        item["final_candidates"] = _to_jsonable(final_candidates)
        item["latency"]["rerank_ms"] = _elapsed_ms(rerank_started)
        return RetrievalStageResult(vector_result, structured_result, graph_result, final_candidates)


def _initial_temporal_metrics(temporal_gate_result) -> dict[str, Any]:
    return {
        "temporal_gate_ms": temporal_gate_result.temporal_gate_ms,
        "gate_prompt_eval_count": temporal_gate_result.gate_prompt_eval_count,
        "gate_eval_count": temporal_gate_result.gate_eval_count,
        "gate_model_call_count": temporal_gate_result.gate_model_call_count,
        "temporal_route_none_count": 0,
        "temporal_route_reason_count": 0,
        "temporal_route_both_count": 0,
        "temporal_reasoning_ms": 0.0,
        "temporal_reasoning_call_count": 0,
        "temporal_normalization_ms": 0.0,
        "temporal_normalization_call_count": 0,
        "total_temporal_ms": temporal_gate_result.temporal_gate_ms,
        "fallback_used": temporal_gate_result.fallback_used,
        "fallback_error": temporal_gate_result.error,
    }


def _initial_assertion_item(index: int, assertion: CandidateSemanticAssertion, validation_report: Any) -> dict[str, Any]:
    return {
        "index": index,
        "semantic_assertion": _to_jsonable(assertion),
        "validation_report": _to_jsonable(validation_report),
        "temporal_metadata": None,
        "temporal_route": None,
        "sensitivity_metadata": None,
        "entity_resolution": {"subject": None, "semantic_arguments": []},
        "vector_candidates": [],
        "structured_candidates": [],
        "graph_candidates": [],
        "final_candidates": [],
        "reconciliation_decision": None,
        "mutation_result": None,
        "latency": {},
        "errors": [],
    }


def _vector_metrics(vector_result: Any) -> dict[str, Any]:
    return {
        "vector_search_mode": vector_result.vector_search_mode,
        "vector_embedding_ms": vector_result.embedding_ms,
        "vector_db_search_ms": vector_result.db_search_ms,
        "vector_total_ms": vector_result.total_ms,
        "vector_top_k": vector_result.vector_top_k,
        "vector_returned_count": vector_result.vector_returned_count,
        "hnsw_ef_search": vector_result.hnsw_ef_search,
        "hnsw_iterative_scan": vector_result.hnsw_iterative_scan,
        "hnsw_fallback_used": vector_result.hnsw_fallback_used,
        "hnsw_fallback_reason": vector_result.hnsw_fallback_reason,
    }


def _graph_metrics(graph_result: Any) -> dict[str, Any]:
    return {
        "graph_retrieval_ms": graph_result.graph_retrieval_ms,
        "graph_seed_count": graph_result.graph_seed_count,
        "graph_direct_count": graph_result.graph_direct_count,
        "graph_expanded_count": graph_result.graph_expanded_count,
        "graph_returned_count": graph_result.graph_returned_count,
    }


def _resolve_entities(assertion: CandidateSemanticAssertion, episode_id: str) -> dict[str, Any]:
    subject = resolve_or_create_entity(assertion.subject.text, assertion.subject.entity_type, episode_id)
    semantic_arguments = []
    for argument in assertion.semantic_arguments:
        if not argument.is_entity:
            continue
        semantic_arguments.append(
            {
                "role": argument.role,
                "result": _resolve_argument(argument, episode_id),
            }
        )
    return {"subject": subject, "semantic_arguments": semantic_arguments}


def _resolve_argument(argument: SemanticArgument, episode_id: str):
    if argument.entity_type is None:
        return resolve_entity_mention(argument.text, None)
    return resolve_or_create_entity(argument.text, argument.entity_type, episode_id)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
