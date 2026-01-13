# scripts/run_indexer.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.decision_graph.config import AppConfig, LLMConfig, load_config
from src.decision_graph.indexer import DecisionRelatedIndexer
from src.decision_graph.models import LLMExtractedDecision
from src.decision_graph.neo4j_store import create_neo4j_vector_store

from src.decision_graph.agenda_chain import build_nn_agenda_cls_chain, build_nn_agenda_cls_gpt5_chain

load_dotenv()


def load_decisions_json(path: str) -> list[dict]:
    """
    decisions JSON 파일을 로드합니다.

    기대 포맷:
      - JSON 최상위는 list[dict]
      - 각 dict는 meeting_id/date/title/summary 등 필요한 필드를 포함

    Raises:
      - FileNotFoundError: 파일이 없을 때
      - ValueError: JSON 포맷이 기대와 다를 때
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"decisions_json_path not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("decisions.json must be a list of objects (list[dict])")

    return data


def build_agenda_chain(cfg: LLMConfig):
    """
    YAML 설정(cfg.llm)에 따라 agenda classification chain을 생성합니다.
    """
    if cfg.provider == "openai":
        if cfg.reasoning_effort is not None:
            return build_nn_agenda_cls_gpt5_chain(
                model_name=cfg.model,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
            )
        return build_nn_agenda_cls_chain(
            model_name=cfg.model,
            temperature=cfg.temperature,
        )

    if cfg.provider == "gemini":
        return build_nn_agenda_cls_chain(
            model_name=cfg.model,
            temperature=cfg.temperature,
        )

    raise ValueError(f"Unsupported llm.provider: {cfg.provider}")


def _parse_date_maybe(value: Any) -> Any:
    """
    Pydantic이 date 파싱을 처리할 수도 있지만,
    안전하게 'YYYY-MM-DD' 문자열이면 date로 변환해줍니다.
    """
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return value
    return value


def main() -> None:
    # 1) config 로드
    cfg: AppConfig = load_config("./config/config.yaml")

    # 2) Vector store 생성
    related_store = create_neo4j_vector_store(
        neo4j_url=cfg.neo4j.url,
        neo4j_user=cfg.neo4j.username,
        neo4j_password=cfg.neo4j.password,  # YAML에 적힌 값 그대로 사용
        node_label=cfg.neo4j.node_label,
        collection_name=cfg.neo4j.collection_name,
        emb_model_name=cfg.embeddings.model_name,
        emb_device=cfg.embeddings.device,
    )

    # 3) LLM chain 생성
    agenda_cls_chain = build_agenda_chain(cfg.llm)

    # 4) Indexer 생성
    decision_related_indexer = DecisionRelatedIndexer(
        related_store=related_store,
        agenda_cls_chain=agenda_cls_chain,
        similarity_threshold=cfg.indexer.similarity_threshold,
        related_top_k=cfg.indexer.related_top_k,
        related_recent_meetings_num=cfg.indexer.related_recent_meetings_num,
        max_concurrency=cfg.indexer.max_concurrency,
        relationship_type=cfg.indexer.relationship_type,
        node_label=cfg.indexer.node_label,
    )

    # 5) decisions 로드 후 실행
    raw_decisions = load_decisions_json(cfg.decisions_json_path)

    for raw in raw_decisions:
        llm_extracted = LLMExtractedDecision(
            decision_id=raw.get("decision_id"),
            meeting_id=raw["meeting_id"],
            date=_parse_date_maybe(raw["date"]),
            title=raw["title"],
            summary=raw["summary"],
            tags=raw.get("tags", []),
            speakers=raw.get("speakers", []),
            category=raw.get("category", "decision"),
        )

        decision_related_indexer.upsert_new_llm_extracted_decision(llm_extracted)


if __name__ == "__main__":
    main()
