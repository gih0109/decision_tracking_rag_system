# src/decision_graph/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class Neo4jConfig:
    """Neo4j 접속 및 벡터 인덱스 관련 설정"""
    url: str
    username: str
    password: str  # YAML에 직접 기입
    node_label: str = "Decision"
    collection_name: str = "decision_related"


@dataclass
class EmbeddingsConfig:
    """임베딩 모델 설정"""
    model_name: str = "BAAI/bge-m3"
    device: Literal["cpu", "cuda"] = "cpu"


@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: Literal["openai", "gemini"]
    model: str
    temperature: float = 0.0
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None


@dataclass
class IndexerConfig:
    """DecisionRelatedIndexer 동작 파라미터"""
    similarity_threshold: float = 0.6
    related_top_k: int = 10
    related_recent_meetings_num: Optional[int] = None
    max_concurrency: int = 10
    relationship_type: str = "RELATED"
    node_label: str = "Decision"


@dataclass
class AppConfig:
    """전체 앱 설정"""
    decisions_json_path: str
    neo4j: Neo4jConfig
    embeddings: EmbeddingsConfig
    llm: LLMConfig
    indexer: IndexerConfig


def load_config(config_path: str = "configs/indexer.yaml") -> AppConfig:
    """
    YAML 설정 파일을 읽어서 AppConfig로 변환합니다.

    Args:
        config_path: 설정 YAML 경로 (기본값: configs/indexer.yaml)

    Returns:
        AppConfig

    Raises:
        FileNotFoundError: config_path가 존재하지 않을 때
        KeyError: 필수 키가 없을 때
        ValueError: 타입/값이 기대와 다를 때
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"- Create it from configs/indexer.example.yaml"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping (dict)")

    # 필수 키 검증
    if "decisions_json_path" not in raw:
        raise KeyError("Missing required key: decisions_json_path")
    if "neo4j" not in raw:
        raise KeyError("Missing required key: neo4j")
    if "llm" not in raw:
        raise KeyError("Missing required key: llm")

    neo4j = Neo4jConfig(**raw["neo4j"])
    embeddings = EmbeddingsConfig(**raw.get("embeddings", {}))
    llm = LLMConfig(**raw["llm"])
    indexer = IndexerConfig(**raw.get("indexer", {}))

    return AppConfig(
        decisions_json_path=raw["decisions_json_path"],
        neo4j=neo4j,
        embeddings=embeddings,
        llm=llm,
        indexer=indexer,
    )
