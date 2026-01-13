from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class LLMExtractedDecision(BaseModel):
    """
    JSON/LLM 결과 등 외부 입력 DTO
    """
    decision_id: Optional[str] = None
    meeting_id: str
    date: date
    title: str
    summary: str
    tags: List[str] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)
    category: str = "decision"


class Decision(BaseModel):
    """
    Neo4j 노드/벡터스토어 적재용 엔티티
    """
    decision_id: str
    version: int = 1

    # 신규 노드에 반영될 change_type (LLM 판정 결과)
    change_type: Literal["created", "updated", "changed", "canceled", "unchanged"] = "created"
    base_decision_id: Optional[str] = None

    meeting_id: str
    meeting_date: date

    decision_title: str
    raw_text: str
    tags: List[str] = Field(default_factory=list)

    related_decision_ids: List[str] = Field(default_factory=list)


def ensure_decision_id(raw: Dict[str, Any]) -> str:
    did = raw.get("decision_id")
    if isinstance(did, str) and did.strip():
        return did.strip()
    return f"dec:{uuid.uuid4().hex[:10]}"


def convert_extracted_to_decision(extracted: LLMExtractedDecision) -> Decision:
    """
    LLMExtractedDecision 을 Decision 로 변환
    """
    return Decision(
        decision_id=extracted.decision_id ,
        version=1,
        change_type="created", # 기본값, 이후 indexer가 LLM 결과로 갱신
        base_decision_id=None, # 이후 indexer가 LLM 결과로 갱신
        meeting_id=extracted.meeting_id,
        meeting_date=extracted.date,
        decision_title=extracted.title,
        raw_text=extracted.summary,
        tags=extracted.tags,
        related_decision_ids=[],
    )


def decision_node_id(decision: Decision) -> str:
    """
    neo4j 용 decision node id 생성 함수
    """
    return f"{decision.decision_id}:v{decision.version}"


def decision_to_page_content(decision: Decision) -> str:
    """
    decision 내용을 Document 의 page_content 변환
    """
    return f"[Title] {decision.decision_title}\n[Text] {decision.raw_text}\n"


def decision_to_document(decision: Decision) -> Document:
    """
    decision 을 Document 로 변환
    """
    node_id = decision_node_id(decision)
    metadata = {
        "id": node_id,  # Neo4j node id
        "decision_id": decision.decision_id,
        "version": decision.version,
        "change_type": decision.change_type,
        "base_decision_id": decision.base_decision_id,
        "meeting_id": decision.meeting_id,
        "meeting_date": decision.meeting_date.isoformat(),
        "decision_title": decision.decision_title,
        "raw_text": decision.raw_text,
        "tags": decision.tags,
        "related_decision_ids": decision.related_decision_ids,
    }
    return Document(
        page_content=decision_to_page_content(decision), 
        metadata=metadata
    )


def format_decision_for_prompt(doc: Document) -> str:
    """
    후보 결정사항들을 프롬프트에 넣기 위해 문자열로 변환
    """
    m = doc.metadata or {}
    return (
        f"- decision_id: {m.get('decision_id')}\n"
        f"  version: {m.get('version')}\n"
        f"  meeting_id: {m.get('meeting_id')}\n"
        f"  meeting_date: {m.get('meeting_date')}\n"
        f"  decision_title: {m.get('decision_title')}\n"
        f"  text: {doc.page_content}\n"
    )
