from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import VectorStore

from .agenda_chain import AgendaClsOutput
from .models import Decision, decision_to_document, decision_to_page_content, format_decision_for_prompt


logger = logging.getLogger(__name__)


class UnionFind:
    """
    방향 무시 연결요소를 고려한
    Union-Find(Disjoint Set Union, DSU) 자료구조.

    목적:
    - 후보 노드들이 "방향을 무시했을 때 같은 연결 요소(component)에 속하는지"를 빠르게 판별합니다.
    - indexer에서는 "컴포넌트 당 대표 후보 1개만 선택"하기 위한 용도로 사용합니다.

    주의:
    - 이 구현은 POC 성격이며, Neo4j의 실제 그래프 구조를 완전히 대체하지 않습니다.
    - indexer가 upsert한 엣지에 대해서만 메모리 상에서 컴포넌트를 유지합니다.

    """
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class CheckReachability:
    """
    reachable(u, v) = u ->* v (u에서 v로 도달 가능한가) 를 메모리에서 빠르게 확인하기 위한 POC 컴포넌트.

    목적:
    - "기존 -> 신규" 방향의 엣지를 계속 추가하는 흐름에서,
    신규 노드 v에 대해 선행(ancestor) 노드들을 누적 저장해두고 reachable을 O(1)에 가깝게 확인합니다.

    동작 가정:
    - indexer가 추가하는 관계 방향은 (기존 candidate) -> (신규)로만 들어온다고 가정합니다.
    - 이 가정이 깨지면(양방향/역방향 엣지 추가) 조상 집합 업데이트 로직이 불완전해질 수 있습니다.
    """
    def __init__(self) -> None:
        self.ancestors: Dict[str, set[str]] = {}

    def add_node(self, x: str) -> None:
        self.ancestors.setdefault(x, set())

    def add_edge(self, u: str, v: str) -> None:
        """
        u -> v 엣지가 추가되었음을 반영합니다.

        업데이트 규칙:
          - v의 모든 선행(ancestors)은 {u} ∪ ancestors[u] 입니다.
        """
        self.add_node(u)
        self.add_node(v)
        self.ancestors[v].add(u)
        self.ancestors[v].update(self.ancestors[u])

    def reachable(self, u: str, v: str) -> bool:
        """
        u가 v의 조상(선행)으로 기록되어 있는지 확인합니다.
        즉, u ->* v 인지 여부를 반환합니다.
        """
        return u in self.ancestors.get(v, set())


class DecisionRelatedIndexer:
    """
    새 Decision을 Neo4jVector + LLM 판정 결과를 이용해 그래프(노드/엣지)로 적재하는 인덱서.

    전체 흐름(핵심 메서드: upsert_new_decision):
      1) 신규 decision을 기준으로 벡터 검색을 수행하여 후보 결정사항들을 가져옵니다.
      2) LLM 체인(agenda_cls_chain)을 batch로 호출하여,
         각 후보가 같은 안건인지(is_same_agenda), 변경 유형(change_type),
         관계 점수(relation_score), 기준 decision_id(base_decision_id)를 판정합니다.
      3) same_agenda=True 후보들만 남기고,
         "연결요소(component) 당 대표 1개"를 선택합니다(Union-Find 기반).
      4) 대표들 중 가장 강한 후보(primary)를 골라,
         신규 노드의 change_type/base_decision_id를 확정하여 노드에 반영합니다.
      5) 대표 후보들에 대해 (기존 candidate) -> (신규) 방향으로 Neo4j 관계를 upsert합니다.

    Neo4j 관련 가정:
      - related_store가 Neo4jVector 또는 query(cypher, params) 호출을 지원하는 VectorStore라고 가정합니다.
      - 노드 속성에 id(=node_id)가 저장되어 있어야 관계 upsert가 정상 동작합니다.
    """
    def __init__(
        self,
        related_store: VectorStore,
        agenda_cls_chain: Runnable,
        similarity_threshold: float = 0.5,
        related_top_k: int = 10,
        related_recent_meetings_num: int | None = None,
        max_concurrency: int = 8,
        relationship_type: str = "RELATED",
        node_label: str = "Decision",
    ) -> None:
        self.related_store = related_store
        self.agenda_cls_chain = agenda_cls_chain

        self.similarity_th = similarity_threshold
        self.related_top_k = related_top_k
        self.related_recent_meetings_num = related_recent_meetings_num
        self.max_concurrency = max_concurrency

        self.relationship_type = relationship_type
        self.node_label = node_label

        self.union_find = UnionFind()
        self.reach = CheckReachability()

    def _register_node_in_memory(self, node_id: str) -> None:
        """
        (POC) 메모리 자료구조(Union-Find/Reachability)에 노드를 등록
        """
        self.union_find.add(node_id)
        self.reach.add_node(node_id)

    def _query(self, cypher: str, params: Dict[str, Any]) -> List[dict]:
        """
        Neo4jVector.query 호환 래퍼
        """
        query_fn = getattr(self.related_store, "query", None)
        if query_fn is None:
            return []
        try:
            return query_fn(cypher, params=params)
        except TypeError:
            return query_fn(cypher, params)

    def _get_previous_meeting_ids(self, *, new_meeting_date: date, n: int) -> set[str]:
        """
        신규 meeting_date 이전에 존재하는 회의들 중 "최신순 직전 n개 회의"의 meeting_id 집합을 가져오는 메서드.
        """
        if n <= 0:
            return set()

        cypher = f"""
        MATCH (d:`{self.node_label}`)
        WHERE d.meeting_date < $new_date
        RETURN DISTINCT d.meeting_id AS meeting_id, d.meeting_date AS meeting_date
        ORDER BY meeting_date DESC
        LIMIT $n
        """
        rows = self._query(cypher, {"new_date": new_meeting_date.isoformat(), "n": int(n)})
        if not rows:
            return set()

        return {r.get("meeting_id") for r in rows if isinstance(r, dict) and r.get("meeting_id")}

    def _search_candidates(self, new_decision: Decision) -> List[Tuple[Document, float]]:
        """
        벡터 검색을 통해 new_decision과 유사한 후보 결정사항들을 찾고 필터링
        """
        query = decision_to_page_content(new_decision)
        raw_results = self.related_store.similarity_search_with_score(
            query=query,
            k=self.related_top_k,
        )

        # 같은 회의 제외
        filtered = [(doc, score) for doc, score in raw_results if doc.metadata.get("meeting_id") != new_decision.meeting_id]

        # 직전 N개 회의만 남기기
        if self.related_recent_meetings_num is not None:
            prev_ids = self._get_previous_meeting_ids(new_meeting_date=new_decision.meeting_date, n=int(self.related_recent_meetings_num))
            if prev_ids:
                filtered = [(doc, score) for doc, score in filtered if doc.metadata.get("meeting_id") in prev_ids]
            else:
                filtered = []

        # similarity threshold
        filtered = [(doc, score) for doc, score in filtered if score >= self.similarity_th]

        # (POC) 후보 노드 등록
        for doc, _ in filtered:
            cid = doc.metadata.get("id")
            if cid:
                self._register_node_in_memory(cid)

        return filtered

    def _add_new_node(self, decision: Decision) -> str:
        """
        Decision 엔티티를 Document로 변환해 vectorstore(Neo4jVector)에 적재
        """
        doc = decision_to_document(decision)
        node_id = doc.metadata["id"]
        self.related_store.add_documents(documents=[doc], ids=[node_id])
        self._register_node_in_memory(node_id)
        return node_id

    @staticmethod
    def _rank_key(row: dict) -> tuple:
        """
        후보(또는 대표 후보) 우선순위를 결정하는 정렬 키를 생성
        정렬 기준:
        1) relation_score(LLM)
        2) rag_sim_score
        3) meeting_date(최신)
        """
        rel = row.get("relation_score")
        rag = row.get("rag_sim_score")
        mdate = row.get("meeting_date") or ""  # YYYY-MM-DD
        return (
            -(float(rel) if rel is not None else -1.0),
            -(float(rag) if rag is not None else -1.0),
            -int(mdate.replace("-", "") or 0),
        )

    def _pick_one_rep_per_component(self, rows: List[dict]) -> List[dict]:
        """
        same_agenda=True 후보 rows를 입력으로 받아, "연결요소(component)당 대표 1개"만 선택
        """
        by_comp: Dict[str, List[dict]] = {}
        for r in rows:
            cand_id = r.get("candidate_node_id")
            if not cand_id:
                continue
            comp = self.union_find.find(cand_id)
            by_comp.setdefault(comp, []).append(r)

        reps: List[dict] = []
        for comp_rows in by_comp.values():
            comp_rows_sorted = sorted(comp_rows, key=self._rank_key)
            rep = comp_rows_sorted[0]
            rep["supporting_candidate_node_ids"] = [
                x.get("candidate_node_id") for x in comp_rows_sorted[1:] if x.get("candidate_node_id")
            ]
            reps.append(rep)

        return reps

    def _choose_primary(self, reps: List[dict]) -> dict:
        """
        신규 노드 change_type/base_decision_id를 결정하는 primary 대표 1개 선택
        """
        reps_sorted = sorted(reps, key=self._rank_key)
        return reps_sorted[0]

    def _upsert_edges_existing_to_new(self, new_node_id: str, rep_rows: List[dict]) -> None:
        """
        (기존 candidate) -> (신규 new_node_id) 방향으로 Neo4j 관계를 upsert
        """
        rows = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for r in rep_rows:
            cand_id = r.get("candidate_node_id")
            if not cand_id:
                continue

            props = {
                "relation_score": r.get("relation_score"),
                "rag_sim_score": r.get("rag_sim_score"),
                "change_type": r.get("change_type"),
                "base_decision_id": r.get("base_decision_id"),
                "reason": r.get("reason"),
                "created_at": now_iso,
            }

            supporting = r.get("supporting_candidate_node_ids") or []
            if supporting:
                props["supporting_candidate_node_ids"] = supporting

            rows.append({"src": cand_id, "dst": new_node_id, "props": props})

        if not rows:
            return

        cypher = f"""
        UNWIND $rows AS row
        MATCH (a:`{self.node_label}` {{id: row.src}})
        MATCH (b:`{self.node_label}` {{id: row.dst}})
        MERGE (a)-[r:{self.relationship_type}]->(b)
        SET r += row.props
        """
        self._query(cypher, {"rows": rows})

        # (POC) 메모리 그래프 갱신
        for row in rows:
            self._register_node_in_memory(row["src"])
            self._register_node_in_memory(row["dst"])
            self.union_find.union(row["src"], row["dst"])
            self.reach.add_edge(row["src"], row["dst"])

    def upsert_new_decision(self, new_decision: Decision) -> None:
        """
        신규 Decision 1개를 처리하여 노드/관계를 그래프에 반영하는 메인 엔트리포인트

        1) 후보 검색
        2) LLM 판정(batch)
        3) 신규 노드의 change_type/base_decision_id를 primary 기준으로 확정
        4) 신규 노드 적재
        5) 대표 후보들(existing) -> 신규(new) 엣지 upsert
        """
        candidates_list = self._search_candidates(new_decision)

        # 후보가 없으면 created로 노드만 적재
        if not candidates_list:
            new_decision.change_type = "created"
            new_decision.base_decision_id = None
            self._add_new_node(new_decision)
            return

        new_decision_text = f"title: {new_decision.decision_title}\ntext: {new_decision.raw_text}\n"

        payload = []
        for doc, _score in candidates_list:
            payload.append(
                {
                    "new_decision": new_decision_text,
                    "candidate_decision": format_decision_for_prompt(doc),
                }
            )

        config = RunnableConfig(max_concurrency=self.max_concurrency)
        outputs = self.agenda_cls_chain.batch(payload, config=config, return_exceptions=True)

        related: List[dict] = []
        for (doc, rag_score), out in zip(candidates_list, outputs):
            if isinstance(out, Exception):
                related.append(
                    {
                        "candidate_node_id": doc.metadata.get("id"),
                        "candidate_decision_id": doc.metadata.get("decision_id"),
                        "error": repr(out),
                    }
                )
                continue

            # LLM 출력 표준화
            out: AgendaClsOutput
            related.append(
                {
                    "candidate_node_id": doc.metadata.get("id"),
                    "candidate_decision_id": doc.metadata.get("decision_id"),
                    "meeting_date": doc.metadata.get("meeting_date"),
                    "relation_score": out.relation_score,
                    "rag_sim_score": rag_score,
                    "is_same_agenda": out.is_same_agenda,
                    "change_type": out.change_type,
                    "base_decision_id": out.base_decision_id or doc.metadata.get("decision_id"),
                    "reason": out.reason,
                }
            )

        same_agenda = [
            r for r in related
            if r.get("is_same_agenda") is True and r.get("candidate_node_id") and "error" not in r
        ]

        # 같은 안건이 없으면 created로 노드만 적재
        if not same_agenda:
            new_decision.change_type = "created"
            new_decision.base_decision_id = None
            self._add_new_node(new_decision)
            return

        # 컴포넌트(연결요소) 당 대표 1개
        reps = self._pick_one_rep_per_component(same_agenda)

        # 신규 노드 change_type/base_decision_id를 primary로 확정
        primary = self._choose_primary(reps)
        new_decision.change_type = primary.get("change_type") or "created"
        new_decision.base_decision_id = primary.get("base_decision_id")

        # 신규 노드 적재(LLM change_type 반영된 상태로 저장)
        new_node_id = self._add_new_node(new_decision)

        # (기존 candidate) -> (신규) 엣지 upsert
        self._upsert_edges_existing_to_new(new_node_id, reps)
