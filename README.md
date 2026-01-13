# decision_tracking_rag_system
회의록에서 추출한 의사결정사항을 추적하는 코드

## description
결정사항(Decision) JSON 데이터를 순차적으로 읽어,
1) Neo4jVector에서 유사 후보를 검색하고  
2) LLM으로 same_agenda / change_type / relation_score를 판정한 뒤  
3) 신규 Decision 노드에 change_type/base_decision_id를 반영하여 저장하고  
4) (기존 후보) -> (신규 노드) 방향의 RELATED 엣지를 upsert 합니다.

## 특징
- relation_score로 네이밍 통일
- 신규 노드의 change_type도 LLM 결과로 확정(대표 후보들 중 primary 기준)
- decisions는 JSON으로 분리하여 관리

## 설치
```bash
pip install -r requirements.txt