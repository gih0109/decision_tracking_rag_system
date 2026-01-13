from __future__ import annotations

from typing import Literal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector


def get_embeddings(
    emb_model_name: str = "BAAI/bge-m3",
    device: Literal["cpu", "cuda"] = "cpu",
) -> HuggingFaceEmbeddings:
    """
    embeddings 생성
    """
    return HuggingFaceEmbeddings(
        model_name=emb_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_neo4j_vector_store(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    collection_name: str,
    node_label: str = "Decision",
    emb_model_name: str = "BAAI/bge-m3",
    emb_device: Literal["cpu", "cuda"] = "cpu",
) -> Neo4jVector:
    """
    Neo4jVector 생성 + 인덱스 자동 생성

    - Neo4j 5.11+ 권장(벡터 인덱스 기능)
    - retrieval_query에서 metadata의 id가 null로 덮이는 문제를 회피
    """
    # id를 Null로 지우지 않는 retrieval_query 지정 - 기본 쿼리는 여기서 id: Null 로 덮어써서 metadata에서 id가 사라질 수 있다
    retrieval_query = """
    RETURN node.`text` AS text, score,
           node {.*, `text`: Null, `embedding`: Null} AS metadata
    """

    embeddings = get_embeddings(emb_model_name, emb_device)

    store = Neo4jVector(
        embedding=embeddings,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=f"{collection_name}_vector",
        node_label=node_label,
        text_node_property="text",
        embedding_node_property="embedding",
        retrieval_query=retrieval_query,
    )

    info = store.retrieve_existing_index()
    if info is None:
        store.create_new_index()
    else:
        dim, _ = info
        if dim is None:
            store.create_new_index()

    return store
