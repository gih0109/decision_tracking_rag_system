from __future__ import annotations

from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


AGENDA_SYS_TEMPLATE = """
너는 회의 결정사항의 변경 여부를 판정하는 전문가이다.
입력으로 '새로운 결정사항'과 '이전 후보 결정사항'이 주어지면, 다음 항목을 판단해야 한다:

1) 같은 안건인지 (is_same_agenda)
2) 변경 유형 (change_type)
3) 같은 안건이라고 판단한 경우에만, 연관관계 점수(relation_score)를 평가한다.
   - is_same_agenda=True 인 경우에만 relation_score(0.0~1.0)을 채운다.
   - is_same_agenda=False 인 경우 relation_score은 반드시 null 로 둔다.

[강도 루브릭: is_same_agenda=True 전제]
- 0.90~1.00 (very_strong): 문구/조건/대상/산출물이 거의 동일
- 0.70~0.89 (strong): 핵심 동일, 일부 조건/범위/일정/수치만 조정
- 0.40~0.69 (medium): 같은 안건이지만 보강/변경이 꽤 큼
- 0.10~0.39 (weak): 같은 안건으로 보이나 공통 근거가 비교적 약함
- 0.00~0.09 (none): (원칙적으로 is_same_agenda=True이면 드물어야 함)

4) 기준이 된 이전 decision_id (base_decision_id)

모든 판단은 텍스트 근거에 기반해야 하며 reason에 근거를 적어라.
""".strip()


AGENDA_HUMAN_TEMPLATE = """
새로운 결정사항:
{new_decision}

이전 후보 결정사항:
{candidate_decision}

위 정보를 바탕으로 JSON 형식의 구조화 결과를 출력하라.
""".strip()

agenda_cls_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENDA_SYS_TEMPLATE),
        ("human", AGENDA_HUMAN_TEMPLATE),
    ]
)


class AgendaClsOutput(BaseModel):
    is_same_agenda: bool

    relation_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="0.0~1.0 (is_same_agenda=True일 때만)",
    )

    change_type: Literal["created", "updated", "changed", "canceled", "unchanged"]
    base_decision_id: Optional[str] = None
    reason: str


def build_nn_agenda_cls_chain(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
) -> Runnable:
    """
    GPT/Gemini 공용 agenda classification chain
    """
    name = model_name.lower()
    if "gpt" in name:
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif "gemini" in name:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return agenda_cls_prompt | llm.with_structured_output(AgendaClsOutput)


def build_nn_agenda_cls_gpt5_chain(
    model_name: str = "gpt-5.1",
    temperature: float = 0.0,
    reasoning_effort: Literal["low", "medium", "high"] | None = "low",
) -> Runnable:
    """
    GPT-5(reasoning_effort 지원)용 chain
    """
    # reasoning_effort를 쓰면 temperature를 None으로 두는 패턴을 유지
    temp = None if reasoning_effort is not None else temperature
    llm = ChatOpenAI(
        model=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temp,
    )
    return agenda_cls_prompt | llm.with_structured_output(AgendaClsOutput)


