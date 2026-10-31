"""API-워커 큐 메시지 계약 v2.

odyssey_backend 의 app/schemas/contract.py 와 같은 내용이어야 한다. 바꿀 때는 두 레포를 함께 고친다.

AI 쪽은 모델과 로직이 자주 바뀌므로 아래 규칙으로 변경에 강하게 만든다.
- 모르는 필드는 무시한다 (extra="ignore").
- findings[].type / source / action 은 enum이 아니라 문자열이다.
- findings 항목의 추가 필드와 details 는 해석하지 않고 그대로 저장한다.
- 필드 삭제·이름 변경처럼 깨지는 변경만 schema_version 을 올리고 두 레포를 함께 수정한다.

Redis Stream 메시지는 {"payload": <JSON 문자열>} 하나의 필드로 주고받는다.

입력 종류(input_type)
- image:      사진만.      object_key 있음, text 없음
- text:       글만.        object_key 없음, text 있음
- image_text: 사진 + 글.   object_key, text 둘 다 있음 (예: SNS 게시물 사진과 본문)
  → 워커는 두 입력을 한 작업에서 함께 분석하고, 결과도 result_object_key 와
    result_text 를 함께 채워 보낸다.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 2


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    DETECTING = "DETECTING"
    ANALYZING = "ANALYZING"
    AWAITING_USER = "AWAITING_USER"  # 탐지·분석 완료, 사용자 선택 대기
    TRANSFORMING = "TRANSFORMING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"


FINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED}
TERMINAL_STATUSES = FINAL_STATUSES | {JobStatus.AWAITING_USER}


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class JobOptions(ContractModel):
    max_iterations: int = Field(default=3, ge=1, le=3)


InputType = Literal["image", "text", "image_text"]


class UserChoice(ContractModel):
    """사용자가 선택한 탐지 항목별 처리 방법."""

    idx: int  # findings 리스트 인덱스
    action: str  # "blur" | "mask" | "remove" | "skip"


class Finding(BaseModel):
    """탐지 항목. AI 쪽이 필드를 추가해도 버리지 않고 보존한다."""

    model_config = ConfigDict(extra="allow")

    type: str
    source: str | None = None
    bbox: list[float] | None = None
    score: float | None = None
    action: str | None = None


class JobMessage(ContractModel):
    """jobs stream (API → 워커)"""

    schema_version: Literal[2] = SCHEMA_VERSION
    job_id: UUID
    execution_id: UUID
    input_type: InputType
    object_key: str | None = None  # 원본 이미지 S3 키 (image, image_text)
    text: str | None = None  # 원본 글 (text, image_text). Postgres에는 저장하지 않는다.
    options: JobOptions = Field(default_factory=JobOptions)
    created_at: datetime
    # None이면 Phase 1 (탐지·분석만 → AWAITING_USER 반환)
    # 값이 있으면 Phase 2 (사용자 선택 반영 후 변환·검증 실행)
    user_choices: list[UserChoice] | None = None
    # Phase 1에 표시한 항목을 그대로 편집한다. 재탐지로 인덱스를 바꾸지 않는다.
    initial_findings: list[Finding] | None = None


class ResultMessage(ContractModel):
    """results stream (워커 → API)"""

    schema_version: Literal[2] = SCHEMA_VERSION
    job_id: UUID
    execution_id: UUID
    status: Literal["DONE", "FAILED", "AWAITING_USER"]
    risk_detected: bool | None = None
    risk_before: float | None = None
    risk_after: float | None = None
    iterations: int = 0
    result_object_key: str | None = None  # 처리된 이미지 S3 키 (image, image_text)
    result_text: str | None = None  # 처리된 글 (text, image_text)
    findings: list[Finding] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
