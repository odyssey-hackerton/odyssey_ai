"""테스트는 실제 모델(PaddleOCR·Veil·ko-pii)과 외부 API 를 호출하지 않는다.
로컬 모델이 설치된 환경에서도 같은 결과가 나오도록 전역으로 스텁을 건다."""

import pytest

from stages import detect, pii


@pytest.fixture(autouse=True)
def no_local_models(monkeypatch):
    monkeypatch.setattr(detect, "read_lines", lambda image: [])  # OCR 없음
    monkeypatch.setattr(pii, "_veil", lambda: None)  # Veil NER 없음 → 정규식만
    monkeypatch.setattr(pii, "_kopii", lambda: None)  # ko-pii 없음 → 정규식만
