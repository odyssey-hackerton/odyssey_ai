"""텍스트 개인정보 판별: Veil-PII-Ko-Lite(NER, ONNX INT8) + ko-pii(규칙·체크섬) + 정규식.

세 탐지기를 모두 돌린 뒤 같은 구간은 ko-pii > veil > regex 순으로 하나만 남긴다.
라이브러리가 설치돼 있지 않으면 그 탐지기만 건너뛴다 (정규식은 항상 동작).
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

from worker.config import get_settings

log = logging.getLogger(__name__)

VEIL_TYPES = {
    "PERSON": "name", "PHONE": "phone_number", "EMAIL": "email", "ADDRESS": "address",
    "ZIPCODE": "zipcode", "URL": "url", "RRN": "rrn", "FRN": "rrn", "SSN": "rrn",
    "ACCOUNT_NUMBER": "account_number", "CARD_NUMBER": "card_number",
    "VIRTUAL_CARD_NUMBER": "card_number", "BUSINESS_ID": "business_id",
    "VEHICLE_PLATE": "license_plate", "USER_ID": "user_id", "DATE": "date",
    "ORGANIZATION": "store_name",
}  # fmt: skip
KOPII_TYPES = {
    "RRN": "rrn", "FRN": "rrn", "PHONE": "phone_number", "EMAIL": "email",
    "CARD": "card_number", "CREDIT_CARD": "card_number", "BANK_ACCOUNT": "account_number",
    "ACCOUNT": "account_number", "BUSINESS_NUMBER": "business_id",
    "VEHICLE_PLATE": "license_plate", "VEHICLE": "license_plate", "PLATE": "license_plate",
    "ADDRESS": "address",
    "PERSON": "name", "URL": "url", "ZIPCODE": "zipcode", "POSTAL_CODE": "zipcode",
}  # fmt: skip
REGEX_TYPES: list[tuple[str, re.Pattern[str], float]] = [
    ("rrn", re.compile(r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)"), 0.9),
    ("phone_number", re.compile(r"(?<!\d)01[016-9]-?\d{3,4}-?\d{4}(?!\d)"), 0.85),
    ("phone_number", re.compile(r"(?<!\d)0(2|[3-6]\d)-?\d{3,4}-?\d{4}(?!\d)"), 0.7),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), 0.9),
    (
        "license_plate",
        re.compile(
            r"(?<![\d가-힣])\d{2,3}[가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
            r"고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후배]\s?\d{4}(?!\d)"
        ),
        0.8,
    ),
    ("apt_dong_ho", re.compile(r"\d{1,4}\s*동\s*\d{1,4}\s*호"), 0.75),
    ("pharmacy_name", re.compile(r"[가-힣A-Za-z0-9]{1,12}(?:약국|의원|병원|한의원)"), 0.65),
    ("tracking_number", re.compile(r"(?<!\d)\d{10,14}(?!\d)"), 0.5),
    (
        "absence_period",
        re.compile(r"(?:오늘|내일|모레)부터\s*(?:\d+박\s*\d+일|일주일|이틀|사흘)"),
        0.8,
    ),
]
_PRIORITY = {"ko-pii": 2, "veil": 1, "regex": 0}


@dataclass
class PiiSpan:
    start: int
    end: int
    type: str
    score: float
    detector: str  # "veil" | "ko-pii" | "regex"


@lru_cache
def _veil():
    directory = get_settings().veil_dir
    if not os.path.exists(os.path.join(directory, "model.int8.onnx")):
        log.warning("Veil 모델 없음(%s). NER 건너뜀 — scripts/download-models.sh 참고", directory)
        return None
    sys.path.insert(0, directory)
    from veil import Veil  # HF 레포(1T/veil-pii-ko-lite)에 포함된 veil.py

    return Veil(os.path.join(directory, "model.int8.onnx"), tokenizer_dir=directory, threads=4)


@lru_cache
def _kopii():
    try:
        from ko_pii import Anonymizer, ProcessingMode
    except ImportError:
        log.warning("ko-pii 미설치. 규칙 탐지 건너뜀")
        return None
    return Anonymizer(mode=ProcessingMode.AUDIT, strategy="redact")


def detect_pii(text: str) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    veil = _veil()
    if veil is not None:
        for s in veil.predict(text):
            type_ = VEIL_TYPES.get(s["label"])
            if type_:
                spans.append(PiiSpan(s["start"], s["end"], type_, float(s["score"]), "veil"))
    kopii = _kopii()
    if kopii is not None:
        try:
            for record in kopii.process(text).detections:
                d = record.detection  # DetectionRecord(detection=DetectionResult(...), action, ...)
                type_ = KOPII_TYPES.get(str(d.label).upper())
                if type_:
                    spans.append(PiiSpan(d.start, d.end, type_, float(d.confidence), "ko-pii"))
        except Exception:  # noqa: BLE001 — 규칙 라이브러리 오류가 전체 탐지를 막지 않게
            log.exception("ko-pii 처리 실패")
    for type_, pattern, score in REGEX_TYPES:
        for m in pattern.finditer(text):
            spans.append(PiiSpan(m.start(), m.end(), type_, score, "regex"))
    return _dedupe(spans)


def _dedupe(spans: list[PiiSpan]) -> list[PiiSpan]:
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[PiiSpan] = []
    for s in spans:
        clash = next((k for k in kept if s.start < k.end and s.end > k.start), None)
        if clash is None:
            kept.append(s)
        elif (_PRIORITY[s.detector], s.score) > (_PRIORITY[clash.detector], clash.score):
            kept[kept.index(clash)] = s
    return kept
