# AI 스테이지 구현 가이드

`stages/` 아래 파일 3개를 구현하면 됩니다.
현재 구현은 Gemini 탐지와 Pillow 편집이다. 아래 모델 목록은 교체 방향이며 실제 구현은 코드가 기준이다.
큐 계약 v2에서 워커는 탐지 완료 직후 원본을 삭제한다. analyze는 원본 이미지 없이 findings로
판단해야 한다. 편집은 재전송 파일과 최초 findings 스냅샷을 사용하며, 재검증은 편집 결과를 받는다.

---

## 파일 위치 및 시그니처

```python
# stages/detect.py
async def detect(media: Media) -> list[dict[str, Any]]: ...


# stages/risk.py
async def analyze(media: Media, findings: list[dict[str, Any]]) -> Assessment: ...


# stages/transform.py
async def transform(media: Media, assessment: Assessment) -> Media: ...
```

**타입 정의** (`stages/types.py`):

```python
@dataclass
class Media:
    image: bytes | None  # 원본 이미지 바이트 (jpg/png/webp)
    image_ext: str | None  # "jpg", "png", "webp"
    text: str | None  # 입력 텍스트


@dataclass
class Assessment:
    risky: bool  # 위험 요소 남아있으면 True
    risk: float  # 0.0 ~ 1.0
    findings: list[dict]  # detect()가 반환한 findings 그대로 넘어옴
```

---

## ① detect.py — 탐지

**역할**: 이미지/텍스트에서 민감 정보 위치와 종류를 찾아 반환

**입력**: `Media` (image, text 둘 다 있을 수도 있음)

**출력**: findings 리스트. 각 항목은 아래 형식을 **반드시** 포함:

```python
{
    "type": str,  # 필수. "face" | "license_plate" | "address" | "phone" | "email" | "id_number" | "gps" 등
    "source": str,  # 필수. "yolo" | "ocr" | "exif" | "regex" | "vlm" 등
    "bbox": [x1, y1, x2, y2],  # 이미지 탐지 시 필수. 픽셀 좌표 (왼쪽 위, 오른쪽 아래)
    "score": float,  # 신뢰도 0.0~1.0
    # 추가 필드 자유롭게 붙여도 됨
}
```

**기대 구현**:

| 입력 | 방법 | 탐지 대상 |
|------|------|-----------|
| 이미지 | PaddleOCR PP-OCRv5 | 텍스트 영역 및 좌표 |
| 이미지 | YOLO 또는 VLM (Gemini API) | 얼굴, 차량번호판, 표지판 등 |
| 이미지 | EXIF 파싱 | GPS 좌표, 촬영 시각, 기기 정보 |
| 텍스트 | NER + Regex | 전화번호, 이메일, 주소, 주민번호 등 |

**주의**: 탐지 실패는 예외로 보고한다. 정상 검사에서 단서가 없을 때만 빈 리스트 `[]`를 반환한다.

---

## ② risk.py — 위험도 분석

**역할**: findings를 보고 "아직 위험한가?"를 판단

**입력**: `Media` + findings 리스트

**출력**: `Assessment`

```python
Assessment(
    risky=True,  # 아직 처리가 필요하면 True → 루프가 transform 한 번 더 돔
    risk=0.85,  # 전체 위험도 점수 (0.0~1.0)
    findings=findings,  # 받은 findings 그대로 또는 보강해서 반환
)
```

**루프 동작 방식** (오케스트레이터가 이렇게 씁니다):

```python
findings = await detect(media)
assessment = await analyze(media, findings)

while assessment.risky and iterations < max_iterations:
    media = await transform(media, assessment)
    findings = await detect(media)
    assessment = await analyze(media, findings)  # ← risky=False 가 되면 루프 종료
    iterations += 1
```

`risky=False`를 반환하는 게 루프 종료 조건입니다.
변환 후 재탐지했을 때 더 이상 위험 요소가 없으면 `risky=False`를 반환해주세요.

**기대 구현**:

- findings 조합 기반 규칙 (예: 얼굴 + 주소 → 고위험, GPS 단독 → 중위험)
- risk 점수: findings의 type별 가중치 × score 합산
- 참고: 개인정보보호법 민감도 등급 (1~4단계) 기준 활용 가능

---

## ③ transform.py — 변환

**역할**: findings의 bbox 위치를 블러/마스킹, EXIF 제거, 텍스트 마스킹

**입력**: `Media` + `Assessment`

**출력**: 변환된 `Media` (**원본 덮어쓰지 말고 새 객체 반환**)

```python
return Media(
    image=processed_image_bytes,  # 블러/마스킹 적용된 이미지
    image_ext=media.image_ext,
    text=masked_text,  # 텍스트 마스킹 결과
)
```

**기대 구현**:

| 대상 | 처리 방법 |
|------|-----------|
| `bbox` 있는 finding | 해당 영역 블러 또는 단색 박스 (SAM2 윤곽 마스크 → OpenCV 덮어쓰기) |
| `source == "exif"` | EXIF 메타데이터 전체 제거 (내장 썸네일 포함) |
| 텍스트 finding | 해당 문자열 `***` 치환 또는 LLM 일반화 재작성 |

**각 finding에 `"action"` 필드 추가 필수** (결과 리포트에 표시됩니다):

```python
finding["action"] = "blur"  # 또는 "mask" | "remove" | "skip"
```

---

## 백엔드가 최종적으로 사용자에게 반환하는 것

AI 스테이지 구현이 완료되면 사용자는 이런 리포트를 받습니다:

```json
{
  "risk_before": 0.85,
  "risk_after": 0.05,
  "iterations": 2,
  "findings": [
    { "type": "face",    "source": "yolo", "bbox": [40, 60, 320, 110], "score": 0.97, "action": "blur" },
    { "type": "address", "source": "ocr",  "bbox": [10, 200, 400, 230], "score": 0.91, "action": "mask" },
    { "type": "gps",     "source": "exif", "score": 1.0, "action": "remove" }
  ]
}
```

`type` · `source` · `action` 값은 팀 내에서 일관성 있게 정해주세요. 이 값이 사용자에게 그대로 보입니다.

---

## 테스트 방법

```bash
# 의존성 설치
uv sync

# 단위 테스트 (루프 동작 검증)
uv run pytest tests/

# Redis/S3 연결 확인
uv run python -m worker.check

# 실제 워커 실행
uv run python -m worker.main
```

궁금한 점은 `stages/types.py` 타입 정의와 `worker/pipeline.py` 루프 로직을 먼저 읽어보시면 전체 흐름이 파악됩니다.
