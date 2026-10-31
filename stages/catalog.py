"""탐지 항목 종류(type) 사전. 리포트에 그대로 보이므로 이름·설명을 한 곳에서 관리한다.

type 값은 Gemini 프롬프트(detect.INSTRUCTION)와 로컬 탐지기(ocr/pii/exif)가 같은 이름을 쓴다.
"""

from typing import Any

# type → (risk_level, title, description). 설명에는 민감한 원문을 넣지 않는다.
CATALOG: dict[str, tuple[str, str, str]] = {
    "face": ("high", "얼굴", "얼굴이 그대로 보여 신원이 특정될 수 있어요."),
    "name": ("medium", "이름", "이름이 보여 다른 단서와 합치면 신원이 드러날 수 있어요."),
    "phone_number": ("high", "전화번호", "전화번호가 그대로 읽혀요."),
    "email": ("medium", "이메일", "이메일 주소가 읽혀요."),
    "address": ("high", "주소", "주소가 읽혀 거주지나 활동 장소가 특정돼요."),
    "apt_dong_ho": ("high", "동·호수", "아파트 동·호수가 보여 집이 특정돼요."),
    "zipcode": ("medium", "우편번호", "우편번호로 지역이 좁혀져요."),
    "rrn": ("high", "주민등록번호", "주민등록번호가 노출돼요. 반드시 가려야 해요."),
    "account_number": ("high", "계좌번호", "계좌번호가 읽혀요."),
    "card_number": ("high", "카드번호", "카드번호가 읽혀요."),
    "business_id": ("low", "사업자번호", "사업자등록번호가 읽혀요."),
    "license_plate": ("high", "차량 번호판", "차량 번호로 소유자와 동선이 추적될 수 있어요."),
    "tracking_number": ("high", "송장번호", "송장번호로 배송지(주소)가 조회될 수 있어요."),
    "delivery_label": ("high", "택배 송장", "송장에 이름·주소·전화번호가 함께 적혀 있어요."),
    "pharmacy_name": ("medium", "약국·병원명", "약국·병원 이름으로 생활 반경이 좁혀져요."),
    "store_name": ("medium", "상호명", "가게 이름으로 위치가 특정될 수 있어요."),
    "school_sign": ("medium", "학교 표지", "학교 이름으로 생활 반경이 특정돼요."),
    "bus_stop": ("medium", "정류장 표지", "정류장 이름으로 위치가 좁혀져요."),
    "id_card": ("high", "신분증", "신분증이 보여요. 전체를 가려야 해요."),
    "user_id": ("low", "계정 아이디", "SNS·서비스 아이디가 읽혀요."),
    "url": ("low", "링크", "개인 페이지 링크가 읽혀요."),
    "date": ("low", "날짜", "날짜 정보가 다른 단서와 합쳐질 수 있어요."),
    "absence_period": ("medium", "부재 기간", "집을 비우는 기간이 드러나요."),
    "gps": ("high", "위치 정보(GPS)", "사진에 촬영 위치 좌표가 들어 있어요."),
    "capture_time": ("medium", "촬영 시각", "사진에 촬영 시각이 들어 있어요."),
    "device_info": ("low", "기기 정보", "촬영 기기 모델이 들어 있어요."),
    "metadata": ("medium", "사진 파일 정보", "촬영 정보가 남지 않도록 사진 메타데이터를 제거해요."),
}

# 응답·리포트에 원문을 절대 싣지 않는 type
REDACT_ALWAYS = {"rrn", "card_number", "account_number"}


def describe(type_: str) -> dict[str, Any]:
    level, title, description = CATALOG.get(type_, ("medium", type_, "개인정보 후보가 보여요."))
    return {"type": type_, "title": title, "description": description, "risk_level": level}
