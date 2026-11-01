"""EXIF 파서: GPS·촬영 시각·기기 정보를 finding 으로 만든다.
리포트(DB)에 좌표 원값이 남지 않도록 details 에는 소수점 2자리(약 1km)까지만 넣는다."""

from io import BytesIO
from typing import Any

from PIL import Image

from stages.catalog import describe

GPS_IFD, TAG_MAKE, TAG_MODEL, TAG_DT_ORIGINAL, TAG_DT = 34853, 271, 272, 36867, 306


def _degrees(value) -> float:
    d, m, s = (float(v) for v in value)
    return d + m / 60 + s / 3600


def exif_findings(data: bytes) -> list[dict[str, Any]]:
    with Image.open(BytesIO(data)) as image:
        exif = image.getexif()
        if not exif:
            return []
        gps = exif.get_ifd(GPS_IFD) if hasattr(exif, "get_ifd") else {}
        findings: list[dict[str, Any]] = []
        if gps and 2 in gps and 4 in gps:
            try:
                lat, lon = _degrees(gps[2]), _degrees(gps[4])
                lat = -lat if gps.get(1) == "S" else lat
                lon = -lon if gps.get(3) == "W" else lon
                findings.append(
                    describe("gps")
                    | {
                        "source": "exif",
                        "score": 1.0,
                        "action": "remove",
                        "details": {"lat": round(lat, 2), "lon": round(lon, 2)},
                    }
                )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        taken = exif.get(TAG_DT_ORIGINAL) or exif.get(TAG_DT)
        if taken:
            findings.append(
                describe("capture_time")
                | {
                    "source": "exif",
                    "score": 1.0,
                    "action": "remove",
                    "details": {"date": str(taken)[:10]},
                }
            )
        make, model = exif.get(TAG_MAKE), exif.get(TAG_MODEL)
        if make or model:
            device = " ".join(str(v).strip("\x00 ") for v in (make, model) if v)
            findings.append(
                describe("device_info")
                | {
                    "source": "exif",
                    "score": 1.0,
                    "action": "remove",
                    "details": {"device": device},
                }
            )
        if not findings:  # 민감 태그는 없지만 메타데이터는 있음 → 기존 동작 유지
            findings.append(
                describe("metadata") | {"source": "exif", "score": 1.0, "action": "remove"}
            )
        return findings
