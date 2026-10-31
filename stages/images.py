"""탐지 좌표와 실제 편집에 동일한 EXIF 회전·픽셀 크기를 사용한다."""

import warnings
from io import BytesIO

from PIL import Image, ImageOps


def open_image(data: bytes) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"} or getattr(
                original, "is_animated", False
            ):
                raise ValueError("unsupported image format")
            if original.width * original.height > 25_000_000:
                raise ValueError("image exceeds 25 megapixels")
            original.load()
            image = ImageOps.exif_transpose(original)
            if "A" in image.getbands() or "transparency" in image.info:
                # 투명 픽셀의 숨은 RGB 값이 결과에서 드러나지 않도록 합성한다.
                image = Image.alpha_composite(
                    Image.new("RGBA", image.size, "white"), image.convert("RGBA")
                )
            return image.convert("RGB")


def png_bytes(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="PNG", exif=b"")
    return out.getvalue()
