"""③ 변환: 위험한 부분을 블러·마스킹·제거한다."""

from stages.types import Assessment, Media


async def transform(media: Media, assessment: Assessment) -> Media:
    # TODO(AI): assessment.findings 의 bbox 블러, 글 마스킹, EXIF 제거
    # 각 finding 에 실제로 한 처리를 "action" 으로 남긴다 (blur, mask, remove)
    return media
