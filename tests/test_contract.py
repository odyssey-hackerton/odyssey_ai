import json
import uuid

from worker.contract import JobMessage


def test_job_message_ignores_unknown_fields():
    """백엔드가 필드를 추가해도 워커가 깨지지 않아야 한다."""
    payload = {
        "schema_version": 1,
        "job_id": str(uuid.uuid4()),
        "input_type": "image_text",
        "object_key": "originals/a.jpg",
        "text": "본문",
        "created_at": "2026-10-31T00:00:00Z",
        "new_field": 1,
    }
    msg = JobMessage.model_validate_json(json.dumps(payload))
    assert msg.input_type == "image_text"
    assert msg.options.max_iterations == 3
