import pytest
from io import BytesIO

from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from main import app


def test_health_returns_liveness_envelope():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "service": "chromasense"}


def test_root_serves_static_ui():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Chromasense" in response.text
    assert "uploadForm" in response.text
    assert "Displayed" in response.text
    assert "Secondary Palette" in response.text


def test_invalid_upload_returns_structured_error_envelope():
    client = TestClient(app)

    response = client.post(
        "/api/analyze",
        files={"file": ("not-image.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "invalid_image",
            "message": "Upload must be a valid image.",
        },
    }


def test_analyze_response_excludes_deprecated_fields():
    client = TestClient(app)
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/analyze",
        files={"file": ("test.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "font" not in payload
    assert "mockup" not in payload
    assert "similar_palettes" not in payload
