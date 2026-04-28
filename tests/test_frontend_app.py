from io import BytesIO

from fastapi.testclient import TestClient

from services.frontend.app import _PENDING, app


def setup_function() -> None:
    _PENDING.clear()


def teardown_function() -> None:
    _PENDING.clear()


def test_backend_status_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/backend-status")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "message" in body


def test_text_mode_accepts_multiple_txt_uploads() -> None:
    files = [
        ("files", ("first.txt", BytesIO(b"alpha article"), "text/plain")),
        ("files", ("second.txt", BytesIO(b"beta article"), "text/plain")),
    ]
    data = {
        "mode": "text",
        "text": "",
        "configs": '["hf_baseline_bf16"]',
        "concurrency": "4",
        "limit": "0",
    }
    with TestClient(app) as client:
        resp = client.post("/api/jobs", data=data, files=files)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    job = _PENDING[payload["id"]]
    assert job.mode == "text"
    assert [item["name"] for item in job.text_inputs] == ["first.txt", "second.txt"]
    assert [item["text"] for item in job.text_inputs] == ["alpha article", "beta article"]


def test_jsonl_mode_accepts_single_jsonl_upload() -> None:
    files = [
        ("files", ("batch.jsonl", BytesIO(b'{"text":"hello"}\n{"text":"world"}\n'), "application/jsonl")),
    ]
    data = {
        "mode": "jsonl",
        "text": "",
        "configs": '["vllm_bf16"]',
        "concurrency": "4",
        "limit": "0",
    }
    with TestClient(app) as client:
        resp = client.post("/api/jobs", data=data, files=files)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    job = _PENDING[payload["id"]]
    assert job.mode == "jsonl"
    assert job.jsonl_path is not None
    assert job.jsonl_path.suffix == ".jsonl"
