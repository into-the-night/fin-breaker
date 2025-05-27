import pytest
from fastapi.testclient import TestClient
from agents.api_agent import router as api_router
from agents.voice_agent import router as voice_router
from main import app

app.include_router(api_router)
app.include_router(voice_router)

client = TestClient(app)

def test_api_market_data():
    response = client.get("/api/market_data?ticker=TSM")
    assert response.status_code == 200
    assert "error" not in response.json()

def test_voice_root():
    response = client.get("/voice/")
    assert response.status_code == 200
    assert response.json()["status"] == "Voice Agent running"