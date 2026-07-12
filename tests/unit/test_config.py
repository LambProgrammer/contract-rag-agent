"""测试配置加载"""

from src.core.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.app_name == "Contract RAG Agent"
        assert s.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert s.deepseek_model == "deepseek-v4-pro"
        assert s.api_v1_prefix == "/api/v1"

    def test_derived_urls(self):
        s = Settings()
        assert "redis://" in s.celery_broker_url
        assert "asyncpg" in s.database_url
        assert s.qdrant_url.startswith("http://")
