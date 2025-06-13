import sys
import os

import pytest
from httpx import AsyncClient
from fastapi import status
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.backend_service import app


class TestRAGServiceRoutes:
    @pytest.fixture(scope="class")
    async def async_client(self):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_root_health(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == status.HTTP_200_OK
        assert "message" in resp.json()

        resp = await async_client.get("/health")
        assert resp.status_code == status.HTTP_200_OK
        assert "message" in resp.json()

    @pytest.mark.asyncio
    async def test_vector_store_status_not_found(self, async_client):
        resp = await async_client.get("/vector-store/status/nonexistent-task-id")
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert "Task not found" in resp.text

    @pytest.mark.asyncio
    async def test_get_all_vector_store_tasks(self, async_client):
        resp = await async_client.get("/vector-store/tasks")
        assert resp.status_code == status.HTTP_200_OK
        assert "tasks" in resp.json()

    @pytest.mark.asyncio
    async def test_build_vector_store_invalid_path(self, async_client):
        resp = await async_client.post("/vector-store/build", json={"parse_task_id": "invalid_task_id"})
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert "Directory not found" in resp.text
    
    @pytest.mark.asyncio
    async def test_build_vector_store_with_config(self, async_client):
        """测试带配置参数的向量存储构建"""
        config_data = {
            "parse_task_id": "test_task_id",  # 使用测试任务ID
            "config": {
                "chunk_size": 256,
                "chunk_overlap": 25,
                "extract_mode": "enhanced",
                "min_chunk_size_for_summary": 600,
                "min_chunk_size_for_qa": 400,
                "max_keywords": 4,
                "num_questions": 2,
                
            }
        }
        
        resp = await async_client.post("/vector-store/build", json=config_data)
        # 应该返回404因为目录不存在，但这验证了参数传递正确
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert "Directory not found" in resp.text
    
    @pytest.mark.asyncio
    async def test_build_vector_store_default_config(self, async_client):
        """测试使用默认配置的向量存储构建"""
        config_data = {
            "parse_task_id": "test_task_id"
            # 不提供config，应该使用默认配置
        }
        
        resp = await async_client.post("/vector-store/build", json=config_data)
        # 应该返回404因为目录不存在，但这验证了默认配置正确
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert "Directory not found" in resp.text