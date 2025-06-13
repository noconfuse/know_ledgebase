import os
import pytest
from httpx import AsyncClient
from fastapi import status
from apps.backend_service import app

@pytest.mark.asyncio
class TestDocumentParserRoutes:
    @pytest.fixture(scope="class")
    async def async_client(self):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    async def test_parse_uploaded_file_invalid_format(self, async_client):
        # 上传不支持的文件格式
        files = {"file": ("test.unsupported", b"dummy content")}
        response = await async_client.post("/parse/upload", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported file format" in response.text

    async def test_parse_uploaded_file_missing_file(self, async_client):
        # 未上传文件
        response = await async_client.post("/parse/upload", files={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_parse_file_path_not_found(self, async_client):
        # 解析不存在的文件路径
        response = await async_client.post("/parse/file", json={"file_path": "/not/exist/file.pdf"})
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "File not found" in response.text

    async def test_get_parse_status_not_found(self, async_client):
        # 查询不存在的任务状态
        response = await async_client.get("/parse/status/nonexistent-task-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Task not found" in response.text

    async def test_get_all_parse_tasks(self, async_client):
        # 查询所有解析任务（初始应为空或为列表）
        response = await async_client.get("/parse/tasks")
        assert response.status_code == status.HTTP_200_OK
        assert "tasks" in response.json()

    async def test_get_task_logs_not_found(self, async_client):
        # 查询不存在任务的日志
        response = await async_client.get("/parse/logs/nonexistent-task-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Task not found" in response.text