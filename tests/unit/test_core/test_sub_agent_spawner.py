"""Unit tests for core.sub_agent_spawner — parallel task management."""
import asyncio

import pytest

from core.sub_agent_spawner import AgentStatus, BatchResult, SubAgentSpawner


async def _success_task(value="ok"):
    return value


async def _failing_task():
    raise ValueError("intentional failure")


async def _slow_task():
    await asyncio.sleep(10)
    return "done"


class TestSubAgentSpawner:
    """Tests for SubAgentSpawner core logic."""

    def test_init_defaults(self):
        s = SubAgentSpawner()
        assert s.max_concurrency == 5
        assert s.tasks == []

    def test_init_custom_concurrency(self):
        s = SubAgentSpawner(max_concurrency=10)
        assert s.max_concurrency == 10

    def test_add_task(self):
        s = SubAgentSpawner()
        s.add_task("t1", _success_task, name="test1", timeout_sec=5.0)
        assert len(s.tasks) == 1
        assert s.tasks[0].task_id == "t1"
        assert s.tasks[0].name == "test1"
        assert s.tasks[0].status == AgentStatus.PENDING

    def test_add_multiple_tasks(self):
        s = SubAgentSpawner()
        s.add_task("t1", _success_task)
        s.add_task("t2", _success_task)
        s.add_task("t3", _failing_task)
        assert len(s.tasks) == 3

    @pytest.mark.asyncio
    async def test_run_all_success(self):
        s = SubAgentSpawner()
        s.add_task("t1", _success_task, "hello")
        s.add_task("t2", _success_task, "world")
        result = await s.run_all()
        assert isinstance(result, BatchResult)
        assert result.total == 2
        assert result.completed == 2
        assert result.failed == 0
        assert result.results["t1"] == "hello"
        assert result.results["t2"] == "world"

    @pytest.mark.asyncio
    async def test_run_all_with_failure(self):
        s = SubAgentSpawner()
        s.add_task("ok", _success_task)
        s.add_task("fail", _failing_task, retries=0)
        result = await s.run_all()
        assert result.completed == 1
        assert result.failed == 1
        assert "ok" in result.results
        assert "fail" in result.errors

    @pytest.mark.asyncio
    async def test_run_all_timeout(self):
        s = SubAgentSpawner()
        s.add_task("slow", _slow_task, timeout_sec=0.1, retries=0)
        result = await s.run_all()
        assert result.timed_out == 1
        assert result.completed == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        call_count = 0

        async def _retry_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("not yet")
            return "finally"

        s = SubAgentSpawner()
        s.add_task("retry", _retry_task, retries=3)
        result = await s.run_all()
        assert result.completed == 1
        assert result.results["retry"] == "finally"

    def test_clear(self):
        s = SubAgentSpawner()
        s.add_task("t1", _success_task)
        s.clear()
        assert len(s.tasks) == 0

    def test_get_status(self):
        s = SubAgentSpawner()
        s.add_task("t1", _success_task, name="Task One")
        status = s.get_status()
        assert status["total_tasks"] == 1
        assert status["max_concurrency"] == 5
        assert status["tasks"][0]["id"] == "t1"
        assert status["tasks"][0]["name"] == "Task One"
        assert status["tasks"][0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Verify concurrency is actually limited."""
        max_concurrent = 0
        current = 0

        async def _track_task():
            nonlocal max_concurrent, current
            current += 1
            if current > max_concurrent:
                max_concurrent = current
            await asyncio.sleep(0.05)
            current -= 1
            return True

        s = SubAgentSpawner(max_concurrency=2)
        for i in range(6):
            s.add_task(f"t{i}", _track_task)
        result = await s.run_all()
        assert result.completed == 6
        assert max_concurrent <= 2
