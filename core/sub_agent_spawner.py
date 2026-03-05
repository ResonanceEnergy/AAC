"""
Sub-Agent Spawner — Parallel Data Collection
=============================================

Manages concurrent sub-agent launches for parallel data gathering.
Used by BigBrainIntelligence and CryptoIntelligence departments
to fetch multiple data sources simultaneously.

Features:
- Configurable concurrency limits
- Automatic timeout & retry
- Result aggregation with deduplication
- Error isolation (one failure doesn't kill the batch)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class SubAgentTask:
    """A single sub-agent task definition."""
    task_id: str
    name: str
    fn: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    timeout_sec: float = 30.0
    retries: int = 1
    status: AgentStatus = AgentStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BatchResult:
    """Aggregated results from a sub-agent batch."""
    total: int
    completed: int
    failed: int
    timed_out: int
    results: Dict[str, Any]
    errors: Dict[str, str]
    total_duration_ms: float


class SubAgentSpawner:
    """
    Spawn and manage parallel sub-agent tasks.

    Usage:
        spawner = SubAgentSpawner(max_concurrency=5)
        spawner.add_task("fetch_reddit", fetch_reddit_data, subreddit="crypto")
        spawner.add_task("fetch_news", fetch_crypto_news, query="bitcoin")
        results = await spawner.run_all()
    """

    def __init__(self, max_concurrency: int = 5) -> None:
        self.max_concurrency = max_concurrency
        self.tasks: List[SubAgentTask] = []
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def add_task(
        self,
        task_id: str,
        fn: Callable[..., Coroutine],
        *args: Any,
        name: Optional[str] = None,
        timeout_sec: float = 30.0,
        retries: int = 1,
        **kwargs: Any,
    ) -> None:
        """Add a sub-agent task to the batch."""
        self.tasks.append(SubAgentTask(
            task_id=task_id,
            name=name or task_id,
            fn=fn,
            args=args,
            kwargs=kwargs,
            timeout_sec=timeout_sec,
            retries=retries,
        ))

    async def run_all(self) -> BatchResult:
        """
        Execute all tasks concurrently (up to max_concurrency).

        Returns:
            BatchResult with all results and errors.
        """
        start = time.monotonic()
        coros = [self._run_task(task) for task in self.tasks]
        await asyncio.gather(*coros, return_exceptions=True)
        elapsed = (time.monotonic() - start) * 1000

        results = {}
        errors = {}
        completed = 0
        failed = 0
        timed_out = 0

        for task in self.tasks:
            if task.status == AgentStatus.COMPLETED:
                results[task.task_id] = task.result
                completed += 1
            elif task.status == AgentStatus.TIMED_OUT:
                errors[task.task_id] = task.error or "Timeout"
                timed_out += 1
            else:
                errors[task.task_id] = task.error or "Unknown error"
                failed += 1

        logger.info(
            f"Batch complete: {completed}/{len(self.tasks)} succeeded, "
            f"{failed} failed, {timed_out} timed out in {elapsed:.0f}ms"
        )

        return BatchResult(
            total=len(self.tasks),
            completed=completed,
            failed=failed,
            timed_out=timed_out,
            results=results,
            errors=errors,
            total_duration_ms=round(elapsed, 1),
        )

    async def _run_task(self, task: SubAgentTask) -> None:
        """Run a single task with timeout and retry logic."""
        async with self._semaphore:
            for attempt in range(task.retries + 1):
                task.status = AgentStatus.RUNNING
                start = time.monotonic()

                try:
                    task.result = await asyncio.wait_for(
                        task.fn(*task.args, **task.kwargs),
                        timeout=task.timeout_sec,
                    )
                    task.status = AgentStatus.COMPLETED
                    task.duration_ms = (time.monotonic() - start) * 1000
                    logger.debug(f"Task {task.task_id} completed in {task.duration_ms:.0f}ms")
                    return

                except asyncio.TimeoutError:
                    task.status = AgentStatus.TIMED_OUT
                    task.error = f"Timeout after {task.timeout_sec}s (attempt {attempt + 1})"
                    task.duration_ms = (time.monotonic() - start) * 1000
                    logger.warning(f"Task {task.task_id}: {task.error}")

                except Exception as e:
                    task.status = AgentStatus.FAILED
                    task.error = f"{type(e).__name__}: {e}"
                    task.duration_ms = (time.monotonic() - start) * 1000

                    if attempt < task.retries:
                        logger.warning(
                            f"Task {task.task_id} failed (attempt {attempt + 1}), retrying: {e}"
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(f"Task {task.task_id} failed permanently: {e}")

    def clear(self) -> None:
        """Clear all tasks for reuse."""
        self.tasks.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "max_concurrency": self.max_concurrency,
            "tasks": [
                {
                    "id": t.task_id,
                    "name": t.name,
                    "status": t.status.value,
                    "duration_ms": t.duration_ms,
                    "error": t.error,
                }
                for t in self.tasks
            ],
        }
