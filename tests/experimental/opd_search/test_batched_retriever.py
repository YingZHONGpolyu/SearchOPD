"""Verify cross-request batching, single native caller and HTTP compatibility."""

import asyncio
import sys
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/opd_search"))
from batched_retriever import QueryBatcher, create_app


class BatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_are_batched_and_serialized(self):
        calls = []
        active = 0

        def search(queries, topk):
            nonlocal active
            active += 1
            self.assertEqual(active, 1)
            calls.append((len(queries), topk, threading.get_ident()))
            time.sleep(0.002)
            active -= 1
            return [[{"contents": query}] for query in queries], [[float(topk)] for _ in queries]

        batcher = QueryBatcher(search, max_queries=8, wait_ms=5)
        worker = asyncio.create_task(batcher.run())
        try:
            result = await asyncio.gather(*(batcher.submit([str(i)], 3) for i in range(32)))
            self.assertEqual([row[0][0][0]["contents"] for row in result], [str(i) for i in range(32)])
            self.assertTrue(any(size > 1 for size, _, _ in calls))
            self.assertTrue(all(size <= 8 for size, _, _ in calls))
            self.assertEqual(len({thread for _, _, thread in calls}), 1)
        finally:
            await batcher.close(worker)

    async def test_different_topk_and_empty_requests(self):
        calls = []

        def search(queries, topk):
            calls.append(topk)
            return [[{"contents": q}] * topk for q in queries], [[1.0] * topk for _ in queries]

        batcher = QueryBatcher(search)
        worker = asyncio.create_task(batcher.run())
        try:
            one, three, empty = await asyncio.gather(
                batcher.submit(["one"], 1), batcher.submit(["three"], 3), batcher.submit([], 3)
            )
            self.assertEqual(len(one[0][0]), 1)
            self.assertEqual(len(three[0][0]), 3)
            self.assertEqual(empty, [])
            self.assertEqual(sorted(calls), [1, 3])
        finally:
            await batcher.close(worker)

    async def test_native_failure_reaches_all_waiters_and_worker_recovers(self):
        def search(queries, topk):
            if "bad" in queries:
                raise ValueError("backend failure")
            return [[{"contents": q}] for q in queries], [[1.0] for _ in queries]

        batcher = QueryBatcher(search)
        worker = asyncio.create_task(batcher.run())
        try:
            errors = await asyncio.gather(
                batcher.submit(["bad"], 3), batcher.submit(["also"], 3), return_exceptions=True
            )
            self.assertTrue(all(isinstance(error, ValueError) for error in errors))
            self.assertEqual((await batcher.submit(["good"], 3))[0][0][0]["contents"], "good")
        finally:
            await batcher.close(worker)


class HttpTests(unittest.TestCase):
    def test_search_r1_response_with_and_without_scores(self):
        def search(queries, topk):
            return [[{"contents": q}] for q in queries], [[0.5] for _ in queries]

        with TestClient(create_app(search)) as client:
            response = client.post("/retrieve", json={"queries": ["Paris"], "return_scores": True})
            self.assertEqual(response.json(), {"result": [[{"document": {"contents": "Paris"}, "score": 0.5}]]})
            response = client.post("/retrieve", json={"queries": ["Paris"], "return_scores": False})
            self.assertEqual(response.json(), {"result": [[{"contents": "Paris"}]]})
            self.assertEqual(client.post("/retrieve", json={"queries": []}).json(), {"result": []})
            self.assertEqual(client.get("/health").json()["queries"], 2)
            self.assertEqual(client.post("/retrieve", json={"queries": ["x"], "topk": 0}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
