"""Batch HTTP queries while keeping native E5/FAISS calls on one worker thread."""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = Field(default=None, ge=1, le=100)
    return_scores: bool = False


class QueryBatcher:
    def __init__(self, search, max_queries=32, wait_ms=10):
        if max_queries < 1 or wait_ms < 0:
            raise ValueError("max_queries must be positive and wait_ms nonnegative")
        self.search = search
        self.max_queries = max_queries
        self.wait_seconds = wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=8192)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="e5-faiss")
        self.batches = self.queries = 0

    async def submit(self, queries, topk):
        futures = []
        loop = asyncio.get_running_loop()
        for query in queries:
            future = loop.create_future()
            await self.queue.put((query, topk, future))
            futures.append(future)
        return await asyncio.gather(*futures)

    async def run(self):
        loop = asyncio.get_running_loop()
        stopping = False
        while not stopping:
            first = await self.queue.get()
            if first is None:
                break
            items = [first]
            deadline = loop.time() + self.wait_seconds
            while len(items) < self.max_queries:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if item is None:
                    stopping = True
                    break
                items.append(item)
            groups = defaultdict(list)
            for item in items:
                if not item[2].cancelled():
                    groups[item[1]].append(item)
            for topk, group in groups.items():
                try:
                    documents, scores = await loop.run_in_executor(
                        self.executor, self.search, [item[0] for item in group], topk
                    )
                    if len(documents) != len(group) or len(scores) != len(group):
                        raise ValueError("Retriever returned a different number of queries")
                    for item, docs, values in zip(group, documents, scores, strict=True):
                        if not item[2].done():
                            item[2].set_result((docs, values))
                    self.batches += 1
                    self.queries += len(group)
                except Exception as exc:
                    LOG.exception("E5/FAISS batch failed")
                    for item in group:
                        if not item[2].done():
                            item[2].set_exception(exc)

    async def close(self, worker):
        await self.queue.put(None)
        await worker
        self.executor.shutdown(wait=True)


def create_app(search, default_topk=3, max_queries=32, wait_ms=10):
    @asynccontextmanager
    async def lifespan(app):
        batcher = QueryBatcher(search, max_queries, wait_ms)
        app.state.batcher = batcher
        worker = asyncio.create_task(batcher.run())
        try:
            yield
        finally:
            await batcher.close(worker)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        batcher = app.state.batcher
        return {
            "status": "ok",
            "batches": batcher.batches,
            "queries": batcher.queries,
            "queued_queries": batcher.queue.qsize(),
        }

    @app.post("/retrieve")
    async def retrieve(request: QueryRequest):
        rows = await app.state.batcher.submit(request.queries, request.topk or default_topk)
        result = []
        for documents, scores in rows:
            if request.return_scores:
                result.append(
                    [{"document": document, "score": score} for document, score in zip(documents, scores, strict=True)]
                )
            else:
                result.append(documents)
        return {"result": result}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", type=Path, required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--batch-queries", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.batch_queries < 1 or args.batch_wait_ms < 0 or args.threads < 1:
        parser.error("Invalid batch/worker settings")

    import faiss
    import uvicorn

    sys.path.insert(0, str(args.search_r1_root))
    from search_r1.search.retrieval_server import Config, get_retriever

    config = Config(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        # Load the original index first; cloning below keeps FP32 explicitly.
        faiss_gpu=False,
        retrieval_model_path=args.retriever_model,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.batch_queries,
    )
    retriever = get_retriever(config)
    if args.faiss_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("GPU FAISS is not installed; run install_retriever_gpu.sh")
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(512 * 1024 * 1024)
        options = faiss.GpuClonerOptions()
        options.useFloat16 = False
        retriever.index = faiss.index_cpu_to_gpu(resources, 0, retriever.index, options)
        # Keep the scratch resources alive as long as the GPU index.
        retriever._opd_gpu_resources = resources
        LOG.warning("FAISS index is resident on GPU in FP32; ntotal=%s", retriever.index.ntotal)

    def search(queries, topk):
        # OpenMP settings are thread-local; set inside the one native caller.
        faiss.omp_set_num_threads(args.threads)
        return retriever.batch_search(queries, num=topk, return_score=True)

    uvicorn.run(create_app(search, args.topk, args.batch_queries, args.batch_wait_ms), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
