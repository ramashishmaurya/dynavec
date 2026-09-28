"""FastAPI REST server wrapper for Dynavec.

Allows running dynavec as a standalone REST API server.
Requires the 'server' extra: pip install dynavec[server]
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .async_client import AsyncDynavec
from .exceptions import MissingDependencyError

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise MissingDependencyError("Dynavec Server", "fastapi", "server") from exc


class DocumentInput(BaseModel):
    id: str
    text: str | None = None
    vector: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: int | None = None


class UpsertRequest(BaseModel):
    documents: list[DocumentInput]
    namespace: str = "default"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    namespace: str = "default"
    filter: dict[str, Any] | None = None
    rerank: str | None = None
    mmr_lambda: float = 0.5


class DeleteRequest(BaseModel):
    ids: list[str]
    namespace: str = "default"


def create_app(client: AsyncDynavec) -> FastAPI:
    """Create a FastAPI application wrapping the provided AsyncDynavec client."""
    app = FastAPI(
        title="Dynavec Server",
        description="REST API wrapper for dynavec hybrid vector database.",
        version="0.6.0",
    )

    @app.post("/search")
    async def search(req: SearchRequest) -> JSONResponse:
        try:
            results = await client.search(
                query=req.query,
                top_k=req.top_k,
                namespace=req.namespace,
                filter=req.filter,
            )
            return JSONResponse({"results": [r.to_dict() for r in results]})
        except Exception as e:
            logging.error(f"Search error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/upsert")
    async def upsert(req: UpsertRequest) -> JSONResponse:
        from .models import Document

        docs = []
        for d in req.documents:
            try:
                docs.append(Document(**d.model_dump()))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
                
        try:
            res = await client.upsert(docs, namespace=req.namespace)
            return JSONResponse({"count": res.count, "ids": res.ids})
        except Exception as e:
            logging.error(f"Upsert error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/delete")
    async def delete(req: DeleteRequest) -> JSONResponse:
        try:
            if hasattr(client, "delete"):
                await client.delete(req.ids, namespace=req.namespace) # type: ignore[attr-defined]
            return JSONResponse({"success": True})
        except Exception as e:
            logging.error(f"Delete error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app
