# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import objectives_api, sessions_api, analytics_api, mdx_api

app = FastAPI(title="MCAD Prototype v2", version="0.1.0")

# Pour le développement : CORS ouvert (à adapter si besoin)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routers
app.include_router(objectives_api.router)
app.include_router(sessions_api.router)
app.include_router(analytics_api.router)
app.include_router(mdx_api.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
