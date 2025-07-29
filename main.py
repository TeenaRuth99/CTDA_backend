# --- main.py ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ctda_unstructured import router as doc_router

app = FastAPI(title="CTDA Ask")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Router
app.include_router(doc_router)

# Main route (optional)
@app.get("/")
def read_root():
    return {"message": "CTDA RAG"}
