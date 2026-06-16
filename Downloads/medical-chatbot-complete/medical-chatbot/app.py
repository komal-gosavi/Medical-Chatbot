from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import os
from src.config import load_config
from src.rag_chain import create_rag_chain

# ============================================================
# 1. APP INITIALIZATION
# ============================================================
# In Flask: app = Flask(__name__)
# In FastAPI: app = FastAPI()
# Key difference: FastAPI is async-first, Flask is sync

app = FastAPI(
    title="Medical Chatbot API",
    description="RAG-based medical question answering",
    version="1.0.0"
)

# ============================================================
# 2. CORS SETUP (Important for frontend in separate branch)
# ============================================================
# This allows your frontend branch to make requests to this backend
# Without CORS, you'll get "blocked by browser" errors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3. STATIC FILES & TEMPLATES (Optional for backend branch)
# ============================================================
# For backend branch: comment these out (you only serve API)
# For main branch: keep them (serve both API + frontend)

static_path = os.path.join(os.path.dirname(__file__), "static")
templates_path = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

if os.path.exists(templates_path):
    templates = Jinja2Templates(directory=templates_path)
else:
    templates = None

# ============================================================
# 4. CONFIGURATION & RAG CHAIN (Same as Flask)
# ============================================================

config = None
try:
    config = load_config()
    print("✓ Configuration loaded successfully")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
    print("  Please ensure PINECONE_API_KEY and GROQ_API_KEY are set in .env file")

rag_chain_instance = None

def get_rag_chain():
    """Lazy initialize RAG chain on first use (same logic as Flask)."""
    global rag_chain_instance
    if config is None:
        raise RuntimeError("Configuration not loaded. Please check your API keys.")
    if rag_chain_instance is None:
        rag_chain_instance = create_rag_chain(config)
    return rag_chain_instance

# ============================================================
# 5. ROUTES
# ============================================================

# ROUTE 1: GET /
# Flask: @app.route("/")
# FastAPI: @app.get("/")
# This serves the HTML interface if templates exist

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the chat interface."""
    if templates is None:
        return "<h1>Medical Chatbot API</h1><p>API is running. Frontend is in a different branch.</p>"
    
    return templates.get_template("chat.html").render(request=request)


# ROUTE 2: GET or POST /chat
# Flask: 
#   @app.route("/get", methods=["GET", "POST"])
#   user_msg = request.form.get("msg")
#
# FastAPI:
#   @app.post("/chat")
#   user_msg: str = Form(...)
#
# Key differences:
# - Separate @app.post() and @app.get() decorators
# - Parameters are defined in function signature
# - Pydantic automatically validates/converts types

@app.post("/chat")
async def chat(msg: str = Form(...)):
    """
    Handle user chat messages and return AI responses.
    
    Args:
        msg: User message from form data
    
    Returns:
        JSON response with the AI's answer
    
    Raises:
        HTTPException: If message is empty or processing fails
    """
    # Validation (FastAPI does this automatically with Pydantic)
    if not msg or not msg.strip():
        raise HTTPException(status_code=400, detail="No message provided")
    
    print(f"User input: {msg}")
    
    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke(msg)
        print(f"Response: {response}")
        
        # In FastAPI: just return a dict, it's automatically converted to JSON
        # In Flask: had to use str(response) or jsonify()
        return {"response": str(response)}
    
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing message: {str(e)}"
        )


# ROUTE 3: New - Health check endpoint (optional but useful for deployment)
@app.get("/health")
async def health_check():
    """
    Health check endpoint for deployment monitoring.
    Render.com and other platforms use this to check if app is alive.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "rag_chain_loaded": rag_chain_instance is not None
    }


# ============================================================
# 6. APP STARTUP & SHUTDOWN (Optional)
# ============================================================
# Startup message - no heavy operations here
# The RAG chain will be initialized on first use (lazy loading)

print("🚀 Medical Chatbot API initialized")


# ============================================================
# 7. MAIN BLOCK (Usually not needed for FastAPI + Uvicorn)
# ============================================================
# This is optional - uvicorn will run it automatically
# But useful for local testing with: python app.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # "file:app_object"
        host=config.get("HOST", "0.0.0.0"),
        port=config.get("PORT", 8000),
        reload=config.get("DEBUG", False)  # Auto-reload in debug mode
    )