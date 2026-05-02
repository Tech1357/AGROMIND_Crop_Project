from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from eligibility_engine import eligibility_engine

app = FastAPI(
    title="Niti-Setu Eligibility Engine",
    description="AI-powered RAG based farmer scheme eligibility checker",
    version="1.0.0"
)

# ==============================
# CORS (Required for Frontend)
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Root Endpoint
# ==============================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Niti-Setu Eligibility Engine is live 🚀",
        "endpoint": "POST /check-eligibility"
    }

# ==============================
# Health Check
# ==============================

@app.get("/health")
def health():
    return {"status": "healthy"}

# ==============================
# Request Model
# ==============================

class UserInput(BaseModel):

    state: str = Field(..., example="Telangana")
    district: str = Field(..., example="Nalgonda")
    land_size: float = Field(..., example=1.5)
    crop_type: str = Field(..., example="Paddy")
    category: str = Field(..., example="SC")
    annual_income: float = Field(..., example=350000)
    income_tax_payer: bool = Field(..., example=False)
    pension: float = Field(..., example=0)
    electricity_connection: bool = Field(..., example=False)

# ==============================
# Main API Endpoint
# ==============================

@app.post("/check-eligibility")
def check_eligibility(user: UserInput):

    try:

        result = eligibility_engine(user.dict())

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }