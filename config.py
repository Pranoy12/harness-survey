import os
from typing import Optional
from pydantic import BaseModel

class Config(BaseModel):
    # LLM Provider Configuration
    PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")
    API_KEY: str = os.getenv("LLM_API_KEY", "")
    
    # Must be None for Google AI Studio / Gemini; set only for custom OpenAI-compatible endpoints
    BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", None)
    
    # LiteLLM expects 'gemini/' prefix for Google AI Studio models
    MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "deepseek/deepseek-flash")

    # Hardware & Sandbox Safety Caps (Optimized for 8 GB RAM)
    MAX_DOCKER_MEMORY: str = "2g"
    MAX_DOCKER_SWAP: str = "4g"
    COMMAND_TIMEOUT_SECONDS: int = 45
    MAX_AGENT_TURNS: int = 25

    # Budget & Pacing (Free tier Gemini allows 15 RPM - 4.0s delay prevents 429 errors)
    REQUEST_DELAY_SECONDS: float = 0.2 #USe 4.0 for free tier models
    MAX_CONTEXT_TOKENS: int = 32000
    MAX_BUDGET_USD: float = 0.5

CONFIG = Config()