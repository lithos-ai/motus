"""
Configuration constants for Deep Research.
"""

import os

# ============================================================================
# ReAct Loop Configuration
# ============================================================================

# Maximum tool call iterations per research task
MAX_REACT_ITERATIONS = 15

# Maximum parallel research tasks
MAX_CONCURRENT_RESEARCH = 3

# Whether to compress research results before final report
ENABLE_RESEARCH_COMPRESSION = True

# ============================================================================
# API Configuration
# ============================================================================

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY") or ""
BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
MODEL = os.getenv("OPENAI_MODEL") or "gpt-5.2"
