#!/usr/bin/env python3

"""
Model Manager Startup Script
"""

import uvicorn
from main import app
from config import settings

if __name__ == "__main__":
    print("🚀 Starting Model Manager...")
    print(f"📍 Host: {settings.host}")
    print(f"🔌 Port: {settings.port}")
    print(f"🐛 Debug: {settings.debug}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )