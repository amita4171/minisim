"""
MiniSim API package -- production-grade FastAPI service.

Modules:
    app     -- FastAPI app creation, middleware, CORS
    routes  -- endpoint handlers
    models  -- Pydantic request/response models
    auth    -- API key authentication + rate limiting
    deps    -- dependency injection (Database, rate limiter)
"""
