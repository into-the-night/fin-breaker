# Main FastAPI app combining all agent routers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.utils.logging_config import setup_logging
from app.backend.utils.error_logging import setup_enhanced_logging
from app.backend.api.endpoints.orchestrator_api import router as orchestrator_router

# Set up enhanced logging for comprehensive error handling
setup_enhanced_logging(
    log_level="INFO",
    log_file="logs/orchestrator.log",
    enable_console=True,
    enable_structured=True,
    enable_audit=True,
    enable_performance=True
)

# Also call the original setup for compatibility
setup_logging()

app = FastAPI(title="Multi-Agent Finance Assistant")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include all routers
app.include_router(orchestrator_router)

@app.get("/")
def root():
    return {"status": "Multi-Agent Finance Assistant running"}

@app.get("/health/errors")
def get_error_summary():
    """Get comprehensive error summary and audit trail."""
    from app.backend.utils.error_logging import get_error_audit_summary
    from app.backend.agent.error_handling import get_error_handler
    
    # Get audit summary from logging handler
    audit_summary = get_error_audit_summary()
    
    # Get error handler summary
    error_handler_summary = get_error_handler().get_error_summary()
    
    return {
        "audit_trail": audit_summary,
        "error_handler": error_handler_summary
    }

@app.get("/health/performance")
def get_performance_metrics():
    """Get performance metrics and statistics."""
    from app.backend.utils.error_logging import get_performance_metrics
    
    return get_performance_metrics()

@app.post("/health/clear-audit")
def clear_audit_records():
    """Clear all stored audit records."""
    from app.backend.utils.error_logging import clear_audit_records
    
    clear_audit_records()
    return {"message": "Audit records cleared successfully"}
