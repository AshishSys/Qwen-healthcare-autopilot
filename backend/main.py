"""
FastAPI Application — Healthcare Autopilot API

Main entry point for the Healthcare Autopilot backend.
Exposes REST endpoints for:
- Patient symptom submission and triage
- Workflow status tracking
- Agent audit logs
- Provider escalation queue
"""

import os
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.qwen_client import QwenCloudClient
from agents.triage_agent import TriageAgent, PatientSymptoms, TriageWorkflowState

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    logger.info("🏥 Healthcare Autopilot starting...")
    logger.info(f"   Qwen Cloud endpoint: {os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}")
    
    # Initialize Qwen client (validates API key)
    try:
        app.state.qwen_client = QwenCloudClient()
        app.state.triage_agent = TriageAgent(qwen_client=app.state.qwen_client)
        logger.info("   ✅ Qwen Cloud client initialized")
    except ValueError as e:
        logger.warning(f"   ⚠️  Qwen Cloud not configured: {e}")
        app.state.qwen_client = None
        app.state.triage_agent = None
    
    # In-memory workflow store (replace with RDS in production)
    app.state.workflows = {}
    
    logger.info("🏥 Healthcare Autopilot ready!")
    yield
    logger.info("🏥 Healthcare Autopilot shutting down...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Healthcare Autopilot",
    description="Autonomous AI agent for healthcare workflow automation powered by Qwen Cloud",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class TriageRequest(BaseModel):
    """Patient triage submission."""
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chief_complaint: str = Field(..., description="Primary reason for seeking care")
    symptoms: list[str] = Field(..., description="List of symptoms")
    duration: str = Field(..., description="How long symptoms have been present")
    severity: int = Field(..., ge=1, le=10, description="Severity scale 1-10")
    onset: str = Field(default="gradual", description="sudden or gradual")
    associated_symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    vital_signs: dict = Field(default_factory=dict)
    age: int | None = None
    gender: str | None = None


class TriageResponse(BaseModel):
    """Response after triage submission."""
    workflow_id: str
    status: str
    message: str


class WorkflowStatusResponse(BaseModel):
    """Workflow status check response."""
    workflow_id: str
    status: str
    started_at: str | None
    completed_at: str | None
    result_summary: str | None = None


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Healthcare Autopilot",
        "version": "1.0.0",
        "status": "running",
        "powered_by": "Qwen Cloud (Alibaba Cloud Model Studio)",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "qwen_cloud": "connected" if app.state.qwen_client else "not_configured",
        "active_workflows": len(app.state.workflows),
        "uptime": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/triage", response_model=TriageResponse)
async def submit_triage(request: TriageRequest, background_tasks: BackgroundTasks):
    """
    Submit patient symptoms for autonomous triage.
    
    The agent will:
    1. Assess clinical risk
    2. Check for red flags
    3. Route to appropriate department
    4. Schedule appointment
    5. Notify patient
    
    Returns immediately with workflow_id for status tracking.
    """
    if not app.state.triage_agent:
        raise HTTPException(status_code=503, detail="Qwen Cloud not configured. Set DASHSCOPE_API_KEY.")
    
    # Convert request to domain model
    symptoms = PatientSymptoms(
        chief_complaint=request.chief_complaint,
        symptoms=request.symptoms,
        duration=request.duration,
        severity=request.severity,
        onset=request.onset,
        associated_symptoms=request.associated_symptoms,
        medical_history=request.medical_history,
        current_medications=request.current_medications,
        allergies=request.allergies,
        vital_signs=request.vital_signs,
        age=request.age,
        gender=request.gender,
    )
    
    # Generate workflow ID
    workflow_id = f"TRIAGE-{request.patient_id[:8]}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    # Store initial state
    app.state.workflows[workflow_id] = {
        "status": "processing",
        "started_at": datetime.utcnow().isoformat(),
        "patient_id": request.patient_id,
    }
    
    # Run triage in background (non-blocking)
    background_tasks.add_task(
        _run_triage_workflow,
        workflow_id=workflow_id,
        patient_id=request.patient_id,
        symptoms=symptoms,
    )
    
    return TriageResponse(
        workflow_id=workflow_id,
        status="processing",
        message="Triage workflow initiated. The autonomous agent is assessing your symptoms.",
    )


@app.get("/api/v1/triage/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Check the status of a triage workflow."""
    workflow = app.state.workflows.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=workflow.get("status", "unknown"),
        started_at=workflow.get("started_at"),
        completed_at=workflow.get("completed_at"),
        result_summary=workflow.get("result_summary"),
    )


@app.get("/api/v1/workflows")
async def list_workflows():
    """List all active and recent workflows."""
    return {
        "workflows": [
            {"workflow_id": wid, **data}
            for wid, data in app.state.workflows.items()
        ],
        "total": len(app.state.workflows),
    }


@app.get("/api/v1/escalations")
async def list_escalations():
    """List cases escalated to human providers."""
    escalated = {
        wid: data for wid, data in app.state.workflows.items()
        if data.get("status") == "escalated"
    }
    return {"escalations": escalated, "count": len(escalated)}


# ============================================================
# Background Workflow Execution
# ============================================================

async def _run_triage_workflow(workflow_id: str, patient_id: str, symptoms: PatientSymptoms):
    """Execute triage agent in background."""
    try:
        result = app.state.triage_agent.run(patient_id, symptoms)
        
        app.state.workflows[workflow_id].update({
            "status": result.status,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "result_summary": result.audit_log[-1].get("agent_response", "") if result.audit_log else None,
            "audit_log": result.audit_log,
        })
        
    except Exception as e:
        logger.error(f"Background triage failed: {e}")
        app.state.workflows[workflow_id].update({
            "status": "error",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
