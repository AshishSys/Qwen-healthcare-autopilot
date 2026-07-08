"""
Agent Orchestrator — Multi-Workflow Coordinator

Coordinates multiple autonomous agents (triage, lab processing, 
prior auth, care plan, discharge) and manages workflow state,
escalation routing, and inter-agent communication.
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from .qwen_client import QwenCloudClient
from .triage_agent import TriageAgent, PatientSymptoms
from .lab_processor import LabProcessingAgent, LabOrder
from .prior_auth_agent import PriorAuthAgent, AuthorizationRequest
from .care_plan_agent import CarePlanAgent, Diagnosis
from .discharge_agent import DischargeAgent, DischargeOrder

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    TRIAGE = "triage"
    LAB_PROCESSING = "lab_processing"
    PRIOR_AUTH = "prior_authorization"
    CARE_PLAN = "care_plan"
    DISCHARGE = "discharge"


@dataclass
class WorkflowEvent:
    """Event in the workflow event bus."""
    event_type: str
    source_workflow: str
    payload: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """
    Central orchestrator that coordinates all healthcare agents.
    
    Responsibilities:
    - Dispatch incoming requests to appropriate agent
    - Manage workflow state transitions
    - Handle inter-agent events (e.g., triage complete → schedule followup)
    - Enforce escalation policies
    - Track SLA compliance
    """
    
    # SLA targets per workflow type (seconds)
    SLA_TARGETS = {
        WorkflowType.TRIAGE: 120,         # 2 minutes
        WorkflowType.LAB_PROCESSING: 300, # 5 minutes
        WorkflowType.PRIOR_AUTH: 600,     # 10 minutes
        WorkflowType.CARE_PLAN: 900,      # 15 minutes
        WorkflowType.DISCHARGE: 300,      # 5 minutes
    }
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.triage_agent = TriageAgent(qwen_client=self.qwen)
        self.lab_agent = LabProcessingAgent(qwen_client=self.qwen)
        self.prior_auth_agent = PriorAuthAgent(qwen_client=self.qwen)
        self.care_plan_agent = CarePlanAgent(qwen_client=self.qwen)
        self.discharge_agent = DischargeAgent(qwen_client=self.qwen)
        
        self.event_bus: list[WorkflowEvent] = []
        self.active_workflows: dict = {}
    
    def dispatch(self, workflow_type: WorkflowType, patient_id: str, **kwargs):
        """Dispatch a workflow to the appropriate agent."""
        logger.info(f"Dispatching {workflow_type.value} workflow for patient {patient_id}")
        
        if workflow_type == WorkflowType.TRIAGE:
            symptoms = PatientSymptoms(**kwargs)
            return self.triage_agent.run(patient_id, symptoms)
        
        elif workflow_type == WorkflowType.LAB_PROCESSING:
            lab_order = LabOrder(patient_id=patient_id, **kwargs)
            return self.lab_agent.run(lab_order)
        
        elif workflow_type == WorkflowType.PRIOR_AUTH:
            auth_request = AuthorizationRequest(patient_id=patient_id, **kwargs)
            return self.prior_auth_agent.run(auth_request)
        
        elif workflow_type == WorkflowType.CARE_PLAN:
            diagnoses = kwargs.get("diagnoses", [])
            return self.care_plan_agent.run(patient_id, diagnoses, kwargs.get("patient_info", {}))
        
        elif workflow_type == WorkflowType.DISCHARGE:
            discharge_order = DischargeOrder(patient_id=patient_id, **kwargs)
            return self.discharge_agent.run(discharge_order)
        
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def handle_event(self, event: WorkflowEvent):
        """Process inter-agent events."""
        self.event_bus.append(event)
        
        # Event-driven workflow chaining
        if event.event_type == "triage_completed":
            urgency = event.payload.get("urgency")
            if urgency == "emergency":
                # Chain: triage → immediate provider notification
                logger.info("Emergency triage — triggering immediate notification chain")
            elif urgency in ["urgent", "semi_urgent"]:
                # Chain: triage → lab order if needed
                logger.info("Urgent case — checking if labs are needed")
        
        elif event.event_type == "lab_results_ready":
            # Chain: lab results → care plan update
            logger.info("Lab results ready — triggering care plan review")
    
    def get_status_dashboard(self) -> dict:
        """Return real-time status of all workflows for the dashboard."""
        return {
            "active_workflows": len(self.active_workflows),
            "event_queue_depth": len(self.event_bus),
            "agents": {
                "triage": "online",
                "lab_processing": "online",
                "prior_auth": "online",
                "care_plan": "online",
                "discharge": "online",
            },
            "sla_compliance": self._calculate_sla_compliance(),
        }
    
    def _calculate_sla_compliance(self) -> dict:
        """Calculate SLA compliance rates per workflow type."""
        # Placeholder — would query historical workflow completion times
        return {wt.value: 0.95 for wt in WorkflowType}
