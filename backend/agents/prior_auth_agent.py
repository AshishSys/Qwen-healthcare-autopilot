"""
Prior Authorization Agent — Autonomous Insurance Authorization Pipeline

This agent handles the end-to-end prior authorization workflow:
1. Extracts procedure details from provider order
2. Identifies payer-specific requirements and rules
3. Compiles clinical evidence from patient records
4. Submits authorization request to payer
5. Monitors approval status
6. Auto-appeals with additional evidence if denied

Uses Qwen-max for clinical evidence compilation and reasoning.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from .qwen_client import QwenCloudClient, AgentResponse

logger = logging.getLogger(__name__)


# ============================================================
# Data Models
# ============================================================

class AuthStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    DENIED = "denied"
    APPEALED = "appealed"
    APPEAL_APPROVED = "appeal_approved"
    APPEAL_DENIED = "appeal_denied"
    INFO_REQUESTED = "info_requested"


class DenialReason(str, Enum):
    MEDICAL_NECESSITY = "medical_necessity"
    STEP_THERAPY = "step_therapy_not_completed"
    OUT_OF_NETWORK = "out_of_network"
    EXPERIMENTAL = "experimental_treatment"
    DOCUMENTATION = "insufficient_documentation"
    FORMULARY = "not_on_formulary"


@dataclass
class AuthorizationRequest:
    """Prior authorization request details."""
    request_id: str
    patient_id: str
    provider_id: str
    payer_id: str
    procedure_code: str          # CPT code
    procedure_name: str
    diagnosis_codes: list[str]   # ICD-10 codes
    clinical_notes: str = ""
    urgency: str = "standard"    # standard / urgent / retrospective
    requested_date: str = ""
    facility: str = ""


@dataclass
class AuthorizationResult:
    """Result of the authorization workflow."""
    request_id: str
    status: AuthStatus
    determination: str = ""
    authorization_number: str = ""
    valid_from: str = ""
    valid_to: str = ""
    approved_units: int = 0
    denial_reason: Optional[DenialReason] = None
    appeal_submitted: bool = False
    appeal_evidence: list[str] = field(default_factory=list)
    audit_trail: list[dict] = field(default_factory=list)


# ============================================================
# Prior Auth Tools
# ============================================================

PRIOR_AUTH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_payer_requirements",
            "description": "Look up payer-specific prior authorization requirements for a given procedure code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payer_id": {"type": "string"},
                    "procedure_code": {"type": "string", "description": "CPT code"},
                    "diagnosis_code": {"type": "string", "description": "Primary ICD-10 code"}
                },
                "required": ["payer_id", "procedure_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compile_clinical_evidence",
            "description": "Gather clinical evidence from patient records to support medical necessity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "procedure_code": {"type": "string"},
                    "diagnosis_codes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "evidence_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of evidence to gather: clinical_notes, lab_results, imaging, prior_treatments, guidelines"
                    }
                },
                "required": ["patient_id", "procedure_code", "diagnosis_codes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_step_therapy",
            "description": "Verify if required step therapy (trying cheaper alternatives first) has been completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "medication_or_procedure": {"type": "string"},
                    "payer_id": {"type": "string"}
                },
                "required": ["patient_id", "medication_or_procedure", "payer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_auth_request",
            "description": "Submit the prior authorization request to the payer with compiled evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payer_id": {"type": "string"},
                    "patient_id": {"type": "string"},
                    "provider_id": {"type": "string"},
                    "procedure_code": {"type": "string"},
                    "diagnosis_codes": {"type": "array", "items": {"type": "string"}},
                    "clinical_justification": {"type": "string"},
                    "supporting_documents": {"type": "array", "items": {"type": "string"}},
                    "urgency": {"type": "string", "enum": ["standard", "urgent", "retrospective"]}
                },
                "required": ["payer_id", "patient_id", "provider_id", "procedure_code", "clinical_justification"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_auth_status",
            "description": "Check the current status of a submitted authorization request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "auth_reference_number": {"type": "string"},
                    "payer_id": {"type": "string"}
                },
                "required": ["auth_reference_number", "payer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_appeal",
            "description": "Submit an appeal for a denied authorization with additional evidence and clinical argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "auth_reference_number": {"type": "string"},
                    "denial_reason": {"type": "string"},
                    "appeal_argument": {"type": "string", "description": "Clinical argument for why denial should be overturned"},
                    "additional_evidence": {"type": "array", "items": {"type": "string"}},
                    "peer_to_peer_requested": {"type": "boolean", "description": "Request physician-to-physician review"}
                },
                "required": ["auth_reference_number", "denial_reason", "appeal_argument"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_provider_auth_result",
            "description": "Notify the ordering provider about the authorization decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string"},
                    "patient_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["approved", "denied", "info_requested"]},
                    "details": {"type": "string"},
                    "next_steps": {"type": "string"}
                },
                "required": ["provider_id", "patient_id", "status", "details"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class PriorAuthToolExecutor:
    """Executes prior authorization tool calls."""
    
    # Simulated payer requirements database
    PAYER_RULES = {
        "BCBS": {
            "27447": {  # Total Knee Replacement
                "requires_auth": True,
                "step_therapy": ["physical_therapy_6_weeks", "nsaids_trial", "corticosteroid_injection"],
                "documentation": ["x_ray_or_mri", "clinical_notes", "conservative_treatment_history"],
                "turnaround_days": 5,
            },
            "70553": {  # MRI Brain with contrast
                "requires_auth": True,
                "step_therapy": [],
                "documentation": ["clinical_notes", "neurological_exam"],
                "turnaround_days": 3,
            }
        },
        "AETNA": {
            "27447": {
                "requires_auth": True,
                "step_therapy": ["physical_therapy_12_weeks", "weight_management"],
                "documentation": ["x_ray", "bmi_documentation", "functional_assessment"],
                "turnaround_days": 7,
            }
        }
    }
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        handlers = {
            "check_payer_requirements": self._check_payer_requirements,
            "compile_clinical_evidence": self._compile_clinical_evidence,
            "check_step_therapy": self._check_step_therapy,
            "submit_auth_request": self._submit_auth_request,
            "check_auth_status": self._check_auth_status,
            "submit_appeal": self._submit_appeal,
            "notify_provider_auth_result": self._notify_provider,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(**arguments)
    
    def _check_payer_requirements(self, payer_id: str, procedure_code: str, diagnosis_code: str = None) -> dict:
        payer_rules = self.PAYER_RULES.get(payer_id, {})
        proc_rules = payer_rules.get(procedure_code)
        
        if not proc_rules:
            return {
                "payer_id": payer_id,
                "procedure_code": procedure_code,
                "requires_auth": False,
                "message": "No prior authorization required for this procedure with this payer."
            }
        
        return {
            "payer_id": payer_id,
            "procedure_code": procedure_code,
            "requires_auth": proc_rules["requires_auth"],
            "step_therapy_required": proc_rules["step_therapy"],
            "documentation_required": proc_rules["documentation"],
            "expected_turnaround_days": proc_rules["turnaround_days"],
        }
    
    def _compile_clinical_evidence(
        self, patient_id: str, procedure_code: str, diagnosis_codes: list[str],
        evidence_types: list[str] = None
    ) -> dict:
        """Compile clinical evidence (simulated — queries EHR in production)."""
        return {
            "patient_id": patient_id,
            "evidence_compiled": True,
            "documents": [
                {"type": "clinical_notes", "date": "2024-12-15", "summary": "Progressive knee pain, failed conservative management"},
                {"type": "imaging", "date": "2024-12-01", "summary": "X-ray: Grade 4 osteoarthritis, bone-on-bone"},
                {"type": "prior_treatment", "date": "2024-06-01", "summary": "6 weeks PT, NSAIDs x 3 months, cortisone injection x 2"},
                {"type": "functional_assessment", "date": "2024-12-10", "summary": "Unable to walk >1 block, uses assistive device"},
            ],
            "clinical_justification": f"Patient has failed conservative management for diagnosis codes {diagnosis_codes}. "
                                     f"Imaging confirms severe disease. Procedure {procedure_code} is medically necessary.",
        }
    
    def _check_step_therapy(self, patient_id: str, medication_or_procedure: str, payer_id: str) -> dict:
        """Check if step therapy requirements are met."""
        return {
            "patient_id": patient_id,
            "step_therapy_completed": True,
            "steps_completed": [
                {"step": "physical_therapy", "completed": True, "date": "2024-09-01", "duration": "6 weeks"},
                {"step": "nsaids_trial", "completed": True, "date": "2024-07-01", "duration": "3 months"},
                {"step": "corticosteroid_injection", "completed": True, "date": "2024-10-15", "count": 2},
            ],
            "all_requirements_met": True,
        }
    
    def _submit_auth_request(
        self, payer_id: str, patient_id: str, provider_id: str,
        procedure_code: str, clinical_justification: str,
        diagnosis_codes: list[str] = None, supporting_documents: list[str] = None,
        urgency: str = "standard"
    ) -> dict:
        """Submit auth request (simulates payer API call)."""
        ref_number = f"AUTH-{payer_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "auth_reference_number": ref_number,
            "status": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
            "expected_decision_by": (datetime.utcnow() + timedelta(days=5)).isoformat(),
            "payer_id": payer_id,
            "procedure_code": procedure_code,
            "message": f"Authorization request {ref_number} submitted successfully to {payer_id}.",
        }
    
    def _check_auth_status(self, auth_reference_number: str, payer_id: str) -> dict:
        """Check authorization status (simulated — would poll payer API)."""
        # Simulate approval for demo
        return {
            "auth_reference_number": auth_reference_number,
            "status": "approved",
            "determination_date": datetime.utcnow().isoformat(),
            "authorization_number": f"APPR-{auth_reference_number[-6:]}",
            "valid_from": datetime.utcnow().isoformat(),
            "valid_to": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "approved_units": 1,
        }
    
    def _submit_appeal(
        self, auth_reference_number: str, denial_reason: str, appeal_argument: str,
        additional_evidence: list[str] = None, peer_to_peer_requested: bool = False
    ) -> dict:
        """Submit appeal for denied authorization."""
        return {
            "appeal_id": f"APPEAL-{auth_reference_number[-6:]}-{datetime.utcnow().strftime('%H%M%S')}",
            "original_auth": auth_reference_number,
            "denial_reason": denial_reason,
            "appeal_argument_summary": appeal_argument[:200],
            "additional_evidence_count": len(additional_evidence or []),
            "peer_to_peer_requested": peer_to_peer_requested,
            "submitted_at": datetime.utcnow().isoformat(),
            "status": "appeal_under_review",
            "expected_decision_days": 30,
        }
    
    def _notify_provider(
        self, provider_id: str, patient_id: str, status: str, details: str, next_steps: str = ""
    ) -> dict:
        """Notify provider of authorization result."""
        return {
            "notification_id": f"NOTIF-AUTH-{datetime.utcnow().strftime('%H%M%S')}",
            "provider_id": provider_id,
            "patient_id": patient_id,
            "status": status,
            "details": details,
            "next_steps": next_steps,
            "sent_at": datetime.utcnow().isoformat(),
        }


# ============================================================
# Prior Authorization Agent
# ============================================================

class PriorAuthAgent:
    """
    Autonomous Prior Authorization Agent.
    
    Handles the complete auth workflow including evidence compilation,
    submission, status tracking, and auto-appeal on denial.
    """
    
    SYSTEM_PROMPT = """You are an autonomous prior authorization agent. Your role is to
handle the complete insurance authorization process for medical procedures.

WORKFLOW:
1. check_payer_requirements — What does this payer need for this procedure?
2. check_step_therapy — Has patient completed required alternative treatments?
3. compile_clinical_evidence — Gather supporting documentation
4. submit_auth_request — Submit with clinical justification
5. check_auth_status — Monitor for decision
6. If DENIED: submit_appeal with additional evidence and stronger argument
7. notify_provider_auth_result — Inform provider of outcome

RULES:
- Always compile maximum evidence before initial submission (higher first-pass approval)
- If step therapy is incomplete, notify provider BEFORE submitting (it will be auto-denied)
- On denial, analyze the specific reason and tailor the appeal argument
- Request peer-to-peer review for medical necessity denials
- Track turnaround times and escalate if SLA is breached
- Never fabricate or exaggerate clinical evidence
"""
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.tool_executor = PriorAuthToolExecutor()
    
    def run(self, request: AuthorizationRequest) -> AuthorizationResult:
        """Execute the full prior authorization workflow."""
        logger.info(f"Starting prior auth: {request.request_id} for {request.procedure_name}")
        
        context = self._build_context(request)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        
        response = self.qwen.function_call_loop(
            messages=messages,
            tools=PRIOR_AUTH_TOOLS,
            tool_executor=self.tool_executor.execute,
            task_type="prior_auth",
            max_iterations=10,
        )
        
        return AuthorizationResult(
            request_id=request.request_id,
            status=AuthStatus.APPROVED,  # Would parse from actual response
            determination=response.content,
        )
    
    def _build_context(self, req: AuthorizationRequest) -> str:
        return f"""PRIOR AUTHORIZATION REQUEST

Request ID: {req.request_id}
Patient ID: {req.patient_id}
Provider: {req.provider_id}
Payer: {req.payer_id}

PROCEDURE:
  Code: {req.procedure_code}
  Name: {req.procedure_name}
  Diagnosis: {', '.join(req.diagnosis_codes)}

CLINICAL NOTES: {req.clinical_notes}

URGENCY: {req.urgency}
REQUESTED DATE: {req.requested_date}

Please process this authorization request through the complete workflow.
Compile evidence, verify requirements, submit, and track to resolution."""
