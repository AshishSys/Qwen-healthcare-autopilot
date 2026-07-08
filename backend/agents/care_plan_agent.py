"""
Care Plan Generation Agent — Autonomous Evidence-Based Care Planning

This agent handles care plan creation and management:
1. Analyzes diagnosis and patient history
2. Retrieves evidence-based clinical guidelines (via RAG)
3. Generates personalized care plan
4. Checks drug-drug interactions
5. Verifies insurance coverage for proposed treatments
6. Schedules follow-up milestones
7. Monitors patient adherence

Uses Qwen-max for clinical reasoning and guideline interpretation.
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

class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    NOT_STARTED = "not_started"


class InteractionSeverity(str, Enum):
    CONTRAINDICATED = "contraindicated"  # Must not combine
    SEVERE = "severe"                     # Use only if benefit outweighs risk
    MODERATE = "moderate"                 # Monitor closely
    MILD = "mild"                         # Awareness needed


@dataclass
class Diagnosis:
    """Patient diagnosis for care planning."""
    code: str           # ICD-10
    description: str
    severity: str       # mild / moderate / severe
    onset_date: str
    is_primary: bool = True


@dataclass
class CareGoal:
    """Individual goal within a care plan."""
    goal_id: str
    description: str
    target_date: str
    metrics: list[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.NOT_STARTED
    interventions: list[str] = field(default_factory=list)


@dataclass
class CarePlan:
    """Complete patient care plan."""
    plan_id: str
    patient_id: str
    diagnoses: list[Diagnosis]
    goals: list[CareGoal] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    lifestyle_recommendations: list[str] = field(default_factory=list)
    follow_up_schedule: list[dict] = field(default_factory=list)
    monitoring_parameters: list[str] = field(default_factory=list)
    created_at: str = ""
    valid_until: str = ""
    provider_approved: bool = False


# ============================================================
# Care Plan Tools
# ============================================================

CARE_PLAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_clinical_guidelines",
            "description": "Retrieve evidence-based clinical guidelines for a given diagnosis using RAG over clinical knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis_code": {"type": "string", "description": "ICD-10 code"},
                    "diagnosis_name": {"type": "string"},
                    "patient_age": {"type": "integer"},
                    "comorbidities": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["diagnosis_code", "diagnosis_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "Check for drug-drug interactions between proposed medications and current medications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposed_medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New medications to add"
                    },
                    "current_medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patient's existing medications"
                    },
                    "patient_conditions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["proposed_medications", "current_medications"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_insurance_coverage",
            "description": "Check if proposed treatments and medications are covered by patient's insurance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "treatments": {"type": "array", "items": {"type": "string"}},
                    "medications": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["patient_id", "treatments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_care_plan",
            "description": "Create a structured care plan with goals, interventions, medications, and follow-up schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "primary_diagnosis": {"type": "string"},
                    "goals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "target_date": {"type": "string"},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "interventions": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "dose": {"type": "string"},
                                "frequency": {"type": "string"},
                                "duration": {"type": "string"},
                                "purpose": {"type": "string"}
                            }
                        }
                    },
                    "lifestyle": {"type": "array", "items": {"type": "string"}},
                    "monitoring": {"type": "array", "items": {"type": "string"}},
                    "follow_ups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "timing": {"type": "string"},
                                "purpose": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["patient_id", "primary_diagnosis", "goals"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_follow_ups",
            "description": "Schedule follow-up appointments and lab work based on the care plan timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "appointments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "days_from_now": {"type": "integer"},
                                "department": {"type": "string"},
                                "reason": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["patient_id", "appointments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_care_plan_to_patient",
            "description": "Send the finalized care plan to the patient with plain-language instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "plan_summary": {"type": "string", "description": "Patient-friendly summary"},
                    "action_items": {"type": "array", "items": {"type": "string"}},
                    "medication_instructions": {"type": "array", "items": {"type": "string"}},
                    "warning_signs": {"type": "array", "items": {"type": "string"}, "description": "When to seek immediate care"}
                },
                "required": ["patient_id", "plan_summary", "action_items"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class CarePlanToolExecutor:
    """Executes care plan tool calls."""
    
    # Simulated drug interaction database
    INTERACTIONS = {
        ("warfarin", "aspirin"): {"severity": "severe", "effect": "Increased bleeding risk"},
        ("lisinopril", "potassium"): {"severity": "moderate", "effect": "Hyperkalemia risk"},
        ("metformin", "contrast_dye"): {"severity": "severe", "effect": "Lactic acidosis risk"},
        ("ssri", "maoi"): {"severity": "contraindicated", "effect": "Serotonin syndrome"},
        ("statin", "fibrate"): {"severity": "moderate", "effect": "Increased myopathy risk"},
    }
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        handlers = {
            "retrieve_clinical_guidelines": self._retrieve_guidelines,
            "check_drug_interactions": self._check_interactions,
            "verify_insurance_coverage": self._verify_coverage,
            "create_care_plan": self._create_care_plan,
            "schedule_follow_ups": self._schedule_follow_ups,
            "send_care_plan_to_patient": self._send_to_patient,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(**arguments)
    
    def _retrieve_guidelines(
        self, diagnosis_code: str, diagnosis_name: str,
        patient_age: int = None, comorbidities: list[str] = None
    ) -> dict:
        """Retrieve clinical guidelines via RAG (simulated)."""
        # In production: vector search over clinical guideline embeddings in AnalyticDB
        guidelines = {
            "E11": {  # Type 2 Diabetes
                "source": "ADA Standards of Care 2024",
                "recommendations": [
                    "HbA1c target < 7% for most adults",
                    "First-line: Metformin + lifestyle modifications",
                    "Add GLP-1 RA or SGLT2i if cardiovascular disease present",
                    "Blood pressure target < 130/80 mmHg",
                    "Statin therapy for ages 40-75",
                    "Annual dilated eye exam",
                    "Annual foot exam",
                    "Self-monitoring blood glucose as needed",
                ],
                "monitoring": ["HbA1c every 3 months", "Lipid panel annually", "Renal function annually", "Urine albumin annually"],
                "lifestyle": ["150 min/week moderate exercise", "Medical nutrition therapy", "Weight loss 5-7% if overweight"],
            },
            "I10": {  # Hypertension
                "source": "ACC/AHA Hypertension Guidelines 2023",
                "recommendations": [
                    "Target BP < 130/80 for most patients",
                    "First-line: ACE inhibitor, ARB, CCB, or thiazide",
                    "Lifestyle: DASH diet, sodium < 2300mg/day, exercise",
                    "Home BP monitoring recommended",
                ],
                "monitoring": ["BP check every 3-6 months", "Renal function annually", "Electrolytes with diuretic use"],
                "lifestyle": ["DASH diet", "Sodium restriction", "30 min exercise most days", "Limit alcohol", "Maintain healthy weight"],
            },
        }
        
        # Match by first 3 chars of ICD-10 code
        code_prefix = diagnosis_code[:3]
        guideline = guidelines.get(code_prefix, {
            "source": "General clinical practice",
            "recommendations": ["Follow evidence-based treatment for " + diagnosis_name],
            "monitoring": ["Regular follow-up as clinically indicated"],
            "lifestyle": ["Healthy diet and regular exercise"],
        })
        
        return {
            "diagnosis": diagnosis_name,
            "guidelines_found": True,
            "source": guideline["source"],
            "recommendations": guideline["recommendations"],
            "monitoring_parameters": guideline["monitoring"],
            "lifestyle_modifications": guideline["lifestyle"],
            "age_specific_notes": f"Patient age {patient_age}: standard adult guidelines apply" if patient_age else None,
            "comorbidity_considerations": comorbidities or [],
        }
    
    def _check_interactions(
        self, proposed_medications: list[str], current_medications: list[str],
        patient_conditions: list[str] = None
    ) -> dict:
        """Check drug-drug interactions."""
        found_interactions = []
        
        for proposed in proposed_medications:
            for current in current_medications:
                pair = (proposed.lower(), current.lower())
                reverse_pair = (current.lower(), proposed.lower())
                
                interaction = self.INTERACTIONS.get(pair) or self.INTERACTIONS.get(reverse_pair)
                if interaction:
                    found_interactions.append({
                        "drug_1": proposed,
                        "drug_2": current,
                        "severity": interaction["severity"],
                        "effect": interaction["effect"],
                    })
        
        return {
            "interactions_found": len(found_interactions),
            "interactions": found_interactions,
            "has_contraindications": any(i["severity"] == "contraindicated" for i in found_interactions),
            "has_severe": any(i["severity"] == "severe" for i in found_interactions),
            "safe_to_proceed": len(found_interactions) == 0 or all(i["severity"] == "mild" for i in found_interactions),
            "recommendation": "Review interactions with pharmacist" if found_interactions else "No significant interactions detected",
        }
    
    def _verify_coverage(self, patient_id: str, treatments: list[str], medications: list[str] = None) -> dict:
        """Verify insurance coverage (simulated)."""
        return {
            "patient_id": patient_id,
            "coverage_verified": True,
            "covered_treatments": treatments,
            "covered_medications": medications or [],
            "requires_prior_auth": [],
            "copay_estimates": {t: "$30-50" for t in treatments},
            "formulary_status": {m: "preferred" for m in (medications or [])},
        }
    
    def _create_care_plan(
        self, patient_id: str, primary_diagnosis: str, goals: list[dict],
        medications: list[dict] = None, lifestyle: list[str] = None,
        monitoring: list[str] = None, follow_ups: list[dict] = None
    ) -> dict:
        """Create structured care plan."""
        plan_id = f"CP-{patient_id[:8]}-{datetime.utcnow().strftime('%Y%m%d')}"
        
        return {
            "plan_id": plan_id,
            "patient_id": patient_id,
            "primary_diagnosis": primary_diagnosis,
            "created_at": datetime.utcnow().isoformat(),
            "valid_until": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "goals_count": len(goals),
            "goals": goals,
            "medications": medications or [],
            "lifestyle_modifications": lifestyle or [],
            "monitoring_parameters": monitoring or [],
            "follow_up_schedule": follow_ups or [],
            "status": "active",
            "requires_provider_approval": True,
        }
    
    def _schedule_follow_ups(self, patient_id: str, appointments: list[dict]) -> dict:
        """Schedule follow-up appointments."""
        scheduled = []
        for appt in appointments:
            target_date = datetime.utcnow() + timedelta(days=appt.get("days_from_now", 30))
            scheduled.append({
                "appointment_id": f"FU-{patient_id[:6]}-{target_date.strftime('%m%d')}",
                "type": appt["type"],
                "scheduled_date": target_date.isoformat(),
                "department": appt.get("department", "primary_care"),
                "reason": appt.get("reason", "Follow-up"),
                "status": "scheduled",
            })
        
        return {"patient_id": patient_id, "appointments_scheduled": len(scheduled), "appointments": scheduled}
    
    def _send_to_patient(
        self, patient_id: str, plan_summary: str, action_items: list[str],
        medication_instructions: list[str] = None, warning_signs: list[str] = None
    ) -> dict:
        """Send care plan to patient."""
        return {
            "notification_id": f"NOTIF-CP-{patient_id[:8]}",
            "patient_id": patient_id,
            "delivered_via": ["patient_portal", "email"],
            "plan_summary_length": len(plan_summary),
            "action_items_count": len(action_items),
            "sent_at": datetime.utcnow().isoformat(),
            "status": "delivered",
        }


# ============================================================
# Care Plan Agent
# ============================================================

class CarePlanAgent:
    """
    Autonomous Care Plan Generation Agent.
    
    Creates evidence-based, personalized care plans by combining
    clinical guidelines with patient-specific factors.
    """
    
    SYSTEM_PROMPT = """You are an autonomous care plan generation agent. Your role is to
create comprehensive, evidence-based care plans for patients.

WORKFLOW:
1. retrieve_clinical_guidelines — Get evidence-based recommendations for the diagnosis
2. check_drug_interactions — Verify safety of proposed medications
3. verify_insurance_coverage — Ensure treatments are covered
4. create_care_plan — Build structured plan with SMART goals
5. schedule_follow_ups — Set up monitoring appointments
6. send_care_plan_to_patient — Deliver patient-friendly version

RULES:
- Goals must be SMART: Specific, Measurable, Achievable, Relevant, Time-bound
- Always check drug interactions BEFORE finalizing medications
- Include patient warning signs ("When to call your doctor / go to ER")
- Adapt complexity to patient's health literacy
- Include lifestyle modifications alongside medications
- Consider comorbidities when selecting treatments
- Plan must be provider-approved before becoming active
"""
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.tool_executor = CarePlanToolExecutor()
    
    def run(self, patient_id: str, diagnoses: list[Diagnosis], patient_info: dict = None) -> CarePlan:
        """Execute the full care plan generation workflow."""
        logger.info(f"Generating care plan for patient {patient_id}")
        
        context = self._build_context(patient_id, diagnoses, patient_info or {})
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        
        response = self.qwen.function_call_loop(
            messages=messages,
            tools=CARE_PLAN_TOOLS,
            tool_executor=self.tool_executor.execute,
            task_type="care_plan",
            max_iterations=10,
        )
        
        return CarePlan(
            plan_id=f"CP-{patient_id[:8]}-{datetime.utcnow().strftime('%Y%m%d')}",
            patient_id=patient_id,
            diagnoses=diagnoses,
            created_at=datetime.utcnow().isoformat(),
        )
    
    def _build_context(self, patient_id: str, diagnoses: list[Diagnosis], patient_info: dict) -> str:
        diag_str = "\n".join(
            f"  - [{d.code}] {d.description} (severity: {d.severity}, onset: {d.onset_date})"
            for d in diagnoses
        )
        
        return f"""CARE PLAN GENERATION REQUEST

Patient ID: {patient_id}
Age: {patient_info.get('age', 'Unknown')}
Gender: {patient_info.get('gender', 'Unknown')}

DIAGNOSES:
{diag_str}

CURRENT MEDICATIONS: {json.dumps(patient_info.get('medications', []))}
ALLERGIES: {json.dumps(patient_info.get('allergies', []))}
COMORBIDITIES: {json.dumps(patient_info.get('comorbidities', []))}

Please generate a comprehensive care plan:
1. Retrieve evidence-based guidelines for each diagnosis
2. Check drug interactions for any proposed medications
3. Verify insurance coverage
4. Create structured care plan with SMART goals
5. Schedule follow-up appointments
6. Send patient-friendly version to patient
"""
