"""
Triage Agent — Autonomous Patient Triage Workflow

This agent handles the complete patient triage workflow:
1. Collects structured symptom information
2. Performs clinical risk assessment using Qwen reasoning
3. Assigns urgency level (Emergency / Urgent / Routine)
4. Routes to appropriate department
5. Schedules appointment automatically
6. Sends patient confirmation

The agent uses Qwen-max for complex clinical reasoning and
makes autonomous decisions for routine cases, escalating to
human providers only when confidence is below threshold.
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

class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"      # Immediate attention (< 10 min)
    URGENT = "urgent"            # Same-day appointment
    SEMI_URGENT = "semi_urgent"  # Within 24-48 hours
    ROUTINE = "routine"          # Standard scheduling (3-7 days)
    PREVENTIVE = "preventive"    # Wellness/screening


class Department(str, Enum):
    EMERGENCY = "emergency_department"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ORTHOPEDICS = "orthopedics"
    INTERNAL_MEDICINE = "internal_medicine"
    PEDIATRICS = "pediatrics"
    PSYCHIATRY = "psychiatry"
    DERMATOLOGY = "dermatology"
    GASTROENTEROLOGY = "gastroenterology"
    PULMONOLOGY = "pulmonology"
    PRIMARY_CARE = "primary_care"


@dataclass
class PatientSymptoms:
    """Structured symptom information collected from patient."""
    chief_complaint: str
    symptoms: list[str]
    duration: str
    severity: int  # 1-10 scale
    onset: str  # sudden / gradual
    associated_symptoms: list[str] = field(default_factory=list)
    medical_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    vital_signs: dict = field(default_factory=dict)
    age: Optional[int] = None
    gender: Optional[str] = None


@dataclass
class TriageResult:
    """Output of the triage assessment."""
    urgency: UrgencyLevel
    department: Department
    confidence: float  # 0.0 - 1.0
    reasoning: str
    red_flags: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    appointment_window: Optional[str] = None
    escalate_to_human: bool = False
    escalation_reason: Optional[str] = None


@dataclass
class TriageWorkflowState:
    """Complete state of a triage workflow execution."""
    workflow_id: str
    patient_id: str
    status: str = "initiated"  # initiated → assessing → routing → scheduling → completed
    symptoms: Optional[PatientSymptoms] = None
    triage_result: Optional[TriageResult] = None
    appointment_id: Optional[str] = None
    notifications_sent: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    audit_log: list[dict] = field(default_factory=list)


# ============================================================
# Triage Agent Tools (callable by Qwen via function calling)
# ============================================================

TRIAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assess_clinical_risk",
            "description": "Assess clinical risk based on symptoms, vital signs, and patient history. Returns urgency level and confidence score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reported symptoms"
                    },
                    "severity": {
                        "type": "integer",
                        "description": "Patient-reported severity 1-10"
                    },
                    "vital_signs": {
                        "type": "object",
                        "description": "Available vital signs (heart_rate, blood_pressure, temperature, spo2, respiratory_rate)",
                        "properties": {
                            "heart_rate": {"type": "integer"},
                            "blood_pressure_systolic": {"type": "integer"},
                            "blood_pressure_diastolic": {"type": "integer"},
                            "temperature": {"type": "number"},
                            "spo2": {"type": "integer"},
                            "respiratory_rate": {"type": "integer"}
                        }
                    },
                    "age": {"type": "integer", "description": "Patient age"},
                    "medical_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant medical history"
                    }
                },
                "required": ["symptoms", "severity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_red_flags",
            "description": "Check for clinical red flags that require immediate escalation. Returns any detected emergency indicators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chief_complaint": {"type": "string"},
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "vital_signs": {"type": "object"}
                },
                "required": ["chief_complaint", "symptoms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_department",
            "description": "Determine the most appropriate department based on symptoms and urgency level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_symptoms": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["emergency", "urgent", "semi_urgent", "routine", "preventive"]
                    },
                    "age": {"type": "integer"}
                },
                "required": ["primary_symptoms", "urgency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Schedule an appointment with the appropriate provider based on urgency and department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "department": {"type": "string"},
                    "urgency": {"type": "string"},
                    "preferred_time": {"type": "string", "description": "ISO format datetime or 'next_available'"},
                    "reason": {"type": "string"}
                },
                "required": ["patient_id", "department", "urgency", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_patient_notification",
            "description": "Send notification to patient with triage results and next steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "channel": {
                        "type": "string",
                        "enum": ["sms", "email", "push", "in_app"]
                    },
                    "message_type": {
                        "type": "string",
                        "enum": ["triage_result", "appointment_confirmation", "preparation_instructions", "emergency_alert"]
                    },
                    "content": {"type": "string"}
                },
                "required": ["patient_id", "channel", "message_type", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_provider",
            "description": "Escalate to a human healthcare provider when agent confidence is low or case is complex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "urgency": {"type": "string"},
                    "summary": {"type": "string", "description": "Brief clinical summary for the provider"}
                },
                "required": ["patient_id", "reason", "urgency", "summary"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class TriageToolExecutor:
    """Executes triage agent tool calls against backend services."""
    
    # Clinical red flags that require immediate attention
    RED_FLAG_PATTERNS = {
        "chest_pain": ["chest pain", "chest tightness", "pressure in chest"],
        "stroke_signs": ["sudden numbness", "facial drooping", "slurred speech", "sudden confusion"],
        "breathing": ["cannot breathe", "severe shortness of breath", "choking"],
        "bleeding": ["uncontrolled bleeding", "coughing blood", "vomiting blood"],
        "consciousness": ["loss of consciousness", "unresponsive", "seizure"],
        "anaphylaxis": ["throat swelling", "cannot swallow", "severe allergic"],
        "cardiac": ["heart palpitations with dizziness", "irregular heartbeat with fainting"],
    }
    
    # Department routing rules
    SYMPTOM_DEPARTMENT_MAP = {
        "chest pain": Department.CARDIOLOGY,
        "heart": Department.CARDIOLOGY,
        "palpitations": Department.CARDIOLOGY,
        "headache": Department.NEUROLOGY,
        "dizziness": Department.NEUROLOGY,
        "numbness": Department.NEUROLOGY,
        "seizure": Department.NEUROLOGY,
        "bone": Department.ORTHOPEDICS,
        "joint": Department.ORTHOPEDICS,
        "fracture": Department.ORTHOPEDICS,
        "stomach": Department.GASTROENTEROLOGY,
        "nausea": Department.GASTROENTEROLOGY,
        "abdominal": Department.GASTROENTEROLOGY,
        "breathing": Department.PULMONOLOGY,
        "cough": Department.PULMONOLOGY,
        "asthma": Department.PULMONOLOGY,
        "skin": Department.DERMATOLOGY,
        "rash": Department.DERMATOLOGY,
        "anxiety": Department.PSYCHIATRY,
        "depression": Department.PSYCHIATRY,
    }
    
    def __init__(self, db_client=None, notification_service=None, scheduler_service=None):
        self.db = db_client
        self.notifications = notification_service
        self.scheduler = scheduler_service
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Route tool calls to appropriate handler."""
        handlers = {
            "assess_clinical_risk": self._assess_clinical_risk,
            "check_red_flags": self._check_red_flags,
            "route_to_department": self._route_to_department,
            "schedule_appointment": self._schedule_appointment,
            "send_patient_notification": self._send_notification,
            "escalate_to_provider": self._escalate_to_provider,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        logger.info(f"Executing tool: {tool_name}")
        return handler(**arguments)
    
    def _assess_clinical_risk(
        self,
        symptoms: list[str],
        severity: int,
        vital_signs: dict = None,
        age: int = None,
        medical_history: list[str] = None,
    ) -> dict:
        """Rule-based + AI-assisted clinical risk scoring."""
        risk_score = 0
        factors = []
        
        # Severity contributes directly
        risk_score += severity * 10  # 10-100
        
        # Vital signs assessment
        if vital_signs:
            hr = vital_signs.get("heart_rate", 80)
            if hr > 120 or hr < 50:
                risk_score += 30
                factors.append(f"Abnormal heart rate: {hr}")
            
            temp = vital_signs.get("temperature", 98.6)
            if temp > 103 or temp < 95:
                risk_score += 25
                factors.append(f"Abnormal temperature: {temp}°F")
            
            spo2 = vital_signs.get("spo2", 98)
            if spo2 < 92:
                risk_score += 40
                factors.append(f"Low oxygen saturation: {spo2}%")
            elif spo2 < 95:
                risk_score += 20
                factors.append(f"Borderline oxygen: {spo2}%")
            
            systolic = vital_signs.get("blood_pressure_systolic", 120)
            if systolic > 180 or systolic < 90:
                risk_score += 30
                factors.append(f"Critical blood pressure: {systolic}")
        
        # Age factors
        if age:
            if age > 65:
                risk_score += 15
                factors.append("Age > 65 — elevated risk")
            elif age < 5:
                risk_score += 15
                factors.append("Pediatric patient — elevated caution")
        
        # Medical history amplifiers
        high_risk_conditions = ["diabetes", "heart disease", "cancer", "immunocompromised", "copd"]
        if medical_history:
            for condition in medical_history:
                if any(hrc in condition.lower() for hrc in high_risk_conditions):
                    risk_score += 15
                    factors.append(f"High-risk history: {condition}")
        
        # Determine urgency from score
        if risk_score >= 150:
            urgency = UrgencyLevel.EMERGENCY
        elif risk_score >= 100:
            urgency = UrgencyLevel.URGENT
        elif risk_score >= 60:
            urgency = UrgencyLevel.SEMI_URGENT
        else:
            urgency = UrgencyLevel.ROUTINE
        
        confidence = min(0.95, 0.6 + (len(factors) * 0.05))
        
        return {
            "risk_score": min(risk_score, 200),
            "urgency": urgency.value,
            "confidence": confidence,
            "contributing_factors": factors,
            "recommendation": f"Assessed as {urgency.value} with score {risk_score}/200",
        }
    
    def _check_red_flags(
        self,
        chief_complaint: str,
        symptoms: list[str],
        vital_signs: dict = None,
    ) -> dict:
        """Check for clinical red flags requiring immediate intervention."""
        detected_flags = []
        all_text = (chief_complaint + " " + " ".join(symptoms)).lower()
        
        for category, patterns in self.RED_FLAG_PATTERNS.items():
            for pattern in patterns:
                if pattern in all_text:
                    detected_flags.append({
                        "category": category,
                        "pattern_matched": pattern,
                        "action": "immediate_escalation" if category in ["stroke_signs", "breathing", "consciousness"] else "urgent_review"
                    })
        
        # Vital sign red flags
        if vital_signs:
            if vital_signs.get("spo2", 100) < 90:
                detected_flags.append({"category": "hypoxia", "pattern_matched": f"SpO2 {vital_signs['spo2']}%", "action": "immediate_escalation"})
            if vital_signs.get("heart_rate", 80) > 150:
                detected_flags.append({"category": "tachycardia", "pattern_matched": f"HR {vital_signs['heart_rate']}", "action": "urgent_review"})
        
        return {
            "red_flags_detected": len(detected_flags) > 0,
            "flags": detected_flags,
            "immediate_action_required": any(f["action"] == "immediate_escalation" for f in detected_flags),
            "recommendation": "ESCALATE IMMEDIATELY" if detected_flags else "No red flags detected — proceed with standard triage"
        }
    
    def _route_to_department(
        self,
        primary_symptoms: list[str],
        urgency: str,
        age: int = None,
    ) -> dict:
        """Route patient to most appropriate department."""
        
        # Emergency override
        if urgency == "emergency":
            return {
                "department": Department.EMERGENCY.value,
                "reason": "Emergency urgency — routing to ED",
                "alternative": None,
            }
        
        # Pediatric override
        if age and age < 18:
            return {
                "department": Department.PEDIATRICS.value,
                "reason": f"Pediatric patient (age {age})",
                "alternative": None,
            }
        
        # Symptom-based routing
        for symptom in primary_symptoms:
            symptom_lower = symptom.lower()
            for keyword, dept in self.SYMPTOM_DEPARTMENT_MAP.items():
                if keyword in symptom_lower:
                    return {
                        "department": dept.value,
                        "reason": f"Symptom '{symptom}' maps to {dept.value}",
                        "alternative": Department.PRIMARY_CARE.value,
                    }
        
        # Default to primary care
        return {
            "department": Department.PRIMARY_CARE.value,
            "reason": "No specific department match — routing to primary care",
            "alternative": None,
        }
    
    def _schedule_appointment(
        self,
        patient_id: str,
        department: str,
        urgency: str,
        reason: str,
        preferred_time: str = "next_available",
    ) -> dict:
        """Schedule appointment based on urgency-driven timing."""
        
        now = datetime.utcnow()
        
        # Determine appointment window based on urgency
        time_windows = {
            "emergency": timedelta(minutes=0),   # Immediate
            "urgent": timedelta(hours=4),         # Same day
            "semi_urgent": timedelta(hours=24),   # Next day
            "routine": timedelta(days=5),         # Within a week
            "preventive": timedelta(days=14),     # Within 2 weeks
        }
        
        target_time = now + time_windows.get(urgency, timedelta(days=7))
        
        # Simulate appointment creation
        appointment = {
            "appointment_id": f"APT-{patient_id[:8]}-{now.strftime('%Y%m%d%H%M')}",
            "patient_id": patient_id,
            "department": department,
            "scheduled_time": target_time.isoformat(),
            "urgency": urgency,
            "reason": reason,
            "status": "confirmed",
            "provider": f"Dr. Auto-Assigned ({department})",
            "instructions": self._get_preparation_instructions(department, urgency),
        }
        
        logger.info(f"Appointment scheduled: {appointment['appointment_id']}")
        return appointment
    
    def _send_notification(
        self,
        patient_id: str,
        channel: str,
        message_type: str,
        content: str,
    ) -> dict:
        """Send notification to patient via specified channel."""
        notification = {
            "notification_id": f"NOTIF-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "patient_id": patient_id,
            "channel": channel,
            "message_type": message_type,
            "content": content,
            "sent_at": datetime.utcnow().isoformat(),
            "status": "delivered",
        }
        
        logger.info(f"Notification sent: {notification['notification_id']} via {channel}")
        return notification
    
    def _escalate_to_provider(
        self,
        patient_id: str,
        reason: str,
        urgency: str,
        summary: str,
    ) -> dict:
        """Escalate case to human provider."""
        escalation = {
            "escalation_id": f"ESC-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "patient_id": patient_id,
            "reason": reason,
            "urgency": urgency,
            "clinical_summary": summary,
            "assigned_to": "on_call_physician",
            "status": "pending_review",
            "escalated_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"ESCALATED: {escalation['escalation_id']} — {reason}")
        return escalation
    
    def _get_preparation_instructions(self, department: str, urgency: str) -> str:
        """Generate pre-appointment preparation instructions."""
        base = "Please bring your insurance card and photo ID."
        
        instructions = {
            "cardiology": f"{base} Fast for 8 hours if bloodwork may be needed. Wear comfortable clothing.",
            "gastroenterology": f"{base} Do not eat or drink 12 hours before. Bring list of current medications.",
            "neurology": f"{base} Note any triggers or patterns for your symptoms.",
            "orthopedics": f"{base} Wear loose-fitting clothing. Bring any imaging you have.",
            "primary_care": f"{base} List all medications and recent symptom changes.",
        }
        
        return instructions.get(department, base)


# ============================================================
# Triage Agent — Main Orchestrator
# ============================================================

class TriageAgent:
    """
    Autonomous Triage Agent that handles the complete patient intake workflow.
    
    This agent:
    1. Receives patient symptom data
    2. Uses Qwen-max for clinical reasoning (function calling)
    3. Assesses risk, checks red flags, routes to department
    4. Schedules appointment and notifies patient
    5. Escalates to human providers when confidence is low
    
    The entire workflow runs autonomously — no human intervention
    required for routine cases.
    """
    
    SYSTEM_PROMPT = """You are an autonomous healthcare triage agent. Your role is to 
assess patients presenting with symptoms and determine the appropriate urgency level, 
department routing, and next steps.

CLINICAL GUIDELINES:
- Always check for red flags FIRST before proceeding with standard assessment
- Use the assess_clinical_risk tool to generate a quantitative risk score
- Route to appropriate department based on primary symptoms
- Schedule appointments according to urgency-based timing
- Escalate to a human provider if:
  * Your confidence is below 0.7
  * Red flags are detected that require clinical judgment
  * The patient presents with multiple complex comorbidities
  * The case doesn't fit standard triage protocols

WORKFLOW ORDER:
1. check_red_flags — identify any immediate dangers
2. assess_clinical_risk — quantify overall risk
3. route_to_department — determine specialty
4. schedule_appointment — book the visit
5. send_patient_notification — inform the patient

SAFETY RULES:
- Never diagnose — only triage and route
- Always err on the side of caution (upgrade urgency when uncertain)
- Document reasoning for every decision
- Flag potential mental health concerns for dual routing

Respond with a complete triage summary after executing all necessary steps."""
    
    CONFIDENCE_THRESHOLD = 0.7  # Below this, escalate to human
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.tool_executor = TriageToolExecutor()
    
    def run(self, patient_id: str, symptoms: PatientSymptoms) -> TriageWorkflowState:
        """
        Execute the complete triage workflow autonomously.
        
        Args:
            patient_id: Unique patient identifier
            symptoms: Structured symptom information
            
        Returns:
            TriageWorkflowState with complete workflow results
        """
        state = TriageWorkflowState(
            workflow_id=f"TRIAGE-{patient_id[:8]}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            patient_id=patient_id,
            symptoms=symptoms,
            started_at=datetime.utcnow(),
        )
        
        logger.info(f"Starting triage workflow: {state.workflow_id}")
        state.status = "assessing"
        
        # Build the clinical context message for Qwen
        clinical_context = self._build_clinical_context(patient_id, symptoms)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": clinical_context},
        ]
        
        # Run the autonomous agent loop
        try:
            response = self.qwen.function_call_loop(
                messages=messages,
                tools=TRIAGE_TOOLS,
                tool_executor=self.tool_executor.execute,
                task_type="triage",
                max_iterations=8,
            )
            
            state.status = "completed"
            state.completed_at = datetime.utcnow()
            state.audit_log.append({
                "action": "workflow_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "agent_response": response.content,
                "model_used": response.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            })
            
            logger.info(f"Triage completed: {state.workflow_id}")
            
        except Exception as e:
            state.status = "error"
            state.audit_log.append({
                "action": "workflow_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            })
            logger.error(f"Triage error in {state.workflow_id}: {e}")
            
            # On error, escalate to human
            self.tool_executor.execute("escalate_to_provider", {
                "patient_id": patient_id,
                "reason": f"Agent error: {str(e)}",
                "urgency": "urgent",
                "summary": f"Triage agent encountered an error processing patient symptoms: {symptoms.chief_complaint}",
            })
        
        return state
    
    def _build_clinical_context(self, patient_id: str, symptoms: PatientSymptoms) -> str:
        """Build a structured clinical context message for Qwen."""
        context = f"""PATIENT TRIAGE REQUEST
Patient ID: {patient_id}
{"Age: " + str(symptoms.age) if symptoms.age else ""}
{"Gender: " + symptoms.gender if symptoms.gender else ""}

CHIEF COMPLAINT: {symptoms.chief_complaint}

SYMPTOMS:
{chr(10).join(f"- {s}" for s in symptoms.symptoms)}

DURATION: {symptoms.duration}
SEVERITY: {symptoms.severity}/10
ONSET: {symptoms.onset}
"""
        
        if symptoms.associated_symptoms:
            context += f"\nASSOCIATED SYMPTOMS:\n{chr(10).join(f'- {s}' for s in symptoms.associated_symptoms)}\n"
        
        if symptoms.vital_signs:
            context += f"\nVITAL SIGNS: {json.dumps(symptoms.vital_signs)}\n"
        
        if symptoms.medical_history:
            context += f"\nMEDICAL HISTORY:\n{chr(10).join(f'- {h}' for h in symptoms.medical_history)}\n"
        
        if symptoms.current_medications:
            context += f"\nCURRENT MEDICATIONS:\n{chr(10).join(f'- {m}' for m in symptoms.current_medications)}\n"
        
        if symptoms.allergies:
            context += f"\nALLERGIES:\n{chr(10).join(f'- {a}' for a in symptoms.allergies)}\n"
        
        context += """
Please perform complete triage:
1. Check for red flags
2. Assess clinical risk  
3. Route to appropriate department
4. Schedule appointment
5. Notify patient with results and next steps

If confidence is below 70% at any point, escalate to a provider."""
        
        return context
