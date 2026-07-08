"""
Discharge & Follow-up Coordination Agent — Autonomous Post-Discharge Management

This agent handles the complete discharge workflow:
1. Generates patient-literacy-adapted discharge instructions
2. Schedules follow-up appointments
3. Sets up medication reminders
4. Initiates post-discharge check-in protocol
5. Monitors patient-reported outcomes
6. Alerts provider if recovery deviates from expected trajectory

Uses Qwen-plus for patient communication (clear, empathetic language)
and Qwen-turbo for scheduling coordination.
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

class RecoveryStatus(str, Enum):
    ON_TRACK = "on_track"
    MINOR_CONCERN = "minor_concern"
    NEEDS_ATTENTION = "needs_attention"
    EMERGENCY = "emergency"


class LiteracyLevel(str, Enum):
    BASIC = "basic"          # 5th-6th grade reading level
    INTERMEDIATE = "intermediate"  # 8th-9th grade
    ADVANCED = "advanced"    # College level


@dataclass
class DischargeOrder:
    """Physician discharge order."""
    patient_id: str
    encounter_id: str
    diagnosis: str
    procedure_performed: str = ""
    discharge_date: str = ""
    discharge_medications: list[dict] = field(default_factory=list)
    activity_restrictions: list[str] = field(default_factory=list)
    diet_instructions: str = ""
    follow_up_instructions: str = ""
    warning_signs: list[str] = field(default_factory=list)
    provider_id: str = ""
    literacy_level: LiteracyLevel = LiteracyLevel.INTERMEDIATE


@dataclass
class DischargeResult:
    """Result of discharge coordination workflow."""
    patient_id: str
    encounter_id: str
    instructions_delivered: bool = False
    follow_ups_scheduled: list[dict] = field(default_factory=list)
    reminders_configured: list[dict] = field(default_factory=list)
    check_in_protocol: dict = field(default_factory=dict)
    status: str = "completed"
    audit_log: list[dict] = field(default_factory=list)


# ============================================================
# Discharge Tools
# ============================================================

DISCHARGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_discharge_instructions",
            "description": "Generate patient-friendly discharge instructions adapted to health literacy level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "procedure": {"type": "string"},
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "dose": {"type": "string"},
                                "frequency": {"type": "string"},
                                "special_instructions": {"type": "string"}
                            }
                        }
                    },
                    "activity_restrictions": {"type": "array", "items": {"type": "string"}},
                    "diet": {"type": "string"},
                    "warning_signs": {"type": "array", "items": {"type": "string"}},
                    "literacy_level": {"type": "string", "enum": ["basic", "intermediate", "advanced"]}
                },
                "required": ["diagnosis", "medications", "warning_signs", "literacy_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_follow_up_visits",
            "description": "Schedule post-discharge follow-up appointments based on condition and procedure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "appointments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "wound_check, post_op, lab_work, specialist, primary_care"},
                                "days_after_discharge": {"type": "integer"},
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
            "name": "configure_medication_reminders",
            "description": "Set up automated medication reminders for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "times": {"type": "array", "items": {"type": "string"}, "description": "Times of day: 08:00, 12:00, etc."},
                                "special_notes": {"type": "string"},
                                "duration_days": {"type": "integer"}
                            }
                        }
                    },
                    "reminder_channel": {"type": "string", "enum": ["sms", "push", "email", "phone_call"]}
                },
                "required": ["patient_id", "medications"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "setup_check_in_protocol",
            "description": "Configure automated post-discharge check-ins to monitor recovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "check_in_schedule": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "integer", "description": "Days after discharge"},
                                "questions": {"type": "array", "items": {"type": "string"}},
                                "channel": {"type": "string"}
                            }
                        }
                    },
                    "escalation_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Responses that trigger provider alert"
                    }
                },
                "required": ["patient_id", "check_in_schedule", "escalation_criteria"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_patient_response",
            "description": "Evaluate a patient's check-in response and determine if intervention is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "day_post_discharge": {"type": "integer"},
                    "responses": {
                        "type": "object",
                        "description": "Patient's answers to check-in questions"
                    },
                    "pain_level": {"type": "integer", "description": "0-10 scale"},
                    "reported_symptoms": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["patient_id", "day_post_discharge", "responses"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alert_care_team",
            "description": "Alert the care team when patient's recovery is off-track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "provider_id": {"type": "string"},
                    "alert_level": {"type": "string", "enum": ["informational", "needs_attention", "urgent", "emergency"]},
                    "concern": {"type": "string"},
                    "patient_reported_data": {"type": "object"},
                    "recommended_action": {"type": "string"}
                },
                "required": ["patient_id", "provider_id", "alert_level", "concern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deliver_instructions_to_patient",
            "description": "Deliver discharge instructions and care package to the patient via their preferred channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "channels": {"type": "array", "items": {"type": "string"}, "description": "Delivery channels: portal, email, sms, print"},
                    "instructions_content": {"type": "string"},
                    "include_medication_list": {"type": "boolean"},
                    "include_appointment_summary": {"type": "boolean"},
                    "emergency_contact_info": {"type": "string"}
                },
                "required": ["patient_id", "channels", "instructions_content"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class DischargeToolExecutor:
    """Executes discharge workflow tool calls."""
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        handlers = {
            "generate_discharge_instructions": self._generate_instructions,
            "schedule_follow_up_visits": self._schedule_follow_ups,
            "configure_medication_reminders": self._configure_reminders,
            "setup_check_in_protocol": self._setup_check_ins,
            "assess_patient_response": self._assess_response,
            "alert_care_team": self._alert_care_team,
            "deliver_instructions_to_patient": self._deliver_instructions,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(**arguments)
    
    def _generate_instructions(
        self, diagnosis: str, medications: list[dict], warning_signs: list[str],
        literacy_level: str, procedure: str = "", activity_restrictions: list[str] = None,
        diet: str = ""
    ) -> dict:
        """Generate discharge instructions (Qwen generates the actual content)."""
        return {
            "instructions_generated": True,
            "literacy_level": literacy_level,
            "sections": [
                "What happened during your visit",
                "Your medications",
                "Activity guidelines",
                "Diet recommendations",
                "When to call your doctor",
                "When to go to the emergency room",
                "Your follow-up appointments",
            ],
            "medication_count": len(medications),
            "warning_signs_count": len(warning_signs),
            "estimated_reading_time": "5 minutes" if literacy_level == "basic" else "3 minutes",
        }
    
    def _schedule_follow_ups(self, patient_id: str, appointments: list[dict]) -> dict:
        """Schedule post-discharge follow-ups."""
        scheduled = []
        for appt in appointments:
            target_date = datetime.utcnow() + timedelta(days=appt.get("days_after_discharge", 7))
            scheduled.append({
                "appointment_id": f"FU-{patient_id[:6]}-{target_date.strftime('%m%d')}",
                "type": appt["type"],
                "date": target_date.isoformat(),
                "department": appt.get("department", "primary_care"),
                "reason": appt.get("reason", "Post-discharge follow-up"),
                "status": "confirmed",
                "reminder_set": True,
            })
        
        return {
            "patient_id": patient_id,
            "total_scheduled": len(scheduled),
            "appointments": scheduled,
            "next_appointment": scheduled[0] if scheduled else None,
        }
    
    def _configure_reminders(
        self, patient_id: str, medications: list[dict], reminder_channel: str = "push"
    ) -> dict:
        """Configure medication reminders."""
        reminders = []
        for med in medications:
            reminders.append({
                "medication": med["name"],
                "times": med.get("times", ["08:00"]),
                "channel": reminder_channel,
                "duration_days": med.get("duration_days", 30),
                "special_notes": med.get("special_notes", ""),
                "active": True,
            })
        
        return {
            "patient_id": patient_id,
            "reminders_configured": len(reminders),
            "channel": reminder_channel,
            "reminders": reminders,
            "start_date": datetime.utcnow().isoformat(),
        }
    
    def _setup_check_ins(
        self, patient_id: str, check_in_schedule: list[dict], escalation_criteria: list[str]
    ) -> dict:
        """Set up automated check-in protocol."""
        return {
            "patient_id": patient_id,
            "protocol_id": f"CHECKIN-{patient_id[:8]}",
            "total_check_ins": len(check_in_schedule),
            "schedule": check_in_schedule,
            "escalation_criteria": escalation_criteria,
            "first_check_in": (datetime.utcnow() + timedelta(days=check_in_schedule[0]["day"])).isoformat() if check_in_schedule else None,
            "status": "active",
        }
    
    def _assess_response(
        self, patient_id: str, day_post_discharge: int, responses: dict,
        pain_level: int = 0, reported_symptoms: list[str] = None
    ) -> dict:
        """Assess patient check-in response."""
        # Simple rule-based assessment
        status = RecoveryStatus.ON_TRACK
        concerns = []
        
        if pain_level >= 8:
            status = RecoveryStatus.NEEDS_ATTENTION
            concerns.append(f"High pain level: {pain_level}/10")
        elif pain_level >= 6:
            status = RecoveryStatus.MINOR_CONCERN
            concerns.append(f"Elevated pain: {pain_level}/10")
        
        warning_symptoms = ["fever", "bleeding", "swelling", "difficulty breathing", "chest pain"]
        if reported_symptoms:
            for symptom in reported_symptoms:
                if any(ws in symptom.lower() for ws in warning_symptoms):
                    status = RecoveryStatus.NEEDS_ATTENTION
                    concerns.append(f"Warning symptom: {symptom}")
        
        return {
            "patient_id": patient_id,
            "day_post_discharge": day_post_discharge,
            "recovery_status": status.value,
            "concerns": concerns,
            "action_needed": status in [RecoveryStatus.NEEDS_ATTENTION, RecoveryStatus.EMERGENCY],
            "recommendation": "Alert provider" if concerns else "Recovery on track — continue monitoring",
        }
    
    def _alert_care_team(
        self, patient_id: str, provider_id: str, alert_level: str, concern: str,
        patient_reported_data: dict = None, recommended_action: str = ""
    ) -> dict:
        """Alert care team about recovery concerns."""
        return {
            "alert_id": f"ALERT-DC-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "patient_id": patient_id,
            "provider_id": provider_id,
            "level": alert_level,
            "concern": concern,
            "recommended_action": recommended_action,
            "sent_at": datetime.utcnow().isoformat(),
            "channel": "pager" if alert_level in ["urgent", "emergency"] else "inbox",
        }
    
    def _deliver_instructions(
        self, patient_id: str, channels: list[str], instructions_content: str,
        include_medication_list: bool = True, include_appointment_summary: bool = True,
        emergency_contact_info: str = ""
    ) -> dict:
        """Deliver instructions to patient."""
        return {
            "delivery_id": f"DEL-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "patient_id": patient_id,
            "channels_used": channels,
            "content_length": len(instructions_content),
            "includes_med_list": include_medication_list,
            "includes_appointments": include_appointment_summary,
            "delivered_at": datetime.utcnow().isoformat(),
            "status": "delivered",
        }


# ============================================================
# Discharge Agent
# ============================================================

class DischargeAgent:
    """
    Autonomous Discharge & Follow-up Coordination Agent.
    
    Manages the complete post-discharge experience from
    instruction generation through recovery monitoring.
    """
    
    SYSTEM_PROMPT = """You are an autonomous discharge coordination agent. Your role is to
ensure patients have a smooth transition from hospital to home with clear instructions,
proper follow-up, and active monitoring.

WORKFLOW:
1. generate_discharge_instructions — Create literacy-adapted instructions
2. deliver_instructions_to_patient — Send via preferred channels
3. schedule_follow_up_visits — Book post-discharge appointments
4. configure_medication_reminders — Set up med reminders
5. setup_check_in_protocol — Configure recovery monitoring check-ins

POST-DISCHARGE MONITORING:
- assess_patient_response — Evaluate check-in answers
- alert_care_team — Escalate if recovery is off-track

RULES:
- Instructions MUST be adapted to patient's health literacy level
- Always include clear "When to call doctor" and "When to go to ER" sections
- Use simple language for basic literacy: short sentences, avoid medical jargon
- Medication instructions must include purpose, not just dose
- First follow-up within 7 days for all surgical patients
- First follow-up within 48 hours for high-risk patients
- Escalate immediately if patient reports warning signs
- Be empathetic and supportive in all patient communications
"""
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.tool_executor = DischargeToolExecutor()
    
    def run(self, order: DischargeOrder) -> DischargeResult:
        """Execute the full discharge coordination workflow."""
        logger.info(f"Processing discharge for patient {order.patient_id}, encounter {order.encounter_id}")
        
        context = self._build_context(order)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        
        response = self.qwen.function_call_loop(
            messages=messages,
            tools=DISCHARGE_TOOLS,
            tool_executor=self.tool_executor.execute,
            task_type="patient_communication",
            max_iterations=8,
        )
        
        return DischargeResult(
            patient_id=order.patient_id,
            encounter_id=order.encounter_id,
            instructions_delivered=True,
            status="completed",
        )
    
    def _build_context(self, order: DischargeOrder) -> str:
        meds_str = "\n".join(
            f"  - {m.get('name', 'Unknown')}: {m.get('dose', '')} {m.get('frequency', '')}"
            for m in order.discharge_medications
        )
        
        return f"""DISCHARGE COORDINATION REQUEST

Patient ID: {order.patient_id}
Encounter ID: {order.encounter_id}
Discharge Date: {order.discharge_date or datetime.utcnow().strftime('%Y-%m-%d')}
Health Literacy Level: {order.literacy_level.value}

DIAGNOSIS: {order.diagnosis}
PROCEDURE: {order.procedure_performed}

MEDICATIONS:
{meds_str or "  No new medications"}

ACTIVITY RESTRICTIONS:
{chr(10).join(f'  - {r}' for r in order.activity_restrictions) or "  None specified"}

DIET: {order.diet_instructions or "Regular diet"}

WARNING SIGNS TO INCLUDE:
{chr(10).join(f'  - {w}' for w in order.warning_signs) or "  Standard warning signs"}

PROVIDER: {order.provider_id}

Please execute the complete discharge workflow:
1. Generate literacy-adapted discharge instructions
2. Deliver to patient via portal and email
3. Schedule follow-up appointments
4. Configure medication reminders
5. Set up post-discharge check-in protocol (Day 1, 3, 7, 14)
"""
