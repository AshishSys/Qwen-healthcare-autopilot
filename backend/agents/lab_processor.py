"""
Lab Result Processing Agent — Autonomous Lab Interpretation Pipeline

This agent handles the complete lab result workflow:
1. Ingests lab results (HL7/FHIR format)
2. Parses and normalizes values against reference ranges
3. Compares against patient's historical baseline
4. Flags critical/abnormal values for immediate provider alert
5. Generates patient-friendly explanations using Qwen
6. Updates care plan if results indicate changes needed
7. Schedules follow-up if warranted

Uses Qwen-max for clinical interpretation and Qwen-plus for patient communication.
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

class ResultFlag(str, Enum):
    CRITICAL_HIGH = "critical_high"
    CRITICAL_LOW = "critical_low"
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"
    PENDING = "pending"


class AlertPriority(str, Enum):
    STAT = "stat"           # Immediate provider notification
    URGENT = "urgent"       # Within 1 hour
    ROUTINE = "routine"     # Standard notification cycle
    INFORMATIONAL = "info"  # No alert needed


@dataclass
class LabTest:
    """Individual lab test result."""
    test_code: str              # LOINC code
    test_name: str              # Human-readable name
    value: float | str          # Result value
    unit: str                   # Unit of measurement
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    flag: ResultFlag = ResultFlag.NORMAL
    specimen_type: str = "blood"
    collected_at: Optional[str] = None


@dataclass
class LabOrder:
    """Complete lab order with multiple tests."""
    order_id: str
    patient_id: str
    ordering_provider: str
    tests: list[LabTest] = field(default_factory=list)
    order_date: str = ""
    result_date: str = ""
    panel_name: str = ""        # e.g., "Comprehensive Metabolic Panel"
    status: str = "final"


@dataclass
class LabInterpretation:
    """Agent's interpretation of lab results."""
    order_id: str
    patient_id: str
    overall_assessment: str
    abnormal_findings: list[dict] = field(default_factory=list)
    critical_values: list[dict] = field(default_factory=list)
    patient_summary: str = ""   # Patient-friendly explanation
    provider_summary: str = ""  # Clinical summary for provider
    recommended_actions: list[str] = field(default_factory=list)
    follow_up_needed: bool = False
    follow_up_reason: str = ""
    alert_priority: AlertPriority = AlertPriority.ROUTINE


# ============================================================
# Lab Processing Tools (callable by Qwen)
# ============================================================

LAB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_lab_results",
            "description": "Parse and normalize raw lab results. Flags values outside reference ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                                "reference_low": {"type": "number"},
                                "reference_high": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["order_id", "tests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_with_baseline",
            "description": "Compare current results against patient's historical lab values to identify significant changes or trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "test_name": {"type": "string"},
                    "current_value": {"type": "number"},
                    "lookback_months": {"type": "integer", "description": "How many months of history to compare"}
                },
                "required": ["patient_id", "test_name", "current_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_critical_value",
            "description": "Determine if a lab value requires immediate critical value notification per laboratory protocols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "patient_age": {"type": "integer"},
                    "patient_conditions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["test_name", "value", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_patient_explanation",
            "description": "Generate a patient-friendly explanation of lab results at appropriate health literacy level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "flag": {"type": "string", "enum": ["normal", "high", "low", "critical_high", "critical_low"]},
                    "clinical_significance": {"type": "string"},
                    "health_literacy_level": {"type": "string", "enum": ["basic", "intermediate", "advanced"]}
                },
                "required": ["test_name", "value", "flag", "clinical_significance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alert_provider",
            "description": "Send alert to ordering provider about abnormal or critical lab results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string"},
                    "patient_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["stat", "urgent", "routine"]},
                    "alert_message": {"type": "string"},
                    "critical_values": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["provider_id", "patient_id", "priority", "alert_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_care_plan",
            "description": "Recommend updates to the patient's care plan based on lab findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "findings": {"type": "string"},
                    "recommended_changes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "follow_up_labs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labs to reorder for follow-up"
                    },
                    "follow_up_timeframe": {"type": "string"}
                },
                "required": ["patient_id", "findings", "recommended_changes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_patient",
            "description": "Send lab results notification to patient with plain-language explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "channel": {"type": "string", "enum": ["sms", "email", "portal", "push"]},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "include_values": {"type": "boolean"},
                    "action_required": {"type": "boolean"}
                },
                "required": ["patient_id", "channel", "subject", "body"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class LabToolExecutor:
    """Executes lab processing tool calls."""
    
    # Critical value thresholds (simplified — real systems use lab-specific ranges)
    CRITICAL_VALUES = {
        "glucose": {"critical_low": 40, "critical_high": 500, "unit": "mg/dL"},
        "potassium": {"critical_low": 2.5, "critical_high": 6.5, "unit": "mEq/L"},
        "sodium": {"critical_low": 120, "critical_high": 160, "unit": "mEq/L"},
        "hemoglobin": {"critical_low": 6.0, "critical_high": 20.0, "unit": "g/dL"},
        "platelets": {"critical_low": 20, "critical_high": 1000, "unit": "K/uL"},
        "wbc": {"critical_low": 1.0, "critical_high": 30.0, "unit": "K/uL"},
        "creatinine": {"critical_low": None, "critical_high": 10.0, "unit": "mg/dL"},
        "troponin": {"critical_low": None, "critical_high": 0.04, "unit": "ng/mL"},
        "inr": {"critical_low": None, "critical_high": 5.0, "unit": "ratio"},
    }
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Route tool calls to handlers."""
        handlers = {
            "parse_lab_results": self._parse_lab_results,
            "compare_with_baseline": self._compare_with_baseline,
            "check_critical_value": self._check_critical_value,
            "generate_patient_explanation": self._generate_patient_explanation,
            "alert_provider": self._alert_provider,
            "update_care_plan": self._update_care_plan,
            "notify_patient": self._notify_patient,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(**arguments)
    
    def _parse_lab_results(self, order_id: str, tests: list[dict]) -> dict:
        """Parse results and flag abnormals."""
        parsed = []
        abnormals = []
        
        for test in tests:
            value = test.get("value", 0)
            ref_low = test.get("reference_low")
            ref_high = test.get("reference_high")
            
            flag = "normal"
            if ref_low is not None and value < ref_low:
                flag = "low"
            elif ref_high is not None and value > ref_high:
                flag = "high"
            
            result = {
                "test_name": test["test_name"],
                "value": value,
                "unit": test.get("unit", ""),
                "flag": flag,
                "reference_range": f"{ref_low}-{ref_high}" if ref_low and ref_high else "N/A"
            }
            parsed.append(result)
            
            if flag != "normal":
                abnormals.append(result)
        
        return {
            "order_id": order_id,
            "total_tests": len(parsed),
            "abnormal_count": len(abnormals),
            "results": parsed,
            "abnormals": abnormals,
        }
    
    def _compare_with_baseline(
        self, patient_id: str, test_name: str, current_value: float, lookback_months: int = 6
    ) -> dict:
        """Compare against historical values (simulated)."""
        # In production, this queries RDS for historical lab data
        # Simulated baseline for demo
        baseline_map = {
            "glucose": 95.0,
            "hemoglobin": 14.2,
            "creatinine": 0.9,
            "potassium": 4.2,
            "sodium": 140.0,
        }
        
        baseline = baseline_map.get(test_name.lower(), current_value * 0.9)
        change_pct = ((current_value - baseline) / baseline) * 100
        
        return {
            "patient_id": patient_id,
            "test_name": test_name,
            "current_value": current_value,
            "baseline_value": baseline,
            "change_percent": round(change_pct, 1),
            "trend": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable",
            "significant_change": abs(change_pct) > 20,
            "lookback_months": lookback_months,
        }
    
    def _check_critical_value(
        self, test_name: str, value: float, unit: str,
        patient_age: int = None, patient_conditions: list[str] = None
    ) -> dict:
        """Check if value meets critical threshold."""
        test_key = test_name.lower().replace(" ", "_")
        thresholds = self.CRITICAL_VALUES.get(test_key)
        
        if not thresholds:
            return {"is_critical": False, "reason": f"No critical thresholds defined for {test_name}"}
        
        is_critical = False
        direction = None
        
        if thresholds["critical_low"] and value <= thresholds["critical_low"]:
            is_critical = True
            direction = "critically_low"
        elif thresholds["critical_high"] and value >= thresholds["critical_high"]:
            is_critical = True
            direction = "critically_high"
        
        return {
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "is_critical": is_critical,
            "direction": direction,
            "threshold": thresholds.get(f"critical_{direction.split('_')[1]}" if direction else "critical_high"),
            "action": "IMMEDIATE_PROVIDER_NOTIFICATION" if is_critical else "standard_processing",
        }
    
    def _generate_patient_explanation(
        self, test_name: str, value: float, flag: str,
        clinical_significance: str, unit: str = "", health_literacy_level: str = "intermediate"
    ) -> dict:
        """Generate patient-friendly explanation."""
        # In production, Qwen generates this — here we provide a template
        explanations = {
            "normal": f"Your {test_name} result ({value} {unit}) is within the normal range. No concerns here.",
            "high": f"Your {test_name} ({value} {unit}) is slightly above the normal range. {clinical_significance}",
            "low": f"Your {test_name} ({value} {unit}) is below the normal range. {clinical_significance}",
            "critical_high": f"Your {test_name} ({value} {unit}) needs immediate attention. Please contact your healthcare provider right away.",
            "critical_low": f"Your {test_name} ({value} {unit}) is very low and needs immediate medical attention.",
        }
        
        return {
            "explanation": explanations.get(flag, f"Your {test_name} result is {value} {unit}."),
            "flag": flag,
            "action_needed": flag in ["critical_high", "critical_low", "high", "low"],
            "literacy_level": health_literacy_level,
        }
    
    def _alert_provider(
        self, provider_id: str, patient_id: str, priority: str, alert_message: str,
        critical_values: list[str] = None
    ) -> dict:
        """Alert ordering provider."""
        alert = {
            "alert_id": f"ALERT-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "provider_id": provider_id,
            "patient_id": patient_id,
            "priority": priority,
            "message": alert_message,
            "critical_values": critical_values or [],
            "sent_at": datetime.utcnow().isoformat(),
            "acknowledged": False,
            "channel": "pager" if priority == "stat" else "inbox",
        }
        logger.info(f"Provider alert sent: {alert['alert_id']} (priority: {priority})")
        return alert
    
    def _update_care_plan(
        self, patient_id: str, findings: str, recommended_changes: list[str],
        follow_up_labs: list[str] = None, follow_up_timeframe: str = None
    ) -> dict:
        """Recommend care plan updates."""
        return {
            "patient_id": patient_id,
            "care_plan_update": {
                "findings": findings,
                "changes": recommended_changes,
                "follow_up_labs": follow_up_labs or [],
                "follow_up_timeframe": follow_up_timeframe or "4 weeks",
                "status": "pending_provider_approval",
                "updated_at": datetime.utcnow().isoformat(),
            }
        }
    
    def _notify_patient(
        self, patient_id: str, channel: str, subject: str, body: str,
        include_values: bool = True, action_required: bool = False
    ) -> dict:
        """Send patient notification."""
        return {
            "notification_id": f"NOTIF-LAB-{patient_id[:8]}-{datetime.utcnow().strftime('%H%M%S')}",
            "patient_id": patient_id,
            "channel": channel,
            "subject": subject,
            "body": body,
            "action_required": action_required,
            "sent_at": datetime.utcnow().isoformat(),
            "status": "delivered",
        }


# ============================================================
# Lab Processing Agent
# ============================================================

class LabProcessingAgent:
    """
    Autonomous Lab Result Processing Agent.
    
    Processes incoming lab results through a complete pipeline:
    parse → flag → compare baseline → interpret → alert → notify
    """
    
    SYSTEM_PROMPT = """You are an autonomous lab result processing agent. Your role is to 
interpret laboratory results and take appropriate actions.

WORKFLOW:
1. parse_lab_results — Flag any values outside reference ranges
2. For each abnormal: check_critical_value — Determine if immediate action needed
3. compare_with_baseline — Check for significant changes from patient's history
4. generate_patient_explanation — Create plain-language summary for patient
5. If critical values: alert_provider immediately (priority: stat)
6. If abnormal but not critical: alert_provider (priority: routine)
7. If results suggest care plan changes: update_care_plan
8. notify_patient with results summary

RULES:
- Critical values ALWAYS trigger immediate provider alert (stat priority)
- Never withhold results from patients (transparency)
- Adapt explanations to patient health literacy level
- Suggest follow-up labs when trends are concerning
- Include actionable recommendations (diet, medication timing, etc.)
"""
    
    def __init__(self, qwen_client: Optional[QwenCloudClient] = None):
        self.qwen = qwen_client or QwenCloudClient()
        self.tool_executor = LabToolExecutor()
    
    def run(self, lab_order: LabOrder) -> LabInterpretation:
        """Execute the full lab processing pipeline."""
        logger.info(f"Processing lab order: {lab_order.order_id}")
        
        # Build context for Qwen
        context = self._build_context(lab_order)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        
        response = self.qwen.function_call_loop(
            messages=messages,
            tools=LAB_TOOLS,
            tool_executor=self.tool_executor.execute,
            task_type="lab_interpretation",
            max_iterations=10,
        )
        
        return LabInterpretation(
            order_id=lab_order.order_id,
            patient_id=lab_order.patient_id,
            overall_assessment=response.content,
            alert_priority=AlertPriority.ROUTINE,
        )
    
    def _build_context(self, order: LabOrder) -> str:
        """Build structured context for Qwen interpretation."""
        tests_str = "\n".join(
            f"  - {t.test_name}: {t.value} {t.unit} (ref: {t.reference_low}-{t.reference_high})"
            for t in order.tests
        )
        
        return f"""LAB RESULTS TO PROCESS

Order ID: {order.order_id}
Patient ID: {order.patient_id}
Panel: {order.panel_name}
Ordering Provider: {order.ordering_provider}
Result Date: {order.result_date}

RESULTS:
{tests_str}

Please process these results through the complete pipeline:
1. Parse and flag abnormals
2. Check for critical values
3. Compare with patient baseline
4. Generate patient explanation
5. Alert provider if needed
6. Update care plan if warranted
7. Notify patient with results
"""
