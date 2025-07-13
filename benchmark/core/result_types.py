from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

@dataclass
class MetricResult:
    name: str
    value: float
    description: Optional[str] = None

@dataclass
class EvaluationResult:
    task_id: str
    agent_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # e.g., paths to generated files

@dataclass
class TaskResult:
    task_id: str
    agent_id: str
    evaluation: EvaluationResult
    notes: Optional[str] = None
