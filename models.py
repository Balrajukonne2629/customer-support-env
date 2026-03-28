"""
Typed Pydantic models for the Customer Support Triage OpenEnv environment.
Defines Observation, Action, and Reward schemas per OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    FEATURE_REQUEST = "feature_request"


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Department(str, Enum):
    BILLING_TEAM = "billing_team"
    TECH_SUPPORT = "tech_support"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_SUPPORT = "general_support"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    PRIORITIZE = "prioritize"
    ROUTE = "route"
    RESPOND = "respond"
    SKIP = "skip"


# ---------------------------------------------------------------------------
# Ticket data model
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    """Represents a single customer support ticket."""

    id: str
    subject: str
    body: str
    customer_tier: str = Field(
        description="Subscription tier: free | basic | premium | enterprise"
    )
    timestamp: str
    sla_hours: int = Field(
        default=24,
        description="Hours remaining until SLA breach",
    )
    previous_messages: List[str] = Field(
        default_factory=list,
        description="Prior conversation history for context",
    )


# ---------------------------------------------------------------------------
# Core OpenEnv typed models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""

    ticket: Optional[Ticket] = Field(
        default=None,
        description="Current ticket to process. None when episode is done.",
    )
    task_id: str = Field(description="Identifier of the active task (task1/task2/task3)")
    task_description: str = Field(description="Human-readable task objective")
    current_ticket_index: int = Field(description="Zero-based index in ticket queue")
    total_tickets: int = Field(description="Total tickets in this episode")
    current_step: int = Field(description="Step number within the episode")
    max_steps: int = Field(description="Maximum allowed steps")
    completed_actions: List[str] = Field(
        default_factory=list,
        description="Log of actions taken so far",
    )
    score_so_far: float = Field(
        default=0.0,
        description="Cumulative reward earned so far (0.0–1.0 normalised)",
    )
    valid_action_types: List[str] = Field(
        description="Which action types are valid on this step"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra task-specific context (pre-filled fields, hints, etc.)",
    )


class Action(BaseModel):
    """Agent action — only fields relevant to the chosen action_type need to be set."""

    action_type: ActionType = Field(description="Which action the agent is taking")
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Required for action_type=classify",
    )
    priority: Optional[Priority] = Field(
        default=None,
        description="Required for action_type=prioritize",
    )
    department: Optional[Department] = Field(
        default=None,
        description="Required for action_type=route",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Required for action_type=respond — the reply message to the customer",
    )


class Reward(BaseModel):
    """Per-step reward with structured breakdown for interpretability."""

    value: float = Field(description="Scalar reward for this step (range: -0.2 to 1.0)")
    reason: str = Field(description="Human-readable explanation of why this reward was given")
    partial_credit: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward components for transparency",
    )
    is_terminal: bool = Field(
        default=False,
        description="Whether this reward comes at the end of the episode",
    )


# ---------------------------------------------------------------------------
# Step result & state
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result returned by env.step()."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full serialisable state returned by env.state()."""

    task_id: str
    episode_id: str
    tickets: List[Ticket]
    current_ticket_index: int
    step: int
    max_steps: int
    total_reward: float
    done: bool
    action_log: List[Dict[str, Any]]
    grader_scores: Dict[str, float]
