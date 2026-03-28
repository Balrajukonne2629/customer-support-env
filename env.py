"""
CustomerSupportEnv — Main OpenEnv environment class.

Implements the full OpenEnv interface:
  • reset(task_id)  → Observation
  • step(action)    → StepResult(observation, reward, done, info)
  • state()         → EnvironmentState

Real-world domain: SaaS customer support ticket triage & resolution.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .data import TASK_TICKET_MAP
from .graders import run_grader, _response_quality_score, _priority_score
from .models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
    Ticket,
)
from .tasks import TASK_DEFINITIONS, get_task


class CustomerSupportEnv:
    """
    OpenEnv-compliant environment for customer support ticket management.

    Three tasks of increasing difficulty:
      task1 (easy)   — classify tickets into categories
      task2 (medium) — assign priority and route to department
      task3 (hard)   — full resolution: classify + prioritize + route + respond
    """

    VERSION = "1.0.0"
    ENV_ID = "customer-support-triage-v1"

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._task_def: Optional[Dict[str, Any]] = None
        self._tickets: List[Dict[str, Any]] = []
        self._ticket_objects: List[Ticket] = []
        self._episode_id: str = ""
        self._step: int = 0
        self._max_steps: int = 0
        self._current_ticket_index: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0
        self._action_log: List[Dict[str, Any]] = []
        self._grader_scores: Dict[str, float] = {}

        # Per-ticket action tracking for task2 & task3
        self._ticket_action_counts: Dict[int, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """Initialise a new episode for the given task. Returns first observation."""
        self._task_id = task_id
        self._task_def = get_task(task_id)
        self._tickets = deepcopy(self._task_def["tickets"])
        self._ticket_objects = [self._make_ticket(t) for t in self._tickets]
        self._episode_id = str(uuid.uuid4())[:8]
        self._step = 0
        self._max_steps = self._task_def["max_steps"]
        self._current_ticket_index = 0
        self._done = False
        self._total_reward = 0.0
        self._action_log = []
        self._grader_scores = {}
        self._ticket_action_counts = {i: {} for i in range(len(self._tickets))}

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return the next observation, reward, done, info."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        reward, info = self._process_action(action)
        self._total_reward = round(self._total_reward + reward.value, 6)

        # Check episode termination
        if (
            self._current_ticket_index >= len(self._tickets)
            or self._step >= self._max_steps
        ):
            self._done = True
            # Run end-of-episode grader
            grader_result = run_grader(self._task_id, self._action_log, self._tickets)
            self._grader_scores["final"] = grader_result["score"]
            info["grader_result"] = grader_result
            reward = Reward(
                value=0.0,  # terminal reward already accounted for incrementally
                reason="Episode complete. See grader_result for final score.",
                is_terminal=True,
                partial_credit={"final_score": grader_result["score"]},
            )

        obs = self._build_observation()
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> EnvironmentState:
        """Return full serialisable environment state."""
        return EnvironmentState(
            task_id=self._task_id or "",
            episode_id=self._episode_id,
            tickets=[Ticket(**{k: v for k, v in t.items() if k in Ticket.model_fields})
                     for t in self._tickets],
            current_ticket_index=self._current_ticket_index,
            step=self._step,
            max_steps=self._max_steps,
            total_reward=self._total_reward,
            done=self._done,
            action_log=self._action_log,
            grader_scores=self._grader_scores,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_ticket(self, raw: Dict[str, Any]) -> Ticket:
        return Ticket(
            id=raw["id"],
            subject=raw["subject"],
            body=raw["body"],
            customer_tier=raw["customer_tier"],
            timestamp=raw["timestamp"],
            sla_hours=raw.get("sla_hours", 24),
            previous_messages=raw.get("previous_messages", []),
        )

    def _build_observation(self) -> Observation:
        ticket_obj = (
            self._ticket_objects[self._current_ticket_index]
            if self._current_ticket_index < len(self._ticket_objects)
            else None
        )
        task_def = self._task_def or {}
        valid_actions = task_def.get("valid_action_types", [])

        # For task2, pre-fill category in context
        context: Dict[str, Any] = {}
        if self._task_id == "task2" and ticket_obj is not None:
            raw = self._tickets[self._current_ticket_index]
            context["pre_filled_category"] = raw.get("category", "")
            context["hint"] = (
                "Category is already determined. Your job: set priority + route."
            )

        if ticket_obj is not None:
            context["ticket_number"] = self._current_ticket_index + 1
            context["total_tickets"] = len(self._tickets)

        return Observation(
            ticket=ticket_obj,
            task_id=self._task_id or "",
            task_description=task_def.get("description", ""),
            current_ticket_index=self._current_ticket_index,
            total_tickets=len(self._tickets),
            current_step=self._step,
            max_steps=self._max_steps,
            completed_actions=[
                f"[{a['ticket_id']}] {a['action_type']}" for a in self._action_log
            ],
            score_so_far=self._total_reward,
            valid_action_types=valid_actions,
            context=context,
        )

    def _process_action(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        """Route action to the correct handler and return (reward, info)."""
        if self._current_ticket_index >= len(self._tickets):
            return Reward(value=0.0, reason="No more tickets"), {}

        raw_ticket = self._tickets[self._current_ticket_index]
        ticket_id = raw_ticket["id"]
        action_dict = action.model_dump()
        action_dict["ticket_id"] = ticket_id
        action_dict["ticket_index"] = self._current_ticket_index

        if action.action_type == ActionType.SKIP:
            self._action_log.append(action_dict)
            reward = Reward(value=-0.02, reason="Skipped ticket (small penalty)")
            self._advance_ticket(action.action_type)
            return reward, {"skipped": ticket_id}

        if action.action_type == ActionType.CLASSIFY:
            return self._handle_classify(action, raw_ticket, action_dict)

        if action.action_type == ActionType.PRIORITIZE:
            return self._handle_prioritize(action, raw_ticket, action_dict)

        if action.action_type == ActionType.ROUTE:
            return self._handle_route(action, raw_ticket, action_dict)

        if action.action_type == ActionType.RESPOND:
            return self._handle_respond(action, raw_ticket, action_dict)

        return Reward(value=0.0, reason="Unknown action type"), {}

    def _handle_classify(self, action: Action, raw: Dict, action_dict: Dict) -> tuple[Reward, Dict]:
        self._action_log.append(action_dict)
        self._track_action(ActionType.CLASSIFY)

        predicted = action.category.value if action.category else ""
        expected = raw["correct_category"]
        cfg = self._task_def["reward_config"]

        if predicted == expected:
            reward_val = cfg["correct_classify"]
            reason = f"✓ Correct category '{predicted}'"
        else:
            reward_val = cfg.get("wrong_classify", -0.05)
            reason = f"✗ Wrong category '{predicted}' (expected '{expected}')"

        reward = Reward(
            value=reward_val,
            reason=reason,
            partial_credit={"category_correct": 1.0 if predicted == expected else 0.0},
        )

        if self._task_id == "task1":
            self._advance_ticket(ActionType.CLASSIFY)

        return reward, {"predicted": predicted, "expected": expected}

    def _handle_prioritize(self, action: Action, raw: Dict, action_dict: Dict) -> tuple[Reward, Dict]:
        self._action_log.append(action_dict)
        self._track_action(ActionType.PRIORITIZE)

        predicted = action.priority.value if action.priority else ""
        expected = raw["correct_priority"]
        cfg = self._task_def["reward_config"]

        p_score = _priority_score(predicted, expected)
        if p_score == 1.0:
            reward_val = cfg.get("correct_priority", 0.1)
            reason = f"✓ Correct priority '{predicted}'"
        elif p_score == 0.5:
            reward_val = cfg.get("adjacent_priority", 0.05)
            reason = f"~ Adjacent priority '{predicted}' (expected '{expected}')"
        else:
            reward_val = cfg.get("wrong_priority", -0.03)
            reason = f"✗ Wrong priority '{predicted}' (expected '{expected}')"

        reward = Reward(
            value=reward_val,
            reason=reason,
            partial_credit={"priority_score": p_score},
        )
        return reward, {"predicted": predicted, "expected": expected, "score": p_score}

    def _handle_route(self, action: Action, raw: Dict, action_dict: Dict) -> tuple[Reward, Dict]:
        self._action_log.append(action_dict)
        self._track_action(ActionType.ROUTE)

        predicted = action.department.value if action.department else ""
        expected = raw["correct_department"]
        cfg = self._task_def["reward_config"]

        if predicted == expected:
            reward_val = cfg.get("correct_routing", 0.1)
            reason = f"✓ Correct routing to '{predicted}'"
        else:
            reward_val = cfg.get("wrong_routing", -0.03)
            reason = f"✗ Wrong routing '{predicted}' (expected '{expected}')"

        reward = Reward(
            value=reward_val,
            reason=reason,
            partial_credit={"routing_correct": 1.0 if predicted == expected else 0.0},
        )

        if self._task_id == "task2":
            # Advance after both prioritize and route
            counts = self._ticket_action_counts[self._current_ticket_index]
            has_priority = counts.get("prioritize", 0) > 0
            has_route = counts.get("route", 0) > 0
            if has_priority and has_route:
                self._advance_ticket(ActionType.ROUTE)

        return reward, {"predicted": predicted, "expected": expected}

    def _handle_respond(self, action: Action, raw: Dict, action_dict: Dict) -> tuple[Reward, Dict]:
        self._action_log.append(action_dict)
        self._track_action(ActionType.RESPOND)

        resp = action.response_text or ""
        keywords = raw.get("response_keywords", [])
        min_words = raw.get("response_min_words", 60)
        cfg = self._task_def["reward_config"]

        resp_score, breakdown = _response_quality_score(resp, keywords, min_words)
        reward_val = round(cfg.get("response_quality_max", 0.133) * resp_score, 6)

        reward = Reward(
            value=reward_val,
            reason=f"Response quality: {resp_score:.2f}",
            partial_credit=breakdown,
        )

        if self._task_id == "task3":
            # Advance after all 4 actions
            counts = self._ticket_action_counts[self._current_ticket_index]
            all_done = all(
                counts.get(a, 0) > 0
                for a in ["classify", "prioritize", "route", "respond"]
            )
            if all_done:
                self._advance_ticket(ActionType.RESPOND)

        return reward, {"response_score": resp_score, "breakdown": breakdown}

    def _advance_ticket(self, action_type: ActionType) -> None:
        self._current_ticket_index += 1
        if self._current_ticket_index < len(self._tickets):
            self._ticket_action_counts[self._current_ticket_index] = {}

    def _track_action(self, action_type: ActionType) -> None:
        idx = self._current_ticket_index
        key = action_type.value
        self._ticket_action_counts.setdefault(idx, {})[key] = (
            self._ticket_action_counts.get(idx, {}).get(key, 0) + 1
        )
