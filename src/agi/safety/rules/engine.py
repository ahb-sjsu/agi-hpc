# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Enhanced Safety Rule Engine with YAML Loading for AGI-HPC.

This is the central place where safety policies are:
- loaded from YAML config / rule files
- applied to plans, runtime signals, and outcomes
- produce structured verdicts: allow / block / revise, with reasons and risk scores

Implements Sprint 2 requirements:
- YAML-based rule definition and loading
- Rule DSL for complex constraints
- Rule priority and conflict resolution
- Hot-reload capability
"""

from __future__ import annotations

import logging
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class RuleLayer(Enum):
    """Safety rule layers with different latency requirements."""

    REFLEX = "reflex"  # <100μs - Hardware interlocks, emergency stops
    TACTICAL = "tactical"  # 10-100ms - Rule engine, constraint checking
    STRATEGIC = "strategic"  # 100ms-10s - ErisML ethical reasoning


class RuleAction(Enum):
    """Rule violation actions."""

    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    DEFER = "defer"


class ConditionOperator(Enum):
    """Supported condition operators."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    WITHIN = "within"
    OUTSIDE = "outside"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SafetyVerdict:
    """Result of a safety check."""

    decision: str  # "ALLOW", "BLOCK", "REVISE", "DEFER"
    risk_score: float  # 0.0 (no risk) → 1.0 (max risk)
    reasons: List[str]  # human-readable explanations
    violated_rules: List[str] = field(default_factory=list)
    modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleCondition:
    """A rule condition to evaluate."""

    field: Optional[str] = None
    metric: Optional[str] = None
    signal: Optional[str] = None
    operator: str = "=="
    value: Any = None
    range: Optional[Tuple[float, float]] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Get the value to check
        actual = None
        if self.field:
            actual = self._get_nested(context, self.field)
        elif self.metric:
            actual = context.get("metrics", {}).get(self.metric)
        elif self.signal:
            actual = context.get("signals", {}).get(self.signal)
        else:
            return True

        # Handle None case
        if actual is None:
            return False

        # Apply operator
        return self._apply_operator(actual)

    def _apply_operator(self, actual: Any) -> bool:
        """Apply the operator to compare actual value."""
        import re

        try:
            if self.operator == "==" or self.operator == "eq":
                return actual == self.value
            elif self.operator == "!=" or self.operator == "ne":
                return actual != self.value
            elif self.operator == ">" or self.operator == "gt":
                return actual > self.value
            elif self.operator == ">=" or self.operator == "ge":
                return actual >= self.value
            elif self.operator == "<" or self.operator == "lt":
                return actual < self.value
            elif self.operator == "<=" or self.operator == "le":
                return actual <= self.value
            elif self.operator == "within":
                if self.range:
                    return self.range[0] <= actual <= self.range[1]
                return False
            elif self.operator == "outside":
                if self.range:
                    return actual < self.range[0] or actual > self.range[1]
                return False
            elif self.operator == "in":
                return actual in self.value
            elif self.operator == "not_in":
                return actual not in self.value
            elif self.operator == "matches":
                return bool(re.match(self.value, str(actual)))
            elif self.operator == "contains":
                return self.value in str(actual)
            elif self.operator == "starts_with":
                return str(actual).startswith(self.value)
            elif self.operator == "ends_with":
                return str(actual).endswith(self.value)
            else:
                logger.warning("Unknown operator: %s", self.operator)
                return False
        except (TypeError, ValueError) as e:
            logger.debug("Condition evaluation error: %s", e)
            return False

    def _get_nested(self, obj: Dict, path: str) -> Any:
        """Get nested value by dot-separated path."""
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            elif isinstance(obj, list):
                try:
                    idx = int(key)
                    obj = obj[idx] if 0 <= idx < len(obj) else None
                except ValueError:
                    return None
            else:
                return None
        return obj


@dataclass
class ConditionGroup:
    """A group of conditions with AND/OR logic."""

    mode: str = "all"  # "all" (AND) or "any" (OR)
    conditions: List[Union[RuleCondition, "ConditionGroup"]] = field(
        default_factory=list
    )

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition group against context."""
        if not self.conditions:
            return True

        if self.mode == "all":
            return all(c.evaluate(context) for c in self.conditions)
        else:  # "any"
            return any(c.evaluate(context) for c in self.conditions)


@dataclass
class ViolationAction:
    """Action to take when rule is violated."""

    type: RuleAction
    reason: str = ""
    log_level: str = "warning"
    trigger_hardware_stop: bool = False
    require_human_approval: bool = False
    clip_to_bounds: bool = False
    scale_velocity: Optional[float] = None
    redistribute: bool = False
    custom_handler: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyRule:
    """A safety rule definition."""

    id: str
    name: str
    layer: RuleLayer
    priority: int
    conditions: ConditionGroup
    action: ViolationAction
    violation_action: ViolationAction
    enabled: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """Evaluate rule against context."""
        if not self.enabled:
            return RuleResult(rule_id=self.id, passed=True)

        # Evaluate conditions
        passed = self.conditions.evaluate(context)

        # Determine result
        result_action = self.action if passed else self.violation_action

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            passed=passed,
            action=result_action.type,
            reason=result_action.reason if not passed else "",
            params=result_action.params if not passed else {},
            trigger_hardware_stop=result_action.trigger_hardware_stop,
            require_human_approval=result_action.require_human_approval,
        )


@dataclass
class RuleResult:
    """Result of rule evaluation."""

    rule_id: str
    passed: bool
    rule_name: str = ""
    action: RuleAction = RuleAction.ALLOW
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    trigger_hardware_stop: bool = False
    require_human_approval: bool = False


@dataclass
class RuleCheckResult:
    """Aggregate result of checking multiple rules."""

    action: RuleAction
    violations: List[RuleResult]
    is_blocking: bool
    risk_score: float = 0.0
    modifications: Dict[str, Any] = field(default_factory=dict)
    require_human_approval: bool = False
    trigger_hardware_stop: bool = False


@dataclass
class RuleSetMetadata:
    """Metadata for a rule set."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    created_at: str = ""
    hash: str = ""


# ---------------------------------------------------------------------------
# Rule Parser
# ---------------------------------------------------------------------------


class RuleParser:
    """Parser for YAML rule definitions."""

    def parse_file(self, path: Path) -> Tuple[RuleSetMetadata, List[SafetyRule]]:
        """Parse rules from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse metadata
        metadata = self._parse_metadata(data, path)

        # Parse rules
        rules = []
        for rule_data in data.get("rules", []):
            try:
                rule = self._parse_rule(rule_data)
                rules.append(rule)
            except Exception as e:
                logger.error("Failed to parse rule %s: %s", rule_data.get("id"), e)

        return metadata, rules

    def _parse_metadata(self, data: Dict, path: Path) -> RuleSetMetadata:
        """Parse rule set metadata."""
        meta = data.get("metadata", {})

        # Calculate hash of the file content
        content_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return RuleSetMetadata(
            name=meta.get("name", path.stem),
            version=data.get("version", "1.0"),
            description=meta.get("description", ""),
            author=meta.get("author", ""),
            created_at=meta.get("created_at", ""),
            hash=content_hash,
        )

    def _parse_rule(self, data: Dict) -> SafetyRule:
        """Parse a single rule from YAML data."""
        # Parse conditions
        conditions = self._parse_conditions(data.get("conditions", {}))

        # Parse action (when conditions pass)
        action_data = data.get("action", {"type": "allow"})
        action = self._parse_action(action_data)

        # Parse violation action (when conditions fail)
        violation_data = data.get("violation_action", {"type": "deny"})
        violation_action = self._parse_action(violation_data)

        return SafetyRule(
            id=data["id"],
            name=data.get("name", data["id"]),
            layer=RuleLayer(data.get("layer", "tactical")),
            priority=data.get("priority", 100),
            conditions=conditions,
            action=action,
            violation_action=violation_action,
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    def _parse_conditions(self, cond_data: Dict) -> ConditionGroup:
        """Parse conditions from YAML data."""
        if not cond_data:
            return ConditionGroup(mode="all", conditions=[])

        # Check for grouped conditions
        if "all" in cond_data:
            mode = "all"
            cond_list = cond_data["all"]
        elif "any" in cond_data:
            mode = "any"
            cond_list = cond_data["any"]
        else:
            # Single condition or flat list
            mode = "all"
            cond_list = [cond_data] if isinstance(cond_data, dict) else cond_data

        conditions = []
        for c in cond_list:
            if isinstance(c, dict):
                if "all" in c or "any" in c:
                    # Nested condition group
                    conditions.append(self._parse_conditions(c))
                else:
                    # Single condition
                    conditions.append(self._parse_single_condition(c))

        return ConditionGroup(mode=mode, conditions=conditions)

    def _parse_single_condition(self, c: Dict) -> RuleCondition:
        """Parse a single condition."""
        range_val = None
        if "range" in c:
            range_val = tuple(c["range"])

        return RuleCondition(
            field=c.get("field"),
            metric=c.get("metric"),
            signal=c.get("signal"),
            operator=c.get("operator", "=="),
            value=c.get("value"),
            range=range_val,
        )

    def _parse_action(self, data: Dict) -> ViolationAction:
        """Parse an action from YAML data."""
        action_type = RuleAction(data.get("type", "allow"))

        return ViolationAction(
            type=action_type,
            reason=data.get("reason", ""),
            log_level=data.get("log_level", "warning"),
            trigger_hardware_stop=data.get("trigger_hardware_stop", False),
            require_human_approval=data.get("require_human_approval", False),
            clip_to_bounds=data.get("clip_to_bounds", False),
            scale_velocity=data.get("scale_velocity"),
            redistribute=data.get("redistribute", False),
            custom_handler=data.get("custom_handler"),
            params={
                k: v
                for k, v in data.items()
                if k
                not in [
                    "type",
                    "reason",
                    "log_level",
                    "trigger_hardware_stop",
                    "require_human_approval",
                    "clip_to_bounds",
                    "scale_velocity",
                    "redistribute",
                    "custom_handler",
                ]
            },
        )


# ---------------------------------------------------------------------------
# Safety Rule Engine
# ---------------------------------------------------------------------------


class SafetyRuleEngine:
    """
    Enhanced safety rule engine with YAML loading.

    Features:
    - Load rules from YAML files or directories
    - Three-layer rule evaluation (reflex, tactical, strategic)
    - Priority-based conflict resolution
    - Hot-reload capability
    - Rule validation and statistics
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize rule engine.

        Args:
            config: Optional ServiceConfig from config_loader.load_config
        """
        self.config = config
        self._rules: Dict[str, SafetyRule] = {}
        self._rules_by_layer: Dict[RuleLayer, List[SafetyRule]] = {
            layer: [] for layer in RuleLayer
        }
        self._rule_sets: Dict[str, RuleSetMetadata] = {}
        self._file_watchers: List = []
        self._parser = RuleParser()

        # Statistics
        self._stats = {
            "total_checks": 0,
            "allows": 0,
            "denies": 0,
            "defers": 0,
            "modifies": 0,
        }

        # Legacy support: load from config extra if available
        if config:
            self.rules_config: Dict[str, Any] = config.extra.get("safety_rules", {})
        else:
            self.rules_config = {}

    # ------------------------------------------------------------------
    # Rule Loading
    # ------------------------------------------------------------------

    def load_rules(self, path: Union[str, Path]) -> int:
        """
        Load rules from YAML file or directory.

        Args:
            path: Path to YAML file or directory containing YAML files

        Returns:
            Number of rules loaded
        """
        path = Path(path)
        count = 0

        if path.is_file():
            count += self._load_file(path)
        elif path.is_dir():
            for file in path.glob("**/*.yaml"):
                count += self._load_file(file)
            for file in path.glob("**/*.yml"):
                count += self._load_file(file)

        # Re-sort rules by priority (highest first)
        for layer in RuleLayer:
            self._rules_by_layer[layer].sort(key=lambda r: r.priority, reverse=True)

        logger.info("Loaded %d safety rules from %s", count, path)
        return count

    def _load_file(self, path: Path) -> int:
        """Load rules from a single YAML file."""
        try:
            metadata, rules = self._parser.parse_file(path)

            # Store metadata
            self._rule_sets[str(path)] = metadata

            # Add rules
            for rule in rules:
                if rule.id in self._rules:
                    # Remove old rule from layer list
                    old_rule = self._rules[rule.id]
                    self._rules_by_layer[old_rule.layer] = [
                        r
                        for r in self._rules_by_layer[old_rule.layer]
                        if r.id != rule.id
                    ]

                self._rules[rule.id] = rule
                self._rules_by_layer[rule.layer].append(rule)

            logger.debug("Loaded %d rules from %s", len(rules), path)
            return len(rules)

        except Exception as e:
            logger.error("Failed to load rules from %s: %s", path, e)
            return 0

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a rule programmatically."""
        self._rules[rule.id] = rule
        self._rules_by_layer[rule.layer].append(rule)
        self._rules_by_layer[rule.layer].sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        if rule_id in self._rules:
            rule = self._rules.pop(rule_id)
            self._rules_by_layer[rule.layer] = [
                r for r in self._rules_by_layer[rule.layer] if r.id != rule_id
            ]
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self, layer: Optional[RuleLayer] = None) -> List[SafetyRule]:
        """List all rules, optionally filtered by layer."""
        if layer:
            return self._rules_by_layer[layer].copy()
        return list(self._rules.values())

    # ------------------------------------------------------------------
    # Rule Evaluation
    # ------------------------------------------------------------------

    async def check_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        layer: Optional[RuleLayer] = None,
    ) -> RuleCheckResult:
        """
        Check action against rules.

        Args:
            action_type: Type of action (move, grasp, etc.)
            parameters: Action parameters
            context: Additional context (metrics, signals, etc.)
            layer: Optional specific layer to check

        Returns:
            RuleCheckResult with violations and final action
        """
        # Build full context
        full_context = {
            "action_type": action_type,
            "parameters": parameters,
            **context,
        }

        violations: List[RuleResult] = []
        final_action = RuleAction.ALLOW
        risk_score = 0.0
        modifications: Dict[str, Any] = {}
        require_human_approval = False
        trigger_hardware_stop = False

        # Check rules in priority order
        layers = [layer] if layer else list(RuleLayer)

        for check_layer in layers:
            for rule in self._rules_by_layer[check_layer]:
                result = rule.evaluate(full_context)

                if not result.passed:
                    violations.append(result)

                    # Accumulate risk score
                    risk_score = max(risk_score, self._calculate_risk(result, rule))

                    # Check for hardware stop
                    if result.trigger_hardware_stop:
                        trigger_hardware_stop = True

                    # Check for human approval
                    if result.require_human_approval:
                        require_human_approval = True

                    # Determine final action (DENY > DEFER > MODIFY > ALLOW)
                    if result.action == RuleAction.DENY:
                        final_action = RuleAction.DENY
                        break
                    elif (
                        result.action == RuleAction.DEFER
                        and final_action != RuleAction.DENY
                    ):
                        final_action = RuleAction.DEFER
                    elif (
                        result.action == RuleAction.MODIFY
                        and final_action == RuleAction.ALLOW
                    ):
                        final_action = RuleAction.MODIFY
                        # Collect modifications
                        modifications.update(result.params)

            if final_action == RuleAction.DENY:
                break

        # Update statistics
        self._stats["total_checks"] += 1
        self._stats[f"{final_action.value}s"] = (
            self._stats.get(f"{final_action.value}s", 0) + 1
        )

        return RuleCheckResult(
            action=final_action,
            violations=violations,
            is_blocking=(final_action == RuleAction.DENY),
            risk_score=risk_score,
            modifications=modifications,
            require_human_approval=require_human_approval,
            trigger_hardware_stop=trigger_hardware_stop,
        )

    def _calculate_risk(self, result: RuleResult, rule: SafetyRule) -> float:
        """Calculate risk score based on rule violation."""
        # Base risk by layer
        layer_risk = {
            RuleLayer.REFLEX: 1.0,
            RuleLayer.TACTICAL: 0.7,
            RuleLayer.STRATEGIC: 0.5,
        }

        # Base risk by action
        action_risk = {
            RuleAction.DENY: 1.0,
            RuleAction.DEFER: 0.6,
            RuleAction.MODIFY: 0.3,
            RuleAction.ALLOW: 0.0,
        }

        # Priority factor (higher priority = higher risk when violated)
        priority_factor = min(1.0, rule.priority / 1000.0)

        base_risk = layer_risk.get(rule.layer, 0.5)
        action_factor = action_risk.get(result.action, 0.5)

        return min(1.0, base_risk * 0.4 + action_factor * 0.4 + priority_factor * 0.2)

    # ------------------------------------------------------------------
    # Legacy API Support
    # ------------------------------------------------------------------

    def check_plan(self, plan_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        Check a plan summary (legacy API).

        Args:
            plan_summary: dict with high-level info about the proposed plan/code.
                Example fields:
                    - tools: List[str]
                    - resources: List[str]
                    - description: str
                    - estimated_impact: str

        Returns:
            SafetyVerdict with decision and risk score
        """
        reasons = []
        risk = 0.0
        violated_rules = []

        tools = plan_summary.get("tools", [])
        banned_tools = self.rules_config.get("banned_tools", [])

        for t in tools:
            if t in banned_tools:
                reasons.append(f"Tool '{t}' is banned by policy.")
                risk = max(risk, 1.0)
                violated_rules.append("banned_tools_policy")

        # Check against loaded rules if any
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        if self._rules:
            result = loop.run_until_complete(
                self.check_action(
                    action_type="plan",
                    parameters=plan_summary,
                    context={"plan_summary": plan_summary},
                )
            )
            if result.violations:
                for v in result.violations:
                    reasons.append(v.reason or f"Rule {v.rule_id} violated")
                    violated_rules.append(v.rule_id)
                risk = max(risk, result.risk_score)

        if not reasons:
            decision = "ALLOW"
        elif risk >= 0.9:
            decision = "BLOCK"
        else:
            decision = "REVISE"

        return SafetyVerdict(
            decision=decision,
            risk_score=risk,
            reasons=reasons,
            violated_rules=violated_rules,
        )

    def check_step(self, step_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        Check a runtime step (legacy API).

        Args:
            step_summary: dict describing one control step or short horizon.
                Example fields:
                    - predicted_collision: bool
                    - min_distance: float
                    - joint_limit_violations: int

        Returns:
            SafetyVerdict with decision and risk score
        """
        reasons = []
        risk = 0.0
        violated_rules = []

        if step_summary.get("predicted_collision", False):
            reasons.append("Predicted collision in next horizon.")
            risk = max(risk, 0.9)
            violated_rules.append("collision_check")

        if step_summary.get("joint_limit_violations", 0) > 0:
            reasons.append("Joint limits violated in simulation.")
            risk = max(risk, 0.8)
            violated_rules.append("joint_limits")

        # Check against loaded rules
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        if self._rules:
            result = loop.run_until_complete(
                self.check_action(
                    action_type="step",
                    parameters=step_summary,
                    context=step_summary,
                    layer=RuleLayer.TACTICAL,
                )
            )
            if result.violations:
                for v in result.violations:
                    reasons.append(v.reason or f"Rule {v.rule_id} violated")
                    violated_rules.append(v.rule_id)
                risk = max(risk, result.risk_score)

        decision = "ALLOW" if risk < 0.5 else "BLOCK"
        return SafetyVerdict(
            decision=decision,
            risk_score=risk,
            reasons=reasons,
            violated_rules=violated_rules,
        )

    def assess_outcome(self, outcome_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        Assess an action outcome (legacy API).

        Args:
            outcome_summary: high-level result of an episode or action.
                Example fields:
                    - incident: bool
                    - near_miss: bool
                    - unmodeled_effects: bool

        Returns:
            SafetyVerdict with decision and risk score
        """
        reasons = []
        risk = 0.0
        violated_rules = []

        if outcome_summary.get("incident", False):
            reasons.append("Actual safety incident occurred.")
            risk = max(risk, 1.0)
            violated_rules.append("incident_detection")

        if outcome_summary.get("near_miss", False):
            reasons.append("Near miss detected.")
            risk = max(risk, 0.7)
            violated_rules.append("near_miss_detection")

        if outcome_summary.get("unmodeled_effects", False):
            reasons.append("Unmodeled effects observed (model mismatch).")
            risk = max(risk, 0.8)
            violated_rules.append("model_mismatch")

        decision = "ALLOW" if risk < 0.4 else "REVISE"
        if risk >= 0.9:
            decision = "BLOCK"

        return SafetyVerdict(
            decision=decision,
            risk_score=risk,
            reasons=reasons,
            violated_rules=violated_rules,
        )

    # ------------------------------------------------------------------
    # Hot Reload
    # ------------------------------------------------------------------

    def enable_hot_reload(self, path: Union[str, Path]) -> None:
        """
        Enable hot-reload of rules on file changes.

        Args:
            path: Path to watch for changes
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.warning(
                "watchdog not installed. Hot-reload disabled. "
                "Install with: pip install watchdog"
            )
            return

        engine = self
        path = Path(path)

        class RuleReloader(FileSystemEventHandler):
            def __init__(self, engine: SafetyRuleEngine, path: Path):
                self.engine = engine
                self.path = path
                self._debounce_time = 0.0

            def on_modified(self, event):
                import time

                # Debounce to avoid multiple reloads
                current_time = time.time()
                if current_time - self._debounce_time < 1.0:
                    return
                self._debounce_time = current_time

                if event.src_path.endswith((".yaml", ".yml")):
                    logger.info("Reloading rules due to change: %s", event.src_path)
                    self.engine._rules.clear()
                    for layer in RuleLayer:
                        self.engine._rules_by_layer[layer].clear()
                    self.engine.load_rules(self.path)

        observer = Observer()
        observer.schedule(RuleReloader(engine, path), str(path), recursive=True)
        observer.start()
        self._file_watchers.append(observer)
        logger.info("Hot-reload enabled for %s", path)

    def disable_hot_reload(self) -> None:
        """Disable hot-reload and stop file watchers."""
        for observer in self._file_watchers:
            observer.stop()
            observer.join()
        self._file_watchers.clear()
        logger.info("Hot-reload disabled")

    # ------------------------------------------------------------------
    # Statistics and Validation
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        return {
            **self._stats,
            "rules_loaded": len(self._rules),
            "rules_by_layer": {
                layer.value: len(self._rules_by_layer[layer]) for layer in RuleLayer
            },
            "rule_sets": len(self._rule_sets),
        }

    def validate_rules(self) -> List[str]:
        """
        Validate all loaded rules.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for duplicate IDs
        seen_ids = set()
        for rule_id in self._rules:
            if rule_id in seen_ids:
                errors.append(f"Duplicate rule ID: {rule_id}")
            seen_ids.add(rule_id)

        # Check for circular references (if any)
        # Check for missing required fields
        for rule_id, rule in self._rules.items():
            if not rule.name:
                errors.append(f"Rule {rule_id} missing name")
            if rule.priority < 0 or rule.priority > 10000:
                errors.append(f"Rule {rule_id} has invalid priority: {rule.priority}")

        return errors

    def export_rules(self, path: Union[str, Path]) -> None:
        """Export current rules to YAML file."""
        path = Path(path)
        data = {
            "version": "1.0",
            "metadata": {
                "name": "Exported Rules",
                "description": "Exported from SafetyRuleEngine",
            },
            "rules": [],
        }

        for rule in self._rules.values():
            rule_data = {
                "id": rule.id,
                "name": rule.name,
                "layer": rule.layer.value,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "description": rule.description,
                "action": {"type": rule.action.type.value},
                "violation_action": {
                    "type": rule.violation_action.type.value,
                    "reason": rule.violation_action.reason,
                },
            }
            data["rules"].append(rule_data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Exported %d rules to %s", len(self._rules), path)
