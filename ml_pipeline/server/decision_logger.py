"""
Decision Logger - Tracks agent decisions in memory

Stores recent agent decisions for debugging and UI display.
Each user has their own decision history (last N decisions).
"""

import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class AgentDecision:
    """Single agent decision record"""
    timestamp: datetime
    action: str  # "HOLD", "ENTER", "EXIT"
    opportunity_index: Optional[int] = None
    opportunity_symbol: Optional[str] = None
    position_index: Optional[int] = None
    confidence: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"
    enter_probability: Optional[float] = None
    exit_probability: Optional[float] = None
    hold_probability: Optional[float] = None
    state_value: Optional[float] = None
    expected_apr: Optional[float] = None
    portfolio_utilization: Optional[float] = None
    num_opportunities: int = 0
    num_positions: int = 0
    reasoning: Optional[str] = None

    # Execution tracking fields
    execution_status: str = "pending"  # "pending", "filled", "failed"
    execution_id: Optional[int] = None  # Backend execution ID
    filled_price: Optional[float] = None
    filled_amount_usd: Optional[float] = None
    error_message: Optional[str] = None

    # EXIT-specific fields
    profit_usd: Optional[float] = None
    profit_pct: Optional[float] = None
    duration_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'opportunity_index': self.opportunity_index,
            'opportunity_symbol': self.opportunity_symbol,
            'position_index': self.position_index,
            'confidence': self.confidence,
            'enter_probability': self.enter_probability,
            'exit_probability': self.exit_probability,
            'hold_probability': self.hold_probability,
            'state_value': self.state_value,
            'expected_apr': self.expected_apr,
            'portfolio_utilization': self.portfolio_utilization,
            'num_opportunities': self.num_opportunities,
            'num_positions': self.num_positions,
            'reasoning': self.reasoning,
            'execution_status': self.execution_status,
            'execution_id': self.execution_id,
            'filled_price': self.filled_price,
            'filled_amount_usd': self.filled_amount_usd,
            'error_message': self.error_message,
            'profit_usd': self.profit_usd,
            'profit_pct': self.profit_pct,
            'duration_hours': self.duration_hours
        }


class DecisionLogger:
    """
    Manages agent decision history.

    Thread-safe logger that stores recent decisions for each user.
    Uses deque for efficient FIFO storage with max size limit.
    """

    def __init__(self, max_decisions_per_user: int = 1000):
        """
        Initialize decision logger.

        Args:
            max_decisions_per_user: Maximum decisions to store per user (default: 1000)
        """
        self._max_decisions = max_decisions_per_user
        self._decisions: Dict[str, deque] = {}  # user_id -> deque of AgentDecision
        self._lock = threading.Lock()

    def log_decision(
        self,
        user_id: str,
        action: str,
        prediction: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ):
        """
        Log a new agent decision.

        Args:
            user_id: Unique user identifier
            action: Action taken ("HOLD", "ENTER", "EXIT")
            prediction: Prediction dict from ML model
            portfolio_state: Current portfolio state
        """
        # Extract relevant info from prediction and portfolio
        opportunity_index = prediction.get('opportunity_index')
        opportunity_symbol = prediction.get('symbol')
        position_index = prediction.get('position_index')
        confidence = prediction.get('confidence')
        enter_prob = prediction.get('enter_probability')
        exit_prob = prediction.get('exit_probability')
        hold_prob = prediction.get('hold_probability')
        state_value = prediction.get('state_value')

        # Extract portfolio info
        num_opportunities = portfolio_state.get('num_opportunities', 0)
        num_positions = portfolio_state.get('num_positions', 0)
        utilization = portfolio_state.get('utilization', 0.0)

        # Build reasoning
        if action == "ENTER" and opportunity_symbol:
            expected_apr = portfolio_state.get('top_opportunity_apr', 0.0)
            reasoning = f"Entering {opportunity_symbol} (APR: {expected_apr:.1f}%, Confidence: {confidence})"
        elif action == "EXIT" and position_index is not None:
            reasoning = f"Exiting position #{position_index + 1} (Confidence: {confidence})"
        else:
            reasoning = f"Holding (Confidence: {confidence if confidence else 'N/A'})"

        decision = AgentDecision(
            timestamp=datetime.utcnow(),
            action=action,
            opportunity_index=opportunity_index,
            opportunity_symbol=opportunity_symbol,
            position_index=position_index,
            confidence=confidence,
            enter_probability=enter_prob,
            exit_probability=exit_prob,
            hold_probability=hold_prob,
            state_value=state_value,
            expected_apr=portfolio_state.get('top_opportunity_apr'),
            portfolio_utilization=utilization,
            num_opportunities=num_opportunities,
            num_positions=num_positions,
            reasoning=reasoning
        )

        with self._lock:
            # Create deque for user if doesn't exist
            if user_id not in self._decisions:
                self._decisions[user_id] = deque(maxlen=self._max_decisions)

            # Add decision (deque automatically removes oldest if at max)
            self._decisions[user_id].append(decision)

        print(f"[DecisionLogger] Logged decision for user {user_id}: {action} ({reasoning})")

    def get_recent_decisions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent decisions for user.

        Args:
            user_id: Unique user identifier
            limit: Maximum number of decisions to return (default: 100)

        Returns:
            List of decision dicts (most recent first)
        """
        with self._lock:
            if user_id not in self._decisions:
                return []

            # Get decisions (most recent first)
            decisions = list(self._decisions[user_id])
            decisions.reverse()  # Reverse to get newest first

            # Limit results
            decisions = decisions[:limit]

            return [d.to_dict() for d in decisions]

    def get_decision_count(self, user_id: str) -> int:
        """
        Get total decision count for user.

        Args:
            user_id: Unique user identifier

        Returns:
            Number of decisions logged
        """
        with self._lock:
            if user_id not in self._decisions:
                return 0
            return len(self._decisions[user_id])

    def clear_decisions(self, user_id: str):
        """
        Clear all decisions for user.

        Args:
            user_id: Unique user identifier
        """
        with self._lock:
            if user_id in self._decisions:
                self._decisions[user_id].clear()
                print(f"[DecisionLogger] Cleared decisions for user {user_id}")

    def get_latest_decision(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent decision for user.

        Args:
            user_id: Unique user identifier

        Returns:
            Decision dict or None if no decisions
        """
        with self._lock:
            if user_id not in self._decisions or len(self._decisions[user_id]) == 0:
                return None

            # Get last item from deque
            latest = self._decisions[user_id][-1]
            return latest.to_dict()

    def get_decision_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get decision summary statistics for user.

        Args:
            user_id: Unique user identifier

        Returns:
            Summary dict with counts and percentages
        """
        with self._lock:
            if user_id not in self._decisions:
                return {
                    'total_decisions': 0,
                    'hold_count': 0,
                    'enter_count': 0,
                    'exit_count': 0,
                    'hold_pct': 0.0,
                    'enter_pct': 0.0,
                    'exit_pct': 0.0
                }

            decisions = list(self._decisions[user_id])
            total = len(decisions)

            if total == 0:
                return {
                    'total_decisions': 0,
                    'hold_count': 0,
                    'enter_count': 0,
                    'exit_count': 0,
                    'hold_pct': 0.0,
                    'enter_pct': 0.0,
                    'exit_pct': 0.0
                }

            hold_count = sum(1 for d in decisions if d.action == "HOLD")
            enter_count = sum(1 for d in decisions if d.action == "ENTER")
            exit_count = sum(1 for d in decisions if d.action == "EXIT")

            return {
                'total_decisions': total,
                'hold_count': hold_count,
                'enter_count': enter_count,
                'exit_count': exit_count,
                'hold_pct': (hold_count / total) * 100,
                'enter_pct': (enter_count / total) * 100,
                'exit_pct': (exit_count / total) * 100
            }

    def update_decision_execution(
        self,
        user_id: str,
        timestamp: datetime,
        execution_status: str,
        execution_id: Optional[int] = None,
        filled_price: Optional[float] = None,
        filled_amount_usd: Optional[float] = None,
        error_message: Optional[str] = None,
        profit_usd: Optional[float] = None,
        profit_pct: Optional[float] = None,
        duration_hours: Optional[float] = None
    ) -> bool:
        """
        Update a decision with execution results.

        Finds the most recent decision matching the timestamp (within 5 seconds)
        and updates its execution fields.

        Args:
            user_id: Unique user identifier
            timestamp: Timestamp of the decision to update
            execution_status: "filled", "failed", or "pending"
            execution_id: Backend execution ID (optional)
            filled_price: Price at which order was filled (optional)
            filled_amount_usd: USD amount filled (optional)
            error_message: Error message if failed (optional)
            profit_usd: Profit in USD for EXIT actions (optional)
            profit_pct: Profit percentage for EXIT actions (optional)
            duration_hours: Position duration for EXIT actions (optional)

        Returns:
            True if decision was found and updated, False otherwise
        """
        with self._lock:
            if user_id not in self._decisions:
                print(f"[DecisionLogger] No decisions found for user {user_id}")
                return False

            # Find decision matching timestamp (within 5 seconds tolerance)
            decisions_deque = self._decisions[user_id]
            for decision in reversed(decisions_deque):  # Search from most recent
                time_diff = abs((decision.timestamp - timestamp).total_seconds())
                if time_diff <= 5.0:
                    # Update execution fields
                    decision.execution_status = execution_status
                    if execution_id is not None:
                        decision.execution_id = execution_id
                    if filled_price is not None:
                        decision.filled_price = filled_price
                    if filled_amount_usd is not None:
                        decision.filled_amount_usd = filled_amount_usd
                    if error_message is not None:
                        decision.error_message = error_message
                    if profit_usd is not None:
                        decision.profit_usd = profit_usd
                    if profit_pct is not None:
                        decision.profit_pct = profit_pct
                    if duration_hours is not None:
                        decision.duration_hours = duration_hours

                    print(f"[DecisionLogger] Updated decision for user {user_id}: "
                          f"{decision.action} {decision.opportunity_symbol} -> {execution_status}")
                    return True

            print(f"[DecisionLogger] No matching decision found for timestamp {timestamp}")
            return False
