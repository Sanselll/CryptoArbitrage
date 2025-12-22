"""
Agent Manager - Handles autonomous trading agent sessions

Manages agent state, configuration, and execution loop for each user.
Supports multiple concurrent agents (one per user).
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Agent trading configuration (V9: single position only)"""
    max_leverage: float = 1.0          # 1-5x
    target_utilization: float = 0.9    # 50-100%
    max_positions: int = 1             # V9: single position only
    prediction_interval_sec: int = 60  # Seconds between predictions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        return cls(
            max_leverage=data.get('max_leverage', 1.0),
            target_utilization=data.get('target_utilization', 0.9),
            max_positions=1,  # V9: force single position
            prediction_interval_sec=data.get('prediction_interval_sec', 60)
        )

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate config ranges. Returns (is_valid, error_message)"""
        if not (1.0 <= self.max_leverage <= 5.0):
            return False, "max_leverage must be between 1.0 and 5.0"
        if not (0.5 <= self.target_utilization <= 1.0):
            return False, "target_utilization must be between 0.5 and 1.0"
        if self.max_positions != 1:  # V9: single position only
            return False, "max_positions must be 1 (V9: single position mode)"
        if not (10 <= self.prediction_interval_sec <= 300):
            return False, "prediction_interval_sec must be between 10 and 300"
        return True, None


@dataclass
class AgentSession:
    """Active agent session state"""
    user_id: str
    status: AgentStatus
    config: AgentConfig
    start_time: Optional[datetime] = None
    pause_time: Optional[datetime] = None
    error_message: Optional[str] = None
    prediction_count: int = 0
    last_prediction_time: Optional[datetime] = None

    def get_duration_seconds(self) -> int:
        """Get total running duration in seconds"""
        if not self.start_time:
            return 0

        if self.status == AgentStatus.RUNNING:
            return int((datetime.utcnow() - self.start_time).total_seconds())
        elif self.status == AgentStatus.PAUSED and self.pause_time:
            return int((self.pause_time - self.start_time).total_seconds())
        elif self.status == AgentStatus.STOPPED:
            return 0
        else:
            return int((datetime.utcnow() - self.start_time).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'status': self.status.value,
            'config': self.config.to_dict(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'pause_time': self.pause_time.isoformat() if self.pause_time else None,
            'duration_seconds': self.get_duration_seconds(),
            'error_message': self.error_message,
            'prediction_count': self.prediction_count,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }


class AgentManager:
    """
    Manages autonomous trading agent sessions.

    Thread-safe manager for multiple concurrent agents (one per user).
    Each agent runs in its own thread with continuous prediction loop.
    """

    def __init__(self):
        self._sessions: Dict[str, AgentSession] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start_agent(self, user_id: str, config: AgentConfig) -> tuple[bool, Optional[str]]:
        """
        Start agent for user.

        Args:
            user_id: Unique user identifier
            config: Agent configuration

        Returns:
            (success, error_message)
        """
        # Validate config
        is_valid, error_msg = config.validate()
        if not is_valid:
            return False, f"Invalid configuration: {error_msg}"

        with self._lock:
            # Check if agent already running
            if user_id in self._sessions:
                session = self._sessions[user_id]
                if session.status == AgentStatus.RUNNING:
                    return False, "Agent already running for this user"
                elif session.status == AgentStatus.PAUSED:
                    return False, "Agent is paused. Use /resume to continue"

            # Create new session
            session = AgentSession(
                user_id=user_id,
                status=AgentStatus.RUNNING,
                config=config,
                start_time=datetime.utcnow()
            )

            self._sessions[user_id] = session

            # Create stop event for thread
            stop_event = threading.Event()
            self._stop_events[user_id] = stop_event

            # Start agent thread (prediction loop will be implemented by caller)
            # Note: The actual prediction loop will be implemented in app.py
            # This manager just tracks state

            print(f"[AgentManager] Started agent for user {user_id}")
            return True, None

    def stop_agent(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Stop agent for user.

        Args:
            user_id: Unique user identifier

        Returns:
            (success, error_message)
        """
        with self._lock:
            if user_id not in self._sessions:
                return False, "No active agent found for this user"

            session = self._sessions[user_id]

            if session.status == AgentStatus.STOPPED:
                return False, "Agent is already stopped"

            # Signal thread to stop
            if user_id in self._stop_events:
                self._stop_events[user_id].set()

            # Update status
            session.status = AgentStatus.STOPPED
            session.pause_time = None
            session.error_message = None

            print(f"[AgentManager] Stopped agent for user {user_id}")
            return True, None

    def pause_agent(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Pause agent for user.

        Args:
            user_id: Unique user identifier

        Returns:
            (success, error_message)
        """
        with self._lock:
            if user_id not in self._sessions:
                return False, "No active agent found for this user"

            session = self._sessions[user_id]

            if session.status != AgentStatus.RUNNING:
                return False, f"Cannot pause agent with status: {session.status}"

            session.status = AgentStatus.PAUSED
            session.pause_time = datetime.utcnow()

            print(f"[AgentManager] Paused agent for user {user_id}")
            return True, None

    def resume_agent(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Resume paused agent for user.

        Args:
            user_id: Unique user identifier

        Returns:
            (success, error_message)
        """
        with self._lock:
            if user_id not in self._sessions:
                return False, "No active agent found for this user"

            session = self._sessions[user_id]

            if session.status != AgentStatus.PAUSED:
                return False, f"Cannot resume agent with status: {session.status}"

            session.status = AgentStatus.RUNNING
            session.pause_time = None

            print(f"[AgentManager] Resumed agent for user {user_id}")
            return True, None

    def update_config(self, user_id: str, config: AgentConfig) -> tuple[bool, Optional[str]]:
        """
        Update agent configuration (only allowed when stopped).

        Args:
            user_id: Unique user identifier
            config: New configuration

        Returns:
            (success, error_message)
        """
        # Validate config
        is_valid, error_msg = config.validate()
        if not is_valid:
            return False, f"Invalid configuration: {error_msg}"

        with self._lock:
            if user_id not in self._sessions:
                # Create new session with stopped status
                session = AgentSession(
                    user_id=user_id,
                    status=AgentStatus.STOPPED,
                    config=config
                )
                self._sessions[user_id] = session
                print(f"[AgentManager] Created new session with config for user {user_id}")
                return True, None

            session = self._sessions[user_id]

            # Only allow config update when stopped
            if session.status != AgentStatus.STOPPED:
                return False, "Agent must be stopped to update configuration"

            session.config = config
            print(f"[AgentManager] Updated config for user {user_id}")
            return True, None

    def set_error(self, user_id: str, error_message: str):
        """
        Set agent to error state.

        Args:
            user_id: Unique user identifier
            error_message: Error description
        """
        with self._lock:
            if user_id in self._sessions:
                session = self._sessions[user_id]
                session.status = AgentStatus.ERROR
                session.error_message = error_message
                print(f"[AgentManager] Agent error for user {user_id}: {error_message}")

    def update_prediction_time(self, user_id: str):
        """
        Update last prediction timestamp and increment count.

        Args:
            user_id: Unique user identifier
        """
        with self._lock:
            if user_id in self._sessions:
                session = self._sessions[user_id]
                session.last_prediction_time = datetime.utcnow()
                session.prediction_count += 1

    def get_session(self, user_id: str) -> Optional[AgentSession]:
        """
        Get agent session for user.

        Args:
            user_id: Unique user identifier

        Returns:
            AgentSession or None if not found
        """
        with self._lock:
            return self._sessions.get(user_id)

    def get_stop_event(self, user_id: str) -> Optional[threading.Event]:
        """
        Get stop event for agent thread.

        Args:
            user_id: Unique user identifier

        Returns:
            threading.Event or None
        """
        with self._lock:
            return self._stop_events.get(user_id)

    def is_running(self, user_id: str) -> bool:
        """
        Check if agent is running for user.

        Args:
            user_id: Unique user identifier

        Returns:
            True if running, False otherwise
        """
        with self._lock:
            if user_id not in self._sessions:
                return False
            return self._sessions[user_id].status == AgentStatus.RUNNING

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active sessions (for monitoring/debugging).

        Returns:
            Dict mapping user_id to session dict
        """
        with self._lock:
            return {
                user_id: session.to_dict()
                for user_id, session in self._sessions.items()
            }
