"""
Curriculum Learning Scheduler

Implements 3-phase curriculum learning as specified in IMPLEMENTATION_PLAN.md (lines 793-816):

Phase 1: Simple Config (Episodes 0-500)
- Fixed simple configuration for basic learning
- Episode length: 3 days (72 hours)

Phase 2: Variable Config (Episodes 500-1500)
- Moderate config sampling for generalization
- Episode length: 5 days (120 hours)

Phase 3: Full System (Episodes 1500-3000+)
- Full config range for robust performance
- Episode length: 7 days (168 hours)
"""

from dataclasses import dataclass
from typing import Tuple
from .config import TradingConfig


@dataclass
class CurriculumPhase:
    """Represents a single curriculum phase."""
    name: str
    episode_start: int
    episode_end: int
    episode_length_hours: int
    description: str


class CurriculumScheduler:
    """
    Manages curriculum learning progression.

    Automatically adjusts training difficulty based on episode number:
    - Phase 1: Simple, fixed config
    - Phase 2: Moderate, variable config
    - Phase 3: Full, complex config
    """

    def __init__(self,
                 phase1_end: int = 500,
                 phase2_end: int = 1500,
                 phase3_end: int = 3000):
        """
        Initialize curriculum scheduler.

        Args:
            phase1_end: Episode when Phase 1 ends (default: 500)
            phase2_end: Episode when Phase 2 ends (default: 1500)
            phase3_end: Episode when Phase 3 ends (default: 3000, can be infinite)
        """
        self.phases = [
            CurriculumPhase(
                name='simple',
                episode_start=0,
                episode_end=phase1_end,
                episode_length_hours=72,  # 3 days
                description='Basic multi-position management with fixed simple config'
            ),
            CurriculumPhase(
                name='variable',
                episode_start=phase1_end,
                episode_end=phase2_end,
                episode_length_hours=120,  # 5 days
                description='Generalization across moderate configs'
            ),
            CurriculumPhase(
                name='full',
                episode_start=phase2_end,
                episode_end=phase3_end,
                episode_length_hours=168,  # 7 days
                description='Robust performance across full config range'
            ),
        ]

    def get_current_phase(self, episode: int) -> CurriculumPhase:
        """
        Get the current curriculum phase for an episode.

        Args:
            episode: Current episode number

        Returns:
            CurriculumPhase object
        """
        for phase in self.phases:
            if phase.episode_start <= episode < phase.episode_end:
                return phase

        # If past all phases, return the last phase
        return self.phases[-1]

    def get_config(self, episode: int) -> TradingConfig:
        """
        Get trading config for current episode based on curriculum phase.

        V3 FIXED CONFIGS: Using stable configs instead of random sampling
        for better learning stability and reproducibility.

        Args:
            episode: Current episode number

        Returns:
            TradingConfig instance
        """
        phase = self.get_current_phase(episode)

        if phase.name == 'simple':
            # Phase 1: Simple - Learn basics
            # 2 positions, conservative settings
            return TradingConfig(
                max_leverage=1.5,
                target_utilization=0.6,
                max_positions=2,
                stop_loss_threshold=-0.03,
                liquidation_buffer=0.15,
            )

        elif phase.name == 'variable':
            # Phase 2: Intermediate - Scale up gradually
            # 2 positions, moderate settings (this is where your 105% model likely was)
            return TradingConfig(
                max_leverage=2.0,
                target_utilization=0.8,
                max_positions=2,
                stop_loss_threshold=-0.02,
                liquidation_buffer=0.15,
            )

        else:  # 'full'
            # Phase 3: Advanced - Production-like settings
            # 3 positions, more aggressive
            return TradingConfig(
                max_leverage=2.5,
                target_utilization=0.8,
                max_positions=3,
                stop_loss_threshold=-0.02,
                liquidation_buffer=0.10,
            )

    def get_episode_length_hours(self, episode: int) -> int:
        """
        Get episode length in hours for current phase.

        Args:
            episode: Current episode number

        Returns:
            Episode length in hours
        """
        phase = self.get_current_phase(episode)
        return phase.episode_length_hours

    def get_episode_length_days(self, episode: int) -> int:
        """
        Get episode length in days for current phase.

        Args:
            episode: Current episode number

        Returns:
            Episode length in days
        """
        return self.get_episode_length_hours(episode) // 24

    def get_phase_progress(self, episode: int) -> Tuple[str, float]:
        """
        Get current phase name and progress percentage.

        Args:
            episode: Current episode number

        Returns:
            Tuple of (phase_name, progress_pct)
        """
        phase = self.get_current_phase(episode)
        phase_length = phase.episode_end - phase.episode_start
        episode_in_phase = episode - phase.episode_start
        progress_pct = (episode_in_phase / phase_length) * 100

        return phase.name, progress_pct

    def print_phase_info(self, episode: int):
        """Print current phase information."""
        phase = self.get_current_phase(episode)
        phase_name, progress = self.get_phase_progress(episode)

        print(f"Episode {episode:4d} | Phase: {phase_name:8s} ({progress:5.1f}%) | {phase.description}")
        print(f"             | Episode length: {phase.episode_length_hours}h ({phase.episode_length_hours // 24} days)")

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(\n"
            f"  Phase 1: Episodes 0-{self.phases[0].episode_end} (72h episodes)\n"
            f"  Phase 2: Episodes {self.phases[1].episode_start}-{self.phases[1].episode_end} (120h episodes)\n"
            f"  Phase 3: Episodes {self.phases[2].episode_start}+ (168h episodes)\n"
            f")"
        )


# Success criteria for each phase (optional, for monitoring)
PHASE_SUCCESS_CRITERIA = {
    'simple': {
        'mean_reward_100': 20.0,  # Mean reward over last 100 episodes
        'description': 'Agent learns basic multi-position management'
    },
    'variable': {
        'mean_reward_100': 30.0,
        'positive_return_rate': 0.8,  # 80% of episodes have positive return
        'description': 'Agent generalizes across moderate configs'
    },
    'full': {
        'sharpe_ratio': 1.5,
        'max_drawdown_pct': 20.0,
        'description': 'Robust performance across all configs'
    },
}
