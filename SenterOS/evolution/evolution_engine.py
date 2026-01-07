#!/usr/bin/env python3
"""
SenterOS Evolution Engine
=========================

The self-optimizing brain of SenterOS. This engine continuously evolves
the system's configuration based on interaction outcomes.

The evolution engine implements natural selection at the configuration level:
- Mutation: Propose changes to configuration
- Selection: Choose beneficial changes based on fitness
- Inheritance: Pass successful changes forward

Key insight: The system doesn't just learn from interactions—it evolves
its own configuration to be better at learning.

Evolution Types:
================

1. PROMPT REFINEMENT
   - Adjust system prompts based on what works
   - Add/remove instructions based on outcomes
   - Tune tone and style

2. CAPABILITY EXPANSION
   - Add new capabilities based on user needs
   - Discover new tools automatically
   - Extend the genome

3. MEMORY PRUNING
   - Remove unused or outdated memories
   - Strengthen frequently accessed connections
   - Adjust importance scores

4. MODEL ROUTING
   - Learn which models work best for which tasks
   - Adjust parameters based on outcomes
   - Optimize for latency vs quality

5. INTERACTION PATTERN
   - Learn interaction patterns that work
   - Adjust initiative level
   - Tune explanation depth

Usage:
======

evolution = EvolutionEngine(genome=my_genome)

# After an interaction
changes = evolution.evolve(
    interaction={
        "input": "Hello",
        "response": "Hi there!",
        "outcome": "positive",
        "duration": 0.5
    },
    fitness_score=0.9
)

# Apply changes to genome
for change in changes:
    evolution.apply(change, genome)

# Get evolution statistics
stats = evolution.get_stats()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
import json
import uuid


@dataclass
class Mutation:
    """A proposed mutation to the genome."""

    id: str
    mutation_type: str  # prompt_refinement, capability_expansion, memory_pruning, etc.
    target: str  # What part of the genome to change
    current_value: Any
    proposed_value: Any
    rationale: str  # Why this mutation should help
    confidence: float = 0.5  # How confident we are in this mutation
    expected_improvement: float = 0.1  # Expected improvement if successful

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mutation_type": self.mutation_type,
            "target": self.target,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "expected_improvement": self.expected_improvement,
        }


@dataclass
class FitnessRecord:
    """A record of fitness for an interaction."""

    id: str
    timestamp: str
    interaction_type: str  # What kind of interaction
    fitness_score: float  # 0-1, how successful
    duration: float  # How long it took
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    mutations_applied: List[str] = field(
        default_factory=list
    )  # Mutations from this interaction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "interaction_type": self.interaction_type,
            "fitness_score": self.fitness_score,
            "duration": self.duration,
            "context": self.context,
            "mutations_applied": self.mutations_applied,
        }


@dataclass
class EvolutionStats:
    """Statistics about the evolution process."""

    total_interactions: int = 0
    total_mutations_proposed: int = 0
    total_mutations_applied: int = 0
    total_mutations_reverted: int = 0
    average_fitness: float = 0.5
    mutation_success_rate: float = 0.5  # What % of mutations improve fitness
    best_improvement: float = 0.0
    last_evolution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_interactions": self.total_interactions,
            "total_mutations_proposed": self.total_mutations_proposed,
            "total_mutations_applied": self.total_mutations_applied,
            "total_mutations_reverted": self.total_mutations_reverted,
            "average_fitness": self.average_fitness,
            "mutation_success_rate": self.mutation_success_rate,
            "best_improvement": self.best_improvement,
            "last_evolution": self.last_evolution,
        }


class EvolutionEngine:
    """
    The self-optimizing evolution engine for SenterOS.

    This engine implements natural selection at the configuration level:
    - Mutation: Propose changes to configuration based on outcomes
    - Selection: Choose beneficial changes based on fitness
    - Inheritance: Pass successful changes forward

    The key insight: The system doesn't just learn from interactions—
    it evolves its own configuration to be better at learning.

    Evolution Strategy:
    ===================

    1. ANALYZE: Look at interaction outcomes to identify improvement opportunities
    2. PROPOSE: Generate mutation candidates
    3. SELECT: Choose mutations based on expected improvement
    4. APPLY: Apply mutations to the genome
    5. EVALUATE: Track fitness to measure success
    6. REVERT: Remove mutations that don't help
    """

    def __init__(
        self,
        genome=None,
        storage_path: Optional[Path] = None,
        mutation_rate: float = 0.05,
        selection_pressure: str = "user_satisfaction",
    ):
        self.genome = genome
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure

        # History for learning
        self.fitness_history: List[FitnessRecord] = []
        self.mutation_history: List[Mutation] = []
        self.active_mutations: Dict[str, Mutation] = {}

        # Statistics
        self.stats = EvolutionStats()

        # Mutation strategies
        self.mutation_strategies: Dict[str, Callable] = {
            "prompt_refinement": self._mutate_prompt,
            "capability_expansion": self._mutate_capabilities,
            "memory_pruning": self._mutate_memory,
            "interaction_tuning": self._mutate_interaction,
            "model_optimization": self._mutate_model,
        }

    def _mutate_prompt(self, interaction: Dict[str, Any]) -> Mutation:
        """Propose a prompt refinement mutation."""
        return Mutation(
            id=str(uuid.uuid4()),
            mutation_type="prompt_refinement",
            target="system_prompt",
            current_value=self.genome.system_prompt if self.genome else "",
            proposed_value=self.genome.system_prompt + " Be more concise."
            if self.genome
            else "",
            rationale="Exploring efficiency improvements",
            confidence=0.3,
            expected_improvement=0.05,
        )

    def _mutate_capabilities(self, interaction: Dict[str, Any]) -> Mutation:
        """Propose a capability expansion mutation."""
        return Mutation(
            id=str(uuid.uuid4()),
            mutation_type="capability_expansion",
            target="capabilities",
            current_value=[],
            proposed_value=["new_capability"],
            rationale="Adding new capability based on interaction",
            confidence=0.4,
            expected_improvement=0.1,
        )

    def _mutate_memory(self, interaction: Dict[str, Any]) -> Mutation:
        """Propose a memory pruning mutation."""
        return Mutation(
            id=str(uuid.uuid4()),
            mutation_type="memory_pruning",
            target="evolution.memory_decay",
            current_value=0.01,
            proposed_value=0.02,
            rationale="Adjusting memory decay for efficiency",
            confidence=0.3,
            expected_improvement=0.05,
        )

    def _mutate_interaction(self, interaction: Dict[str, Any]) -> Mutation:
        """Propose an interaction tuning mutation."""
        return Mutation(
            id=str(uuid.uuid4()),
            mutation_type="interaction_tuning",
            target="interaction.initiative_level",
            current_value=0.5,
            proposed_value=0.6,
            rationale="Adjusting initiative level based on feedback",
            confidence=0.4,
            expected_improvement=0.08,
        )

    def _mutate_model(self, interaction: Dict[str, Any]) -> Mutation:
        """Propose a model optimization mutation."""
        return Mutation(
            id=str(uuid.uuid4()),
            mutation_type="model_optimization",
            target="model.temperature",
            current_value=0.7,
            proposed_value=0.8,
            rationale="Adjusting temperature for more creative responses",
            confidence=0.3,
            expected_improvement=0.05,
        )

        # Storage
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def evolve(
        self, interaction: Dict[str, Any], fitness_score: float
    ) -> List[Mutation]:
        """
        Main evolution step—analyze interaction and propose mutations.

        Args:
            interaction: The interaction data (input, response, outcome, etc.)
            fitness_score: How successful the interaction was (0-1)

        Returns:
            List of proposed mutations
        """
        self.stats.total_interactions += 1
        self.stats.last_evolution = datetime.now().isoformat()

        # Update average fitness
        n = self.stats.total_interactions
        old_avg = self.stats.average_fitness
        self.stats.average_fitness = ((old_avg * (n - 1)) + fitness_score) / n

        # Create fitness record
        record = FitnessRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            interaction_type=interaction.get("type", "general"),
            fitness_score=fitness_score,
            duration=interaction.get("duration", 0.0),
            context={
                "input_length": len(str(interaction.get("input", ""))),
                "response_length": len(str(interaction.get("response", ""))),
            },
        )
        self.fitness_history.append(record)

        # Track best improvement
        if fitness_score > 0.8:
            improvement = fitness_score - 0.5  # Compare to baseline
            if improvement > self.stats.best_improvement:
                self.stats.best_improvement = improvement

        # Propose mutations based on fitness
        mutations = []

        if fitness_score < 0.5:
            # Poor outcome—try to fix it
            mutations.extend(self._propose_fixes(interaction, fitness_score))
        elif fitness_score > 0.8:
            # Great outcome—reinforce what worked
            mutations.extend(self._propose_reinforcements(interaction, fitness_score))
        else:
            # Normal outcome—explore potential improvements
            if random.random() < self.mutation_rate:
                mutations.extend(self._propose_explorations(interaction))

        # Store mutations
        for mutation in mutations:
            self.stats.total_mutations_proposed += 1
            self.mutation_history.append(mutation)
            self.active_mutations[mutation.id] = mutation

        return mutations

    def _propose_fixes(
        self, interaction: Dict[str, Any], fitness_score: float
    ) -> List[Mutation]:
        """Propose mutations to fix poor outcomes."""
        mutations = []
        rationale_base = (
            f"Low fitness ({fitness_score:.2f}) suggests improvement opportunity"
        )

        # Analyze what went wrong
        response = str(interaction.get("response", ""))
        input_text = str(interaction.get("input", ""))

        # If response was too short
        if len(response) < 50 and len(input_text) > 100:
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="prompt_refinement",
                    target="system_prompt",
                    current_value="",
                    proposed_value="Provide more detailed responses when users ask complex questions.",
                    rationale=f"{rationale_base}: Response may be too brief for complex queries",
                    confidence=0.6,
                    expected_improvement=0.15,
                )
            )

        # If response was unclear
        if "sorry" in response.lower() or "don't understand" in response.lower():
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="capability_expansion",
                    target="capabilities",
                    current_value=[],
                    proposed_value=["ask_for_clarification"],
                    rationale=f"{rationale_base}: System struggles with ambiguous queries",
                    confidence=0.7,
                    expected_improvement=0.2,
                )
            )

        # If interaction was slow
        if interaction.get("duration", 0) > 5.0:
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="model_optimization",
                    target="model.n_ctx",
                    current_value=8192,
                    proposed_value=4096,
                    rationale=f"{rationale_base}: High latency suggests model optimization opportunity",
                    confidence=0.5,
                    expected_improvement=0.1,
                )
            )

        return mutations

    def _propose_reinforcements(
        self, interaction: Dict[str, Any], fitness_score: float
    ) -> List[Mutation]:
        """Propose mutations to reinforce successful outcomes."""
        mutations = []
        rationale_base = (
            f"High fitness ({fitness_score:.2f}) indicates effective behavior"
        )

        response = str(interaction.get("response", ""))

        # If response was helpful, note the style
        if fitness_score > 0.9:
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="interaction_tuning",
                    target="interaction.initiative_level",
                    current_value=0.5,
                    proposed_value=0.6,
                    rationale=f"{rationale_base}: User responds well to proactive suggestions",
                    confidence=0.6,
                    expected_improvement=0.05,
                )
            )

        return mutations

    def _propose_explorations(self, interaction: Dict[str, Any]) -> List[Mutation]:
        """Propose exploratory mutations."""
        mutations = []

        # Randomly explore different improvements
        strategies = list(self.mutation_strategies.keys())
        strategy = random.choice(strategies)

        if strategy == "prompt_refinement":
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="prompt_refinement",
                    target="system_prompt",
                    current_value="",
                    proposed_value="Be more concise in responses.",
                    rationale="Exploration: Testing brevity for efficiency",
                    confidence=0.3,
                    expected_improvement=0.02,
                )
            )
        elif strategy == "interaction_tuning":
            mutations.append(
                Mutation(
                    id=str(uuid.uuid4()),
                    mutation_type="interaction_tuning",
                    target="interaction.style",
                    current_value="collaborative",
                    proposed_value="socratic",
                    rationale="Exploration: Testing Socratic style for engagement",
                    confidence=0.3,
                    expected_improvement=0.02,
                )
            )

        return mutations

    def apply(self, mutation: Mutation, genome=None) -> bool:
        """
        Apply a mutation to the genome.

        Args:
            mutation: The mutation to apply
            genome: The genome to mutate (uses self.genome if not provided)

        Returns:
            True if successful, False otherwise
        """
        genome = genome or self.genome
        if not genome:
            return False

        try:
            # Apply based on target
            if mutation.target == "system_prompt":
                genome.system_prompt = mutation.proposed_value
            elif mutation.target == "capabilities":
                if mutation.proposed_value not in genome.phenotype.capabilities:
                    genome.phenotype.capabilities.append(mutation.proposed_value)
            elif mutation.target.startswith("interaction."):
                attr = mutation.target.split(".")[1]
                if hasattr(genome.interaction, attr):
                    setattr(genome.interaction, attr, mutation.proposed_value)
            elif mutation.target.startswith("model."):
                attr = mutation.target.split(".")[1]
                if attr in genome.phenotype.model:
                    genome.phenotype.model[attr] = mutation.proposed_value

            self.stats.total_mutations_applied += 1
            mutation.confirmed = True

            return True

        except Exception as e:
            return False

    def revert(self, mutation: Mutation, genome=None) -> bool:
        """
        Revert a mutation (it didn't help).

        Args:
            mutation: The mutation to revert
            genome: The genome to revert (uses self.genome if not provided)

        Returns:
            True if successful, False otherwise
        """
        genome = genome or self.genome
        if not genome:
            return False

        try:
            # Revert the change
            if mutation.target == "system_prompt":
                genome.system_prompt = mutation.current_value
            elif mutation.target == "capabilities":
                if mutation.proposed_value in genome.phenotype.capabilities:
                    genome.phenotype.capabilities.remove(mutation.proposed_value)
            elif mutation.target.startswith("interaction."):
                attr = mutation.target.split(".")[1]
                if hasattr(genome.interaction, attr):
                    setattr(genome.interaction, attr, mutation.current_value)
            elif mutation.target.startswith("model."):
                attr = mutation.target.split(".")[1]
                if attr in genome.phenotype.model:
                    genome.phenotype.model[attr] = mutation.current_value

            self.stats.total_mutations_reverted += 1
            self.active_mutations.pop(mutation.id, None)

            # Update mutation success rate
            total = (
                self.stats.total_mutations_applied + self.stats.total_mutations_reverted
            )
            if total > 0:
                self.stats.mutation_success_rate = (
                    self.stats.total_mutations_applied / total
                )

            return True

        except Exception as e:
            return False

    def evaluate_mutation(self, mutation: Mutation, outcome: Dict[str, Any]) -> float:
        """
        Evaluate how well a mutation worked.

        Args:
            mutation: The mutation to evaluate
            outcome: The outcome data

        Returns:
            Improvement score (-1 to 1)
        """
        # Check if outcome improved
        fitness_before = outcome.get("fitness_before", 0.5)
        fitness_after = outcome.get("fitness_after", 0.5)

        improvement = fitness_after - fitness_before

        # Adjust by confidence
        adjusted = improvement * mutation.confidence

        return adjusted

    def get_stats(self) -> EvolutionStats:
        """Get evolution statistics."""
        # Update mutation success rate
        total = self.stats.total_mutations_applied + self.stats.total_mutations_reverted
        if total > 0:
            self.stats.mutation_success_rate = (
                self.stats.total_mutations_applied / total
            )

        return self.stats

    def get_active_mutations(self) -> List[Mutation]:
        """Get all active mutations."""
        return list(self.active_mutations.values())

    def get_fitness_history(self, n: int = 100) -> List[FitnessRecord]:
        """Get recent fitness history."""
        return self.fitness_history[-n:]

    def save(self) -> None:
        """Save evolution state to disk."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "evolution_state.json"
        state = {
            "stats": self.stats.to_dict(),
            "active_mutations": {
                k: v.to_dict() for k, v in self.active_mutations.items()
            },
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> None:
        """Load evolution state from disk."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "evolution_state.json"
        if not state_file.exists():
            return

        with open(state_file, "r") as f:
            state = json.load(f)

        self.stats = EvolutionStats.from_dict(state.get("stats", {}))

        for mut_id, mut_data in state.get("active_mutations", {}).items():
            self.active_mutations[mut_id] = Mutation.from_dict(mut_data)

    def to_dict(self) -> Dict[str, Any]:
        """Export evolution engine state."""
        return {
            "stats": self.stats.to_dict(),
            "active_mutations": {
                k: v.to_dict() for k, v in self.active_mutations.items()
            },
            "mutation_rate": self.mutation_rate,
            "selection_pressure": self.selection_pressure,
        }

    def from_dict(self, data: Dict[str, Any]) -> "EvolutionEngine":
        """Import evolution engine state."""
        self.stats = EvolutionStats.from_dict(data.get("stats", {}))
        self.mutation_rate = data.get("mutation_rate", 0.05)
        self.selection_pressure = data.get("selection_pressure", "user_satisfaction")

        for mut_id, mut_data in data.get("active_mutations", {}).items():
            self.active_mutations[mut_id] = Mutation.from_dict(mut_data)

        return self


def create_default_evolution_engine(
    genome=None, storage_path: Optional[Path] = None
) -> EvolutionEngine:
    """Create a default evolution engine."""
    return EvolutionEngine(
        genome=genome,
        storage_path=storage_path,
        mutation_rate=0.05,
        selection_pressure="user_satisfaction",
    )
