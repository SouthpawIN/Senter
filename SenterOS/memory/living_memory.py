#!/usr/bin/env python3
"""
SenterOS Living Memory
======================

A 4-layer memory system that doesn't just store information—it lives it.

The living memory is organized into four layers:

1. SEMANTIC MEMORY (Facts, Concepts, Knowledge)
   - Structured knowledge extracted from interactions
   - Facts, concepts, principles
   - Organized by meaning, not chronology

2. EPISODIC MEMORY (Specific Interactions)
   - Detailed records of specific conversations
   - What was said, when, in what context
   - Preserves the "story" of interactions

3. PROCEDURAL MEMORY (How to Help)
   - Learned procedures for helping this specific human
   - What approaches work, what don't
   - Preferences, habits, patterns

4. AFFECTIVE MEMORY (Emotional Context)
   - Emotional tone of interactions
   - What emotions were present
   - How the human felt about responses

The key insight: Memory isn't passive storage—it's active understanding.
Each layer learns from interactions and evolves to better serve the human.

Usage:
======

memory = LivingMemory()

# After an interaction
memory.absorb(
    user_input="I'm working on a Python project",
    ai_response="That sounds great! What kind of project?",
    outcome="positive",
    emotion="enthusiastic"
)

# When responding, retrieve relevant memories
context = memory.retrieve(
    query="What does this human like to work on?",
    layers=["semantic", "procedural"]
)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import uuid
import math


@dataclass
class SemanticMemory:
    """Layer 1: Facts, concepts, and structured knowledge."""

    id: str
    content: str  # The fact or concept
    category: str  # fact, concept, principle, rule
    confidence: float = 1.0
    source: str = ""  # Where this came from
    learned_from: str = ""  # Which interaction
    abstractions: List[str] = field(default_factory=list)  # Higher-level concepts
    examples: List[str] = field(default_factory=list)  # Concrete examples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "learned_from": self.learned_from,
            "abstractions": self.abstractions,
            "examples": self.examples,
        }


@dataclass
class EpisodicMemory:
    """Layer 2: Specific interactions and conversations."""

    id: str
    timestamp: str
    summary: str  # Brief summary of the episode
    full_transcript: str = ""  # Full conversation if important
    topic: str = ""  # Main topic
    outcome: str = ""  # positive, neutral, negative
    duration_seconds: float = 0.0
    key_moments: List[str] = field(default_factory=list)  # Important moments
    follow_ups: List[str] = field(default_factory=list)  # Things to follow up on

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "full_transcript": self.full_transcript,
            "topic": self.topic,
            "outcome": self.outcome,
            "duration_seconds": self.duration_seconds,
            "key_moments": self.key_moments,
            "follow_ups": self.follow_ups,
        }


@dataclass
class ProceduralMemory:
    """Layer 3: How to help this specific human."""

    id: str
    procedure: str  # What works for this human
    context: str  # When this procedure applies
    success_rate: float = 1.0  # How often it works
    trial_count: int = 1
    last_tested: str = ""  # When last used
    variations: List[str] = field(default_factory=list)  # Different ways this works

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "procedure": self.procedure,
            "context": self.context,
            "success_rate": self.success_rate,
            "trial_count": self.trial_count,
            "last_tested": self.last_tested,
            "variations": self.variations,
        }


@dataclass
class AffectiveMemory:
    """Layer 4: Emotional context and tone."""

    id: str
    timestamp: str
    emotion: str  # Primary emotion
    intensity: float = 0.5  # 0-1, how intense
    trigger: str = ""  # What caused this emotion
    response_tone: str = ""  # How AI responded
    outcome: str = ""  # How the emotion evolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "trigger": self.trigger,
            "response_tone": self.response_tone,
            "outcome": self.outcome,
        }


class LivingMemory:
    """
    A 4-layer living memory system for SenterOS.

    Memory isn't passive storage—it's active understanding.
    Each layer learns from interactions and evolves to better serve.

    The four layers work together:
    - Semantic: What the human knows
    - Episodic: What we've talked about
    - Procedural: How to help this human
    - Affective: How the human feels

    When responding, memories are retrieved from relevant layers
    and composed into context for the model.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        # The four memory layers
        self.semantic: Dict[str, SemanticMemory] = {}
        self.episodic: Dict[str, EpisodicMemory] = {}
        self.procedural: Dict[str, ProceduralMemory] = {}
        self.affective: Dict[str, AffectiveMemory] = {}

        # Indexes for fast retrieval
        self.semantic_by_category: Dict[str, Set[str]] = defaultdict(set)
        self.episodic_by_topic: Dict[str, Set[str]] = defaultdict(set)
        self.procedural_by_context: Dict[str, Set[str]] = defaultdict(set)
        self.affective_by_emotion: Dict[str, Set[str]] = defaultdict(set)

        # Working memory (current conversation)
        self.current_conversation: List[Dict[str, str]] = []
        self.conversation_start: str = ""

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "total_facts": 0,
            "total_episodes": 0,
            "total_procedures": 0,
            "total_emotions": 0,
        }

        # Storage
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def absorb(
        self,
        user_input: str,
        ai_response: str,
        outcome: str = "neutral",
        emotion: str = "",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Absorb a new interaction into memory.

        This processes the interaction and stores it in the appropriate layers.

        Args:
            user_input: What the human said
            ai_response: What the AI responded
            outcome: How the interaction went (positive, neutral, negative)
            emotion: Primary emotion detected
            context: Additional context about the interaction

        Returns:
            The episode ID
        """
        context = context or {}
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Ensure conversation tracking
        if not self.conversation_start:
            self.conversation_start = timestamp

        # Add to current conversation
        self.current_conversation.append(
            {"role": "user", "content": user_input, "timestamp": timestamp}
        )
        self.current_conversation.append(
            {"role": "assistant", "content": ai_response, "timestamp": timestamp}
        )

        # Create episodic memory (the "story")
        episode = EpisodicMemory(
            id=episode_id,
            timestamp=timestamp,
            summary=self._summarize_interaction(user_input, ai_response),
            full_transcript=f"User: {user_input}\nAI: {ai_response}",
            topic=context.get("topic", self._extract_topic(user_input)),
            outcome=outcome,
            key_moments=[user_input] if len(user_input) > 50 else [],
        )

        self.episodic[episode_id] = episode
        self.episodic_by_topic[episode.topic].add(episode_id)
        self.stats["total_episodes"] += 1

        # Extract and store semantic memories (facts and concepts)
        self._extract_semantic_memories(user_input, episode_id)
        self._extract_semantic_memories(ai_response, episode_id)

        # Create procedural memory (what worked)
        if outcome == "positive":
            procedure = ProceduralMemory(
                id=str(uuid.uuid4()),
                procedure=ai_response[:200],
                context=user_input[:100],
                success_rate=1.0,
                trial_count=1,
                last_tested=timestamp,
            )
            self.procedural[procedure.id] = procedure
            self.procedural_by_context[self._extract_context_key(user_input)].add(
                procedure.id
            )
            self.stats["total_procedures"] += 1

        # Create affective memory (emotional context)
        if emotion:
            affective = AffectiveMemory(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                emotion=emotion,
                intensity=0.5,
                trigger=user_input[:100],
                response_tone=self._classify_response_tone(ai_response),
                outcome=outcome,
            )
            self.affective[affective.id] = affective
            self.affective_by_emotion[emotion].add(affective.id)
            self.stats["total_emotions"] += 1

        self.stats["total_interactions"] += 1

        return episode_id

    def _summarize_interaction(self, user_input: str, ai_response: str) -> str:
        """Create a brief summary of the interaction."""
        # Simple extraction - in production, use the model
        input_preview = user_input[:50] + "..." if len(user_input) > 50 else user_input
        return f"User asked about: {input_preview}"

    def _extract_topic(self, text: str) -> str:
        """Extract the main topic from text."""
        # Simple keyword extraction - in production, use NLP
        keywords = ["python", "code", "project", "help", "question", "think", "feel"]
        for keyword in keywords:
            if keyword in text.lower():
                return keyword
        return "general"

    def _extract_context_key(self, text: str) -> str:
        """Extract a context key for procedural memory."""
        words = text.lower().split()[:5]
        return "_".join(words)

    def _classify_response_tone(self, response: str) -> str:
        """Classify the tone of the AI response."""
        response_lower = response.lower()
        if "great" in response_lower or "wonderful" in response_lower:
            return "enthusiastic"
        elif "sorry" in response_lower or "unfortunately" in response_lower:
            return "empathetic"
        elif "however" in response_lower or "but" in response_lower:
            return "balanced"
        else:
            return "neutral"

    def _extract_semantic_memories(self, text: str, from_episode: str) -> List[str]:
        """Extract semantic memories from text."""
        # Simple extraction - in production, use NER and fact extraction
        extracted = []

        # Look for facts (simple patterns)
        if "is a" in text.lower() or "are" in text.lower():
            # This is a declarative statement - could be a fact
            sentences = text.replace("?", ".").split(".")
            for sentence in sentences:
                if len(sentence) > 20 and len(sentence) < 200:
                    fact_id = str(uuid.uuid4())
                    fact = SemanticMemory(
                        id=fact_id,
                        content=sentence.strip(),
                        category="fact",
                        learned_from=from_episode,
                    )
                    self.semantic[fact_id] = fact
                    self.semantic_by_category["fact"].add(fact_id)
                    self.stats["total_facts"] += 1
                    extracted.append(fact_id)

        return extracted

    def retrieve(
        self, query: str, layers: List[str] = None, top_k: int = 5, focus: str = ""
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant memories from specified layers.

        Args:
            query: What to search for
            layers: Which layers to search (all if not specified)
            top_k: Maximum results per layer
            focus: Filter by focus area

        Returns:
            Dictionary mapping layer names to retrieved memories
        """
        layers = layers or ["semantic", "episodic", "procedural", "affective"]
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = {layer: [] for layer in layers}

        if "semantic" in layers:
            for fact_id, fact in self.semantic.items():
                if query_lower in fact.content.lower():
                    fact.last_accessed = datetime.now().isoformat()
                    results["semantic"].append(
                        {
                            "type": "semantic",
                            "content": fact.content,
                            "category": fact.category,
                            "confidence": fact.confidence,
                            "source": fact.source,
                        }
                    )

            # Sort by confidence and limit
            results["semantic"].sort(key=lambda x: x["confidence"], reverse=True)
            results["semantic"] = results["semantic"][:top_k]

        if "episodic" in layers:
            for episode_id, episode in self.episodic.items():
                if (
                    query_lower in episode.summary.lower()
                    or query_lower in episode.full_transcript.lower()
                ):
                    results["episodic"].append(
                        {
                            "type": "episodic",
                            "summary": episode.summary,
                            "topic": episode.topic,
                            "outcome": episode.outcome,
                            "timestamp": episode.timestamp,
                        }
                    )

            # Sort by recency
            results["episodic"].sort(key=lambda x: x["timestamp"], reverse=True)
            results["episodic"] = results["episodic"][:top_k]

        if "procedural" in layers:
            for proc_id, proc in self.procedural.items():
                if (
                    query_lower in proc.context.lower()
                    or query_lower in proc.procedure.lower()
                ):
                    results["procedural"].append(
                        {
                            "type": "procedural",
                            "procedure": proc.procedure,
                            "context": proc.context,
                            "success_rate": proc.success_rate,
                            "trial_count": proc.trial_count,
                        }
                    )

            # Sort by success rate
            results["procedural"].sort(key=lambda x: x["success_rate"], reverse=True)
            results["procedural"] = results["procedural"][:top_k]

        if "affective" in layers:
            for aff_id, aff in self.affective.items():
                if query_lower in aff.trigger.lower() or aff.emotion in query_words:
                    results["affective"].append(
                        {
                            "type": "affective",
                            "emotion": aff.emotion,
                            "intensity": aff.intensity,
                            "trigger": aff.trigger,
                            "outcome": aff.outcome,
                        }
                    )

            results["affective"] = results["affective"][:top_k]

        return results

    def get_user_profile(self) -> Dict[str, Any]:
        """
        Build a user profile from all memory layers.

        This creates a summary of what the system knows about the human.
        """
        profile = {
            "interactions": self.stats["total_interactions"],
            "topics_discussed": list(self.episodic_by_topic.keys()),
            "key_facts": [],
            "working_styles": [],
            "emotional_patterns": [],
        }

        # Get top semantic memories
        for fact_id, fact in self.semantic.items():
            if fact.category == "fact" and fact.confidence > 0.8:
                profile["key_facts"].append(fact.content)

        profile["key_facts"] = profile["key_facts"][:10]

        # Get working style patterns
        for proc_id, proc in self.procedural.items():
            if proc.success_rate > 0.7:
                profile["working_styles"].append(
                    {"context": proc.context, "what_works": proc.procedure}
                )

        profile["working_styles"] = profile["working_styles"][:5]

        # Get emotional patterns
        for aff_id, aff in self.affective.items():
            if aff.intensity > 0.6:
                profile["emotional_patterns"].append(
                    {
                        "emotion": aff.emotion,
                        "trigger": aff.trigger,
                        "response": aff.response_tone,
                    }
                )

        return profile

    def get_conversation_context(self) -> str:
        """Get the current conversation as context."""
        if not self.current_conversation:
            return ""

        # Get last N messages
        recent = self.current_conversation[-10:]

        context_parts = []
        for msg in recent:
            role = "Human" if msg["role"] == "user" else "AI"
            content = (
                msg["content"][:200] + "..."
                if len(msg["content"]) > 200
                else msg["content"]
            )
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def clear_conversation(self) -> None:
        """Clear the current conversation but keep memories."""
        self.current_conversation = []
        self.conversation_start = ""

    def evolve(
        self, interaction_id: str, feedback: str, fitness_score: float
    ) -> List[Dict[str, Any]]:
        """
        Evolve memories based on feedback.

        This strengthens or weakens memories based on outcomes.
        """
        changes = []

        if interaction_id not in self.episodic:
            return changes

        episode = self.episodic[interaction_id]
        episode.outcome = feedback

        # Update procedural memories
        for proc_id, proc in self.procedural.items():
            if proc.context in episode.full_transcript:
                if feedback == "positive":
                    proc.success_rate = min(1.0, proc.success_rate + 0.1)
                    proc.trial_count += 1
                    changes.append(
                        {
                            "type": "strengthen_procedure",
                            "procedure_id": proc_id,
                            "new_success_rate": proc.success_rate,
                        }
                    )
                elif feedback == "negative":
                    proc.success_rate = max(0.0, proc.success_rate - 0.2)
                    changes.append(
                        {
                            "type": "weaken_procedure",
                            "procedure_id": proc_id,
                            "new_success_rate": proc.success_rate,
                        }
                    )

        return changes

    def save(self) -> None:
        """Save all memory layers to disk."""
        if not self.storage_path:
            return

        layers = {
            "semantic": self.semantic,
            "episodic": self.episodic,
            "procedural": self.procedural,
            "affective": self.affective,
        }

        for layer_name, layer_data in layers.items():
            layer_file = self.storage_path / f"{layer_name}.json"
            with open(layer_file, "w") as f:
                json.dump({k: v.to_dict() for k, v in layer_data.items()}, f, indent=2)

    def load(self) -> None:
        """Load all memory layers from disk."""
        if not self.storage_path:
            return

        layers = {
            "semantic": (self.semantic, SemanticMemory),
            "episodic": (self.episodic, EpisodicMemory),
            "procedural": (self.procedural, ProceduralMemory),
            "affective": (self.affective, AffectiveMemory),
        }

        for layer_name, (storage, cls) in layers.items():
            layer_file = self.storage_path / f"{layer_name}.json"
            if not layer_file.exists():
                continue

            with open(layer_file, "r") as f:
                data = json.load(f)
                for item_id, item_data in data.items():
                    item = cls.from_dict(item_data)
                    storage[item_id] = item

    def to_dict(self) -> Dict[str, Any]:
        """Export memory to dictionary."""
        return {
            "semantic": {k: v.to_dict() for k, v in self.semantic.items()},
            "episodic": {k: v.to_dict() for k, v in self.episodic.items()},
            "procedural": {k: v.to_dict() for k, v in self.procedural.items()},
            "affective": {k: v.to_dict() for k, v in self.affective.items()},
            "stats": self.stats,
        }

    def from_dict(self, data: Dict[str, Any]) -> "LivingMemory":
        """Import memory from dictionary."""
        for item_id, item_data in data.get("semantic", {}).items():
            self.semantic[item_id] = SemanticMemory.from_dict(item_data)

        for item_id, item_data in data.get("episodic", {}).items():
            self.episodic[item_id] = EpisodicMemory.from_dict(item_data)

        for item_id, item_data in data.get("procedural", {}).items():
            self.procedural[item_id] = ProceduralMemory.from_dict(item_data)

        for item_id, item_data in data.get("affective", {}).items():
            self.affective[item_id] = AffectiveMemory.from_dict(item_data)

        self.stats = data.get("stats", self.stats)
        return self


def create_default_memory(storage_path: Optional[Path] = None) -> LivingMemory:
    """Create a default living memory system."""
    return LivingMemory(storage_path)
