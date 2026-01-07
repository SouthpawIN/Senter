#!/usr/bin/env python3
"""
SenterOS Configuration Engine
==============================

The heart of Senter v3.0. This single engine replaces the 7 separate agents
from v2.0 with a unified, configuration-driven approach.

The insight: Everything is configuration. Code is just the interpreter.

The engine implements the minimal sufficient processing pattern:
1. UNDERSTAND → Parse intent and extract meaning
2. RETRIEVE → Fetch relevant knowledge and capabilities
3. COMPOSE → Assemble prompt from components
4. EXECUTE → Generate response
5. EVOLVE → Optimize configuration based on outcomes

Usage:
======

engine = ConfigurationEngine(
    genome_path=Path("/path/to/genome"),
    storage_path=Path("/path/to/storage")
)

# Process a user interaction
response = engine.interact("How do I learn Python?")

# Get system status
status = engine.get_status()
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import time
import uuid


class ConfigurationEngine:
    """
    The unified configuration engine for Senter v3.0.

    This single engine replaces the 7 separate agents from v2.0:
    - Router → UNDERSTAND (semantic understanding)
    - Goal_Detector → UNDERSTAND (goal extraction)
    - Context_Gatherer → RETRIEVE (knowledge retrieval)
    - Tool_Discovery → RETRIEVE (capability discovery)
    - Planner → COMPOSE (prompt assembly)
    - Profiler → MEMORY (user profile)
    - Chat Agent → EXECUTE (model invocation)

    The engine follows the minimal sufficient processing pattern:
    1. UNDERSTAND - Parse intent and extract meaning
    2. RETRIEVE - Fetch relevant knowledge and capabilities
    3. COMPOSE - Assemble prompt from components
    4. EXECUTE - Generate response
    5. EVOLVE - Optimize configuration based on outcomes

    Key insight: Configuration is DNA, code is cell membrane.
    """

    def __init__(
        self,
        genome_path: Optional[Path] = None,
        storage_path: Optional[Path] = None,
        user_id: str = "default",
    ):
        self.user_id = user_id
        self.start_time = datetime.now()

        # Storage paths
        self.genome_path = genome_path or Path(
            "/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"
        )
        self.storage_path = storage_path or Path(
            f"/home/sovthpaw/ai-toolbox/Senter/SenterOS/data/{user_id}"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load or create genome
        self.genome = self._load_genome()

        # Initialize components
        self.knowledge_graph = self._init_knowledge_graph()
        self.memory = self._init_memory()
        self.evolution = self._init_evolution()
        self.capabilities = self._init_capabilities()

        # Model client (lazy loaded)
        self.model_client = None

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "total_tokens": 0,
            "average_latency_ms": 0,
            "last_interaction": None,
        }

        print(f"   🧠 Senter v3.0 Configuration Engine initialized")
        print(f"   📁 Genome: {self.genome_path}")
        print(f"   💾 Storage: {self.storage_path}")

    def _load_genome(self):
        """Load the genome from disk or create default."""
        genome_file = self.genome_path / "default.yaml"

        if genome_file.exists():
            try:
                from genome.genome import Genome

                return Genome.load(genome_file)
            except ImportError:
                pass

        # Create default genome
        from genome.genome import Genome, create_default_genome

        genome = create_default_genome()
        genome.meta.id = f"ajson://senteros/user/{self.user_id}"
        genome.meta.name = f"SenterOS for {self.user_id}"

        # Save
        genome_file.parent.mkdir(parents=True, exist_ok=True)
        genome.save(genome_file)

        return genome

    def _init_knowledge_graph(self):
        """Initialize the knowledge graph."""
        try:
            from knowledge.knowledge_graph import (
                KnowledgeGraph,
                create_default_knowledge_graph,
            )

            kg_path = self.storage_path / "knowledge"
            return create_default_knowledge_graph(kg_path)
        except ImportError:
            return None

    def _init_memory(self):
        """Initialize the living memory."""
        try:
            from memory.living_memory import LivingMemory, create_default_memory

            memory_path = self.storage_path / "memory"
            return create_default_memory(memory_path)
        except ImportError:
            return None

    def _init_evolution(self):
        """Initialize the evolution engine."""
        try:
            from evolution.evolution_engine import (
                EvolutionEngine,
                create_default_evolution_engine,
            )

            evolution_path = self.storage_path / "evolution"
            return create_default_evolution_engine(self.genome, evolution_path)
        except ImportError:
            return None

    def _init_capabilities(self):
        """Initialize the capability registry."""
        try:
            from capabilities.capability_registry import (
                CapabilityRegistry,
                create_default_capability_registry,
            )

            capabilities_path = self.storage_path / "capabilities"
            return create_default_capability_registry(capabilities_path)
        except ImportError:
            return None

    def interact(
        self, user_input: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user interaction through the full pipeline.

        This is the main entry point for all user interactions.

        Args:
            user_input: What the user said
            context: Additional context (images, files, etc.)

        Returns:
            Response dictionary with text and metadata
        """
        start_time = time.time()
        context = context or {}

        print(
            f"\n   👤 User: {user_input[:100]}{'...' if len(user_input) > 100 else ''}"
        )

        try:
            # Step 1: UNDERSTAND
            understanding = self._understand(user_input, context)
            print(f"   🧠 Understood intent: {understanding.get('intent', 'unknown')}")

            # Step 2: RETRIEVE
            retrieved = self._retrieve(understanding, context)
            print(
                f"   📚 Retrieved {len(retrieved.get('knowledge', []))} knowledge items, "
                f"{len(retrieved.get('capabilities', []))} capabilities"
            )

            # Step 3: COMPOSE
            composed = self._compose(user_input, understanding, retrieved)
            print(f"   ✍️  Composed prompt ({len(composed.get('prompt', ''))} chars)")

            # Step 4: EXECUTE
            executed = self._execute(composed)
            response_text = executed.get("response", "I couldn't generate a response.")
            print(f"   🤖 Generated response ({len(response_text)} chars)")

            # Step 5: EVOLVE
            evolution = self._evolve(user_input, response_text, understanding, executed)

            # Update stats
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(response_text, latency_ms)

            return {
                "response": response_text,
                "understanding": understanding,
                "retrieved": retrieved,
                "executed": executed,
                "evolved": evolution,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }

    def _understand(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Understand the user's intent and extract meaning.

        This replaces the Router and Goal_Detector agents from v2.0.
        """
        # Simple intent extraction (in production, use the model)
        input_lower = user_input.lower()

        # Detect intent
        intent = "general"
        if any(word in input_lower for word in ["how", "what", "why", "explain"]):
            intent = "question"
        elif any(
            word in input_lower for word in ["create", "make", "generate", "write"]
        ):
            intent = "creation"
        elif any(word in input_lower for word in ["help", "assist", "support"]):
            intent = "help"
        elif any(word in input_lower for word in ["code", "program", "debug", "fix"]):
            intent = "coding"
        elif any(word in input_lower for word in ["search", "find", "look up"]):
            intent = "search"

        # Detect goals (simplified)
        goals = []
        if (
            "want to" in input_lower
            or "need to" in input_lower
            or "goal" in input_lower
        ):
            goals.append({"text": user_input, "priority": "high"})

        # Detect emotion (simplified)
        emotion = ""
        if any(word in input_lower for word in ["frustrated", "annoyed", "angry"]):
            emotion = "frustrated"
        elif any(word in input_lower for word in ["excited", "happy", "great"]):
            emotion = "happy"
        elif any(word in input_lower for word in ["confused", "unsure", "don't know"]):
            emotion = "confused"

        return {
            "intent": intent,
            "goals": goals,
            "emotion": emotion,
            "input_length": len(user_input),
            "has_images": "images" in context,
            "has_files": "files" in context,
        }

    def _retrieve(
        self, understanding: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 2: Retrieve relevant knowledge and capabilities.

        This replaces the Context_Gatherer and Tool_Discovery agents from v2.0.
        """
        results = {"knowledge": [], "capabilities": [], "memories": [], "context": {}}

        # Query knowledge graph
        if self.knowledge_graph:
            query = understanding.get("intent", "general")
            results["knowledge"] = self.knowledge_graph.query(query, top_k=5)

        # Query capabilities
        if self.capabilities:
            intent = understanding.get("intent", "general")
            capability_type = ""
            if intent == "creation":
                capability_type = "tool"
            elif intent == "question":
                capability_type = "agent"

            results["capabilities"] = self.capabilities.query(
                intent, type_filter=capability_type, top_k=3
            )

        # Retrieve memories
        if self.memory:
            query = understanding.get("intent", "general")
            results["memories"] = self.memory.retrieve(
                query, layers=["semantic", "procedural"], top_k=3
            )

        # Get conversation context
        if self.memory:
            results["context"]["conversation"] = self.memory.get_conversation_context()

        return results

    def _compose(
        self, user_input: str, understanding: Dict[str, Any], retrieved: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 3: Compose the prompt from components.

        This replaces the Planner agent from v2.0.
        """
        # Start with system prompt
        prompt_parts = []

        # Add genome system prompt
        if self.genome.system_prompt:
            prompt_parts.append(self.genome.system_prompt)

        # Add interaction style
        style = self.genome.interaction.style
        if style == "collaborative":
            prompt_parts.append(
                "\n\nBe collaborative and supportive. Work with the user as a partner."
            )
        elif style == "directive":
            prompt_parts.append("\n\nBe clear and directive. Provide clear guidance.")
        elif style == "socratic":
            prompt_parts.append(
                "\n\nGuide through questions. Help the user discover answers."
            )

        # Add retrieved knowledge
        knowledge_items = retrieved.get("knowledge", [])
        if knowledge_items:
            prompt_parts.append("\n\n## Relevant Knowledge:")
            for item, score in knowledge_items[:3]:
                prompt_parts.append(f"- {item.concept}")

        # Add memories
        memories = retrieved.get("memories", {})
        if memories.get("procedural"):
            prompt_parts.append("\n\n## What works for this user:")
            for mem in memories["procedural"][:2]:
                prompt_parts.append(f"- {mem.get('procedure', '')}")

        # Add conversation context
        conversation = retrieved.get("context", {}).get("conversation", "")
        if conversation:
            prompt_parts.append(f"\n\n## Recent conversation:\n{conversation}")

        # Add user input
        prompt_parts.append(f"\n\n## Current User Request:\n{user_input}")

        # Add response guidance based on intent
        intent = understanding.get("intent", "general")
        if intent == "question":
            prompt_parts.append("\n\nProvide a clear, informative answer.")
        elif intent == "creation":
            prompt_parts.append("\n\nCreate or generate what the user requests.")
        elif intent == "coding":
            prompt_parts.append("\n\nProvide code with explanations.")

        prompt = "\n".join(prompt_parts)

        return {
            "prompt": prompt,
            "model_config": self.genome.phenotype.model,
            "max_tokens": self.genome.phenotype.model.get("max_tokens", 512),
            "temperature": self.genome.phenotype.model.get("temperature", 0.7),
        }

    def _execute(self, composed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Execute the prompt to generate a response.

        This replaces the Chat Agent from v2.0.
        """
        # Simple response generation (in production, call the model)
        prompt = composed.get("prompt", "")

        # Mock response for demonstration
        # In production, this would call the actual model
        response = self._generate_mock_response(prompt, composed)

        return {
            "response": response,
            "model_used": "mock",
            "tokens_used": len(prompt.split()) * 4 // 3,
        }

    def _generate_mock_response(self, prompt: str, composed: Dict[str, Any]) -> str:
        """Generate a mock response (for testing without a model)."""
        # Parse the intent from the prompt
        if "What works for this user" in prompt:
            return "Based on what I've learned about you, I'll provide detailed assistance tailored to your preferences."
        elif "creation" in prompt.lower() or "create" in prompt.lower():
            return "I'd be happy to help you create that! Let me provide you with what you need."
        elif "code" in prompt.lower() or "program" in prompt.lower():
            return "I can help you with your coding question. Let me provide a solution with explanations."
        elif "?" in prompt:
            return "That's a great question. Here's what I know about that topic..."
        else:
            return "I understand. How can I help you further with this?"

    def _evolve(
        self,
        user_input: str,
        response_text: str,
        understanding: Dict[str, Any],
        executed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Step 5: Evolve the configuration based on outcomes.

        This replaces the Profiler agent from v2.0 and adds self-optimization.
        """
        evolution_results = {
            "mutations_proposed": 0,
            "mutations_applied": 0,
            "knowledge_added": 0,
            "memory_updated": False,
        }

        # Absorb into memory
        if self.memory:
            self.memory.absorb(
                user_input=user_input,
                ai_response=response_text,
                outcome="positive",  # Would be based on feedback
                emotion=understanding.get("emotion", ""),
                context={
                    "intent": understanding.get("intent"),
                    "topic": understanding.get("intent"),
                },
            )
            evolution_results["memory_updated"] = True

        # Add knowledge to graph
        if self.knowledge_graph:
            # Extract key concepts
            key_concepts = user_input.split()[:5]
            for concept in key_concepts:
                if len(concept) > 3:
                    self.knowledge_graph.add_concept(
                        concept, focus=understanding.get("intent", "general"), layer=1
                    )
            evolution_results["knowledge_added"] = len(key_concepts)

        # Evolve configuration
        if self.evolution:
            mutations = self.evolution.evolve(
                interaction={
                    "input": user_input,
                    "response": response_text,
                    "intent": understanding.get("intent"),
                    "type": understanding.get("intent", "general"),
                    "duration": 0.5,
                },
                fitness_score=0.8,  # Would be based on actual outcome
            )
            evolution_results["mutations_proposed"] = len(mutations)

            # Apply mutations
            for mutation in mutations:
                if self.evolution.apply(mutation, self.genome):
                    evolution_results["mutations_applied"] += 1

        return evolution_results

    def _update_stats(self, response: str, latency_ms: float) -> None:
        """Update interaction statistics."""
        self.stats["total_interactions"] += 1
        self.stats["total_tokens"] += len(response.split())
        self.stats["last_interaction"] = datetime.now().isoformat()

        # Update average latency
        n = self.stats["total_interactions"]
        old_avg = self.stats["average_latency_ms"]
        self.stats["average_latency_ms"] = ((old_avg * (n - 1)) + latency_ms) / n

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the engine."""
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "user_id": self.user_id,
            "genome_version": self.genome.meta.version,
            "stats": self.stats,
            "components": {
                "knowledge_graph": self.knowledge_graph is not None,
                "memory": self.memory is not None,
                "evolution": self.evolution is not None,
                "capabilities": self.capabilities is not None,
            },
        }

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the user profile from memory."""
        if self.memory:
            return self.memory.get_user_profile()
        return {}

    def save_state(self) -> None:
        """Save all state to disk."""
        if self.knowledge_graph:
            self.knowledge_graph.save()
        if self.memory:
            self.memory.save()
        if self.evolution:
            self.evolution.save()
        if self.capabilities:
            self.capabilities.save()

        # Save genome
        self.genome.save(self.genome_path / "default.yaml")

    async def interact_async(
        self, user_input: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Async version of interact."""
        return self.interact(user_input, context)


def create_configuration_engine(
    genome_path: Optional[Path] = None,
    storage_path: Optional[Path] = None,
    user_id: str = "default",
) -> ConfigurationEngine:
    """
    Create and initialize a ConfigurationEngine.

    This is the main entry point for Senter v3.0.
    """
    return ConfigurationEngine(
        genome_path=genome_path, storage_path=storage_path, user_id=user_id
    )
