#!/usr/bin/env python3
"""
Senter v3.0 - Configuration-Driven AI Assistant
=================================================

The perfect AI assistant, built on the insight that configuration is DNA.

This module provides:
- Genome: The DNA of an AI system
- Knowledge Graph: Semantic knowledge storage
- Living Memory: 4-layer memory system
- Evolution Engine: Self-optimizing configuration
- Capability Registry: Dynamic capability discovery
- Configuration Engine: The unified brain

Usage:
======

from Senter import create_configuration_engine

engine = create_configuration_engine()
result = engine.interact("Hello!")

print(result["response"])
"""

__version__ = "3.0.0"
__author__ = "Senter Team"

from .genome.genome import (
    Genome,
    GenomeMeta,
    PhenotypeSpec,
    RepresentationSpec,
    EvolutionSpec,
    InteractionSpec,
    IntegrationSpec,
    create_default_genome,
)

from .knowledge.knowledge_graph import (
    KnowledgeGraph,
    Node,
    Edge,
    create_default_knowledge_graph,
)

from .memory.living_memory import (
    LivingMemory,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    AffectiveMemory,
    create_default_memory,
)

from .evolution.evolution_engine import (
    EvolutionEngine,
    Mutation,
    FitnessRecord,
    EvolutionStats,
    create_default_evolution_engine,
)

from .capabilities.capability_registry import (
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityResult,
    create_default_capability_registry,
)

from .engine.configuration_engine import (
    ConfigurationEngine,
    create_configuration_engine,
)

__all__ = [
    # Genome
    "Genome",
    "GenomeMeta",
    "PhenotypeSpec",
    "RepresentationSpec",
    "EvolutionSpec",
    "InteractionSpec",
    "IntegrationSpec",
    "create_default_genome",
    # Knowledge Graph
    "KnowledgeGraph",
    "Node",
    "Edge",
    "create_default_knowledge_graph",
    # Living Memory
    "LivingMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "AffectiveMemory",
    "create_default_memory",
    # Evolution Engine
    "EvolutionEngine",
    "Mutation",
    "FitnessRecord",
    "EvolutionStats",
    "create_default_evolution_engine",
    # Capability Registry
    "CapabilityRegistry",
    "CapabilitySpec",
    "CapabilityResult",
    "create_default_capability_registry",
    # Configuration Engine
    "ConfigurationEngine",
    "create_configuration_engine",
]
