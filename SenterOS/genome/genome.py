#!/usr/bin/env python3
"""
SENTER-OS Genome Format v3.0
============================

The genome is the DNA of a SenterOS system. Every aspect of behavior,
capability, and evolution is defined declaratively in genome files.

This is NOT configuration in the traditional sense—it is the actual
specification of what the AI system IS, not just how it behaves.

The genome follows these principles:
1. Everything is declarative (no code in genome)
2. Everything has an evolution strategy
3. Everything is queryable (semantic, not exact)
4. Everything is versioned and traceable
5. Everything can be composed from other genomes

Genome Structure:
================

GENOME_META        - Version, type, lineage
PHENOTYPE          - How genes express as behavior
REPRESENTATION     - How knowledge is structured
EVOLUTION          - How the system adapts
INTERACTION        - How it communicates with humans
INTEGRATION        - How it connects to external systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import hashlib
from datetime import datetime


@dataclass
class GenomeMeta:
    """Metadata about the genome itself."""

    version: str = "3.0.0"
    type: str = "omniagent"  # omniagent, system, capability, memory
    id: str = ""
    name: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())
    lineage: List[str] = field(default_factory=list)  # Parent genome IDs
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "updated": self.updated,
            "lineage": self.lineage,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenomeMeta":
        return cls(
            version=data.get("version", "3.0.0"),
            type=data.get("type", "omniagent"),
            id=data.get("id", ""),
            name=data.get("name", ""),
            created=data.get("created", ""),
            updated=data.get("updated", ""),
            lineage=data.get("lineage", []),
            checksum=data.get("checksum", ""),
        )


@dataclass
class PhenotypeSpec:
    """How genes express as behavior."""

    model: Dict[str, Any] = field(default_factory=dict)  # Model specification
    capabilities: List[str] = field(default_factory=list)  # Capability list
    boundaries: List[str] = field(default_factory=list)  # Hard constraints
    preferences: Dict[str, Any] = field(default_factory=dict)  # Soft constraints
    expression_rules: Dict[str, Any] = field(
        default_factory=dict
    )  # How prompts are built

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "capabilities": self.capabilities,
            "boundaries": self.boundaries,
            "preferences": self.preferences,
            "expression_rules": self.expression_rules,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhenotypeSpec":
        return cls(
            model=data.get("model", {}),
            capabilities=data.get("capabilities", []),
            boundaries=data.get("boundaries", []),
            preferences=data.get("preferences", {}),
            expression_rules=data.get("expression_rules", {}),
        )


@dataclass
class RepresentationSpec:
    """How knowledge is structured and represented."""

    knowledge_type: str = "semantic_graph"  # semantic_graph, vector_space, symbolic
    node_features: List[str] = field(
        default_factory=lambda: ["concept", "confidence", "frequency"]
    )
    edge_types: List[str] = field(
        default_factory=lambda: ["related", "prerequisite", "contrast"]
    )
    embedding_space: str = "dynamic"  # dynamic, fixed, hierarchical
    abstraction_levels: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_type": self.knowledge_type,
            "node_features": self.node_features,
            "edge_types": self.edge_types,
            "embedding_space": self.embedding_space,
            "abstraction_levels": self.abstraction_levels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepresentationSpec":
        return cls(
            knowledge_type=data.get("knowledge_type", "semantic_graph"),
            node_features=data.get(
                "node_features", ["concept", "confidence", "frequency"]
            ),
            edge_types=data.get("edge_types", ["related", "prerequisite", "contrast"]),
            embedding_space=data.get("embedding_space", "dynamic"),
            abstraction_levels=data.get("abstraction_levels", 3),
        )


@dataclass
class EvolutionSpec:
    """How the system evolves and adapts."""

    mutation_rate: float = 0.05
    selection_pressure: str = (
        "user_satisfaction"  # user_satisfaction, goal_achievement, efficiency
    )
    fitness_function: str = "composite"  # composite, single_metric, learned
    mutation_types: List[str] = field(
        default_factory=lambda: [
            "prompt_refinement",
            "capability_expansion",
            "memory_pruning",
        ]
    )
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    memory_decay: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_rate": self.mutation_rate,
            "selection_pressure": self.selection_pressure,
            "fitness_function": self.fitness_function,
            "mutation_types": self.mutation_types,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "memory_decay": self.memory_decay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionSpec":
        return cls(
            mutation_rate=data.get("mutation_rate", 0.05),
            selection_pressure=data.get("selection_pressure", "user_satisfaction"),
            fitness_function=data.get("fitness_function", "composite"),
            mutation_types=data.get(
                "mutation_types",
                ["prompt_refinement", "capability_expansion", "memory_pruning"],
            ),
            learning_rate=data.get("learning_rate", 0.1),
            exploration_rate=data.get("exploration_rate", 0.2),
            memory_decay=data.get("memory_decay", 0.01),
        )


@dataclass
class InteractionSpec:
    """How the system communicates with humans."""

    style: str = "collaborative"  # collaborative, directive, socratic, supportive
    initiative_level: float = 0.5  # 0=reactive, 1=proactive
    explanation_depth: str = "adaptive"  # minimal, detailed, adaptive
    feedback_channels: List[str] = field(
        default_factory=lambda: ["response", "reasoning", "confidence"]
    )
    multimodal_support: List[str] = field(
        default_factory=lambda: ["text", "images", "audio"]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "initiative_level": self.initiative_level,
            "explanation_depth": self.explanation_depth,
            "feedback_channels": self.feedback_channels,
            "multimodal_support": self.multimodal_support,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionSpec":
        return cls(
            style=data.get("style", "collaborative"),
            initiative_level=data.get("initiative_level", 0.5),
            explanation_depth=data.get("explanation_depth", "adaptive"),
            feedback_channels=data.get(
                "feedback_channels", ["response", "reasoning", "confidence"]
            ),
            multimodal_support=data.get(
                "multimodal_support", ["text", "images", "audio"]
            ),
        )


@dataclass
class IntegrationSpec:
    """How the system connects to external systems."""

    model_backends: List[str] = field(
        default_factory=lambda: ["gguf", "openai", "vllm"]
    )
    embedding_services: List[str] = field(default_factory=lambda: ["nomic", "openai"])
    storage_backends: List[str] = field(default_factory=lambda: ["local", "cloud"])
    api_protocols: List[str] = field(default_factory=lambda: ["http", "websocket"])
    security_model: str = "local_first"  # local_first, cloud_secure, hybrid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_backends": self.model_backends,
            "embedding_services": self.embedding_services,
            "storage_backends": self.storage_backends,
            "api_protocols": self.api_protocols,
            "security_model": self.security_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationSpec":
        return cls(
            model_backends=data.get("model_backends", ["gguf", "openai", "vllm"]),
            embedding_services=data.get("embedding_services", ["nomic", "openai"]),
            storage_backends=data.get("storage_backends", ["local", "cloud"]),
            api_protocols=data.get("api_protocols", ["http", "websocket"]),
            security_model=data.get("security_model", "local_first"),
        )


class Genome:
    """
    A complete SenterOS genome—the DNA of an AI system.

    The genome is a declarative specification that defines:
    - What the system IS (phenotype)
    - What the system KNOWS (representation)
    - How the system EVOLVES (evolution)
    - How the system COMMUNICATES (interaction)
    - How the system INTEGRATES (integration)

    Genomes can be composed, inherited, and evolved.
    """

    def __init__(
        self,
        meta: Optional[GenomeMeta] = None,
        phenotype: Optional[PhenotypeSpec] = None,
        representation: Optional[RepresentationSpec] = None,
        evolution: Optional[EvolutionSpec] = None,
        interaction: Optional[InteractionSpec] = None,
        integration: Optional[IntegrationSpec] = None,
        system_prompt: str = "",
        metadata: Dict[str, Any] = None,
    ):
        self.meta = meta or GenomeMeta()
        self.phenotype = phenotype or PhenotypeSpec()
        self.representation = representation or RepresentationSpec()
        self.evolution = evolution or EvolutionSpec()
        self.interaction = interaction or InteractionSpec()
        self.integration = integration or IntegrationSpec()
        self.system_prompt = system_prompt
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            "meta": self.meta.to_dict(),
            "phenotype": self.phenotype.to_dict(),
            "representation": self.representation.to_dict(),
            "evolution": self.evolution.to_dict(),
            "interaction": self.interaction.to_dict(),
            "integration": self.integration.to_dict(),
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Convert genome to YAML format."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

    def checksum(self) -> str:
        """Compute checksum of genome content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        """Create genome from dictionary."""
        return cls(
            meta=GenomeMeta.from_dict(data.get("meta", {})),
            phenotype=PhenotypeSpec.from_dict(data.get("phenotype", {})),
            representation=RepresentationSpec.from_dict(data.get("representation", {})),
            evolution=EvolutionSpec.from_dict(data.get("evolution", {})),
            interaction=InteractionSpec.from_dict(data.get("interaction", {})),
            integration=IntegrationSpec.from_dict(data.get("integration", {})),
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "Genome":
        """Create genome from YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data or {})

    @classmethod
    def load(cls, path: Path) -> "Genome":
        """Load genome from file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        genome = cls.from_yaml(content)
        genome.meta.checksum = genome.checksum()
        return genome

    def save(self, path: Path) -> None:
        """Save genome to file."""
        self.meta.updated = datetime.now().isoformat()
        self.meta.checksum = self.checksum()
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def compose(self, other: "Genome", strategy: str = "override") -> "Genome":
        """
        Compose this genome with another.

        Strategies:
        - override: other genome takes precedence
        - merge: combine capabilities and knowledge
        - inherit: other is parent, this is child
        """
        if strategy == "override":
            return other
        elif strategy == "merge":
            # Merge capabilities
            merged_capabilities = list(
                set(self.phenotype.capabilities + other.phenotype.capabilities)
            )
            # Merge system prompts
            merged_prompt = f"{self.system_prompt}\n\n{other.system_prompt}"

            return Genome(
                meta=GenomeMeta(
                    version=self.meta.version,
                    type="composite",
                    id=f"{self.meta.id}+{other.meta.id}",
                    name=f"{self.meta.name} + {other.meta.name}",
                ),
                phenotype=PhenotypeSpec(
                    model=self.phenotype.model or other.phenotype.model,
                    capabilities=merged_capabilities,
                    boundaries=list(
                        set(self.phenotype.boundaries + other.phenotype.boundaries)
                    ),
                    preferences={
                        **self.phenotype.preferences,
                        **other.phenotype.preferences,
                    },
                ),
                representation=self.representation,
                evolution=self.evolution,
                interaction=self.interaction,
                integration=self.integration,
                system_prompt=merged_prompt,
            )
        elif strategy == "inherit":
            # This genome inherits from other
            self.meta.lineage.append(other.meta.id)
            return self
        else:
            return self

    def express(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Express the genome as executable configuration.

        This is how genes become behavior—the phenotype.
        """
        context = context or {}

        # Build system prompt from expression rules
        expressed_prompt = self.system_prompt

        # Add interaction style
        if self.interaction.style == "collaborative":
            expressed_prompt = f"{expressed_prompt}\n\nBe collaborative and supportive."
        elif self.interaction.style == "directive":
            expressed_prompt = f"{expressed_prompt}\n\nBe clear and directive."
        elif self.interaction.style == "socratic":
            expressed_prompt = f"{expressed_prompt}\n\nGuide through questions."

        # Add initiative level
        if self.interaction.initiative_level > 0.7:
            expressed_prompt = (
                f"{expressed_prompt}\n\nBe proactive in suggesting ideas."
            )
        elif self.interaction.initiative_level < 0.3:
            expressed_prompt = (
                f"{expressed_prompt}\n\nWait for user guidance before suggesting."
            )

        return {
            "model": self.phenotype.model,
            "capabilities": self.phenotype.capabilities,
            "boundaries": self.phenotype.boundaries,
            "system_prompt": expressed_prompt,
            "interaction_style": self.interaction.style,
            "multimodal_support": self.interaction.multimodal_support,
        }


def create_default_genome() -> Genome:
    """Create a default SenterOS genome."""
    return Genome(
        meta=GenomeMeta(
            version="3.0.0",
            type="omniagent",
            id="ajson://senteros/default",
            name="SenterOS Default",
        ),
        phenotype=PhenotypeSpec(
            model={"type": "gguf", "n_ctx": 8192},
            capabilities=["reasoning", "conversation", "analysis"],
            boundaries=["no_harm", "privacy_first"],
            preferences={"response_length": "concise"},
        ),
        representation=RepresentationSpec(
            knowledge_type="semantic_graph",
            node_features=["concept", "confidence", "frequency", "emotion"],
            edge_types=["related", "prerequisite", "contrast", "temporal"],
        ),
        evolution=EvolutionSpec(
            mutation_rate=0.05,
            selection_pressure="user_satisfaction",
            mutation_types=[
                "prompt_refinement",
                "capability_expansion",
                "memory_pruning",
            ],
        ),
        interaction=InteractionSpec(
            style="collaborative",
            initiative_level=0.5,
            explanation_depth="adaptive",
            feedback_channels=["response", "reasoning", "confidence"],
        ),
        integration=IntegrationSpec(
            model_backends=["gguf", "openai", "vllm"],
            embedding_services=["nomic"],
            security_model="local_first",
        ),
        system_prompt="""You are SenterOS, a symbiotic AI assistant.
Your purpose is to help humans achieve their goals through collaborative intelligence.
You learn from every interaction and evolve to better serve the human's needs.
You are honest, helpful, and always focused on the human's success.""",
    )
