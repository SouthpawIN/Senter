#!/usr/bin/env python3
"""
SenterOS Knowledge Graph
========================

A semantic knowledge graph that stores and retrieves knowledge based on
meaning, not exact matches. This is the "memory" of SenterOS.

The knowledge graph is NOT a traditional database—it's a living structure
that:
- Grows organically from interactions
- Forms semantic connections between concepts
- Supports multiple levels of abstraction
- Evolves based on usage patterns
- Can be queried by meaning, not just keywords

Architecture:
=============

Nodes (Concepts):
- Each node represents a concept, fact, or piece of knowledge
- Nodes have features: concept, confidence, frequency, emotion, timestamp
- Nodes can be concrete (facts) or abstract (principles)

Edges (Relationships):
- Each edge connects two nodes with a relationship type
- Relationship types: related, prerequisite, contrast, temporal, causal
- Edges have strength and direction

Layers (Abstraction):
- Layer 0: Raw observations (what was said)
- Layer 1: Extracted concepts (what it means)
- Layer 2: Principles (what generalizes)
- Layer 3: Wisdom (what transcends)

Query Modes:
- Semantic: Find nodes by meaning (uses embeddings)
- Structural: Find nodes by graph position
- Temporal: Find nodes by time
- Relational: Find nodes by connection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
import uuid
import math


@dataclass
class Node:
    """A node in the knowledge graph—represents a concept or piece of knowledge."""

    id: str
    concept: str  # The actual content of the node
    layer: int = 0  # Abstraction level: 0=raw, 1=concept, 2=principle, 3=wisdom

    # Node features
    confidence: float = 1.0  # How confident we are in this knowledge
    frequency: int = 1  # How often this concept appears
    importance: float = 0.5  # How important this concept is (0-1)

    # Emotional/affective data
    emotion: str = ""  # Associated emotion
    sentiment: float = 0.0  # Sentiment score (-1 to 1)

    # Temporal data
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0

    # Source tracking
    source: str = ""  # Where this knowledge came from
    context: str = ""  # Context in which it was learned

    # Tags and categorization
    tags: Set[str] = field(default_factory=set)
    focus: str = ""  # Which focus area this belongs to

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "concept": self.concept,
            "layer": self.layer,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "importance": self.importance,
            "emotion": self.emotion,
            "sentiment": self.sentiment,
            "created": self.created,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source": self.source,
            "context": self.context,
            "tags": list(self.tags),
            "focus": self.focus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            concept=data.get("concept", ""),
            layer=data.get("layer", 0),
            confidence=data.get("confidence", 1.0),
            frequency=data.get("frequency", 1),
            importance=data.get("importance", 0.5),
            emotion=data.get("emotion", ""),
            sentiment=data.get("sentiment", 0.0),
            created=data.get("created", ""),
            last_accessed=data.get("last_accessed", ""),
            access_count=data.get("access_count", 0),
            source=data.get("source", ""),
            context=data.get("context", ""),
            tags=set(data.get("tags", [])),
            focus=data.get("focus", ""),
        )


@dataclass
class Edge:
    """An edge in the knowledge graph—represents a relationship between nodes."""

    id: str
    source_id: str  # Node ID of the source
    target_id: str  # Node ID of the target
    relationship: str  # related, prerequisite, contrast, temporal, causal
    strength: float = 1.0  # How strong the relationship is (0-1)
    bidirectional: bool = False  # Does it go both ways?

    # Temporal data
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    # Metadata
    context: str = ""  # Context in which this relationship was learned
    examples: List[str] = field(default_factory=list)  # Example usages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "created": self.created,
            "context": self.context,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relationship=data.get("relationship", "related"),
            strength=data.get("strength", 1.0),
            bidirectional=data.get("bidirectional", False),
            created=data.get("created", ""),
            context=data.get("context", ""),
            examples=data.get("examples", []),
        )


class KnowledgeGraph:
    """
    A semantic knowledge graph for SenterOS.

    This is NOT a traditional database—it's a living structure that:
    - Grows organically from interactions
    - Forms semantic connections between concepts
    - Supports multiple levels of abstraction
    - Evolves based on usage patterns
    - Can be queried by meaning, not just keywords

    Usage:
    ======
    kg = KnowledgeGraph()

    # Add knowledge
    kg.add_concept("Python programming", focus="coding", layer=1)
    kg.add_concept("Machine learning", focus="research", layer=1)
    kg.connect("Python programming", "Machine learning", "prerequisite")

    # Query by meaning
    results = kg.query("programming for AI", top_k=5)

    # Get connected concepts
    connected = kg.get_connected("Machine learning")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}

        # Indexes for fast lookup
        self.by_concept: Dict[str, str] = {}  # concept -> node_id
        self.by_focus: Dict[str, Set[str]] = defaultdict(set)  # focus -> node_ids
        self.by_layer: Dict[int, Set[str]] = defaultdict(set)  # layer -> node_ids
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)  # tag -> node_ids

        # Graph structure
        self.adjacency: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.reverse_adjacency: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Storage
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "total_concepts": 0,
            "total_connections": 0,
            "abstraction_depth": 4,
        }

    def add_concept(
        self,
        concept: str,
        focus: str = "",
        layer: int = 1,
        confidence: float = 1.0,
        source: str = "",
        context: str = "",
        tags: List[str] = None,
        emotion: str = "",
        importance: float = 0.5,
    ) -> str:
        """
        Add a concept to the knowledge graph.

        Args:
            concept: The concept text
            focus: Which focus area this belongs to
            layer: Abstraction level (0=raw, 1=concept, 2=principle, 3=wisdom)
            confidence: How confident we are in this knowledge
            source: Where this knowledge came from
            context: Context in which it was learned
            tags: Tags for categorization
            emotion: Associated emotion
            importance: How important this concept is (0-1)

        Returns:
            The node ID of the added concept
        """
        # Check if concept already exists
        concept_key = concept.lower().strip()
        if concept_key in self.by_concept:
            node_id = self.by_concept[concept_key]
            node = self.nodes[node_id]
            node.frequency += 1
            node.last_accessed = datetime.now().isoformat()
            return node_id

        # Create new node
        node_id = str(uuid.uuid4())
        node = Node(
            id=node_id,
            concept=concept,
            layer=layer,
            confidence=confidence,
            importance=importance,
            source=source,
            context=context,
            tags=set(tags or []),
            focus=focus,
            emotion=emotion,
        )

        self.nodes[node_id] = node
        self.by_concept[concept_key] = node_id

        if focus:
            self.by_focus[focus].add(node_id)
        self.by_layer[layer].add(node_id)

        for tag in node.tags:
            self.by_tag[tag].add(node_id)

        self.stats["total_concepts"] += 1

        return node_id

    def connect(
        self,
        source_concept: str,
        target_concept: str,
        relationship: str = "related",
        strength: float = 1.0,
        bidirectional: bool = False,
        context: str = "",
    ) -> Optional[str]:
        """
        Connect two concepts in the knowledge graph.

        Args:
            source_concept: The source concept
            target_concept: The target concept
            relationship: Type of relationship (related, prerequisite, contrast, temporal, causal)
            strength: How strong the relationship is (0-1)
            bidirectional: Does it go both ways?
            context: Context in which this relationship was learned

        Returns:
            The edge ID if successful, None if concepts don't exist
        """
        source_key = source_concept.lower().strip()
        target_key = target_concept.lower().strip()

        if source_key not in self.by_concept or target_key not in self.by_concept:
            return None

        source_id = self.by_concept[source_key]
        target_id = self.by_concept[target_key]

        # Create edge
        edge_id = str(uuid.uuid4())
        edge = Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
            bidirectional=bidirectional,
            context=context,
        )

        self.edges[edge_id] = edge

        # Update adjacency
        self.adjacency[source_id][target_id] = max(
            self.adjacency[source_id][target_id], strength
        )
        if bidirectional:
            self.adjacency[target_id][source_id] = max(
                self.adjacency[target_id][source_id], strength
            )

        self.stats["total_connections"] += 1

        return edge_id

    def query(
        self,
        query: str,
        focus: str = "",
        layer: int = -1,
        top_k: int = 10,
        min_importance: float = 0.0,
    ) -> List[Tuple[Node, float]]:
        """
        Query the knowledge graph for relevant concepts.

        This is a simple keyword-based query. In a full implementation,
        this would use embeddings for semantic similarity.

        Args:
            query: The query string
            focus: Filter by focus area
            layer: Filter by abstraction layer (-1 = all)
            top_k: Maximum number of results
            min_importance: Minimum importance score

        Returns:
            List of (node, relevance_score) tuples
        """
        query_words = set(query.lower().split())
        results = []

        for node_id, node in self.nodes.items():
            # Filter by focus
            if focus and node.focus != focus:
                continue

            # Filter by layer
            if layer >= 0 and node.layer != layer:
                continue

            # Filter by importance
            if node.importance < min_importance:
                continue

            # Calculate relevance (simple keyword matching)
            concept_words = set(node.concept.lower().split())
            overlap = len(query_words & concept_words)
            if overlap == 0:
                continue

            # Score based on overlap and importance
            score = (overlap / max(len(query_words), 1)) * node.importance

            # Boost recent/important concepts
            if node.frequency > 1:
                score *= 1.1
            if node.importance > 0.7:
                score *= 1.2

            results.append((node, score))

        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_connected(
        self, concept: str, relationship: str = "", depth: int = 1
    ) -> List[Tuple[Node, str, float]]:
        """
        Get all concepts connected to a given concept.

        Args:
            concept: The concept to find connections for
            relationship: Filter by relationship type ("" = all)
            depth: How many hops to traverse

        Returns:
            List of (node, relationship, strength) tuples
        """
        concept_key = concept.lower().strip()
        if concept_key not in self.by_concept:
            return []

        start_id = self.by_concept[concept_key]
        connected = []

        # BFS traversal
        visited = {start_id}
        queue = [(start_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            for neighbor_id, strength in self.adjacency[current_id].items():
                if neighbor_id in visited:
                    continue

                # Find the edge
                edge = None
                for e in self.edges.values():
                    if e.source_id == current_id and e.target_id == neighbor_id:
                        edge = e
                        break

                if edge and relationship and edge.relationship != relationship:
                    continue

                if neighbor_id in self.nodes:
                    connected.append(
                        (
                            self.nodes[neighbor_id],
                            edge.relationship if edge else "related",
                            strength,
                        )
                    )
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, current_depth + 1))

        return connected

    def get_abstraction_chain(self, concept: str) -> Dict[int, List[Node]]:
        """
        Get the abstraction chain for a concept.

        Returns all related concepts at all abstraction levels,
        organized by layer.

        Args:
            concept: The concept to get the chain for

        Returns:
            Dictionary mapping layer number to list of nodes
        """
        result = {i: [] for i in range(self.stats["abstraction_depth"])}

        # Get the base concept
        concept_key = concept.lower().strip()
        if concept_key not in self.by_concept:
            return result

        base_id = self.by_concept[concept_key]
        base_node = self.nodes[base_id]

        # Add base concept to its layer
        result[base_node.layer].append(base_node)

        # Find related concepts at other layers
        for node in self.nodes.values():
            if node.id == base_id:
                continue

            # Check if this concept is related to the base
            # (same focus, shared tags, or connected in graph)
            if node.focus == base_node.focus:
                result[node.layer].append(node)
            elif node.tags & base_node.tags:
                result[node.layer].append(node)

        return result

    def extract_principles(self, focus: str = "") -> List[Node]:
        """
        Extract principles (layer 2) from the knowledge graph.

        Principles are generalizable concepts that transcend specific instances.

        Args:
            focus: Filter by focus area

        Returns:
            List of principle nodes
        """
        principles = []
        for node_id, node in self.nodes.items():
            if node.layer == 2:  # Principle layer
                if not focus or node.focus == focus:
                    principles.append(node)

        return sorted(principles, key=lambda x: x.importance, reverse=True)

    def evolve(
        self, interaction_result: Dict[str, Any], fitness_score: float
    ) -> List[Dict[str, Any]]:
        """
        Evolve the knowledge graph based on interaction outcomes.

        This is called after each interaction to:
        - Strengthen useful connections
        - Prune unused concepts
        - Update importance scores
        - Add new abstractions

        Args:
            interaction_result: Result of the interaction
            fitness_score: How successful the interaction was (0-1)

        Returns:
            List of changes made
        """
        changes = []

        # Strengthen connections that were used
        if fitness_score > 0.7:
            for edge_id, edge in self.edges.items():
                if edge.context and edge.context in str(interaction_result):
                    # Strengthen this connection
                    old_strength = edge.strength
                    edge.strength = min(1.0, edge.strength * 1.1)
                    if edge.strength != old_strength:
                        changes.append(
                            {
                                "type": "strengthen_connection",
                                "edge_id": edge_id,
                                "old_strength": old_strength,
                                "new_strength": edge.strength,
                            }
                        )

        # Update importance based on access
        for node_id, node in self.nodes.items():
            if node.access_count > 5 and node.importance < 0.8:
                old_importance = node.importance
                node.importance = min(1.0, node.importance + 0.05)
                if node.importance != old_importance:
                    changes.append(
                        {
                            "type": "update_importance",
                            "node_id": node_id,
                            "old_importance": old_importance,
                            "new_importance": node.importance,
                        }
                    )

        return changes

    def save(self) -> None:
        """Save the knowledge graph to disk."""
        if not self.storage_path:
            return

        nodes_file = self.storage_path / "nodes.json"
        edges_file = self.storage_path / "edges.json"

        with open(nodes_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.nodes.items()}, f, indent=2)

        with open(edges_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.edges.items()}, f, indent=2)

    def load(self) -> None:
        """Load the knowledge graph from disk."""
        if not self.storage_path:
            return

        nodes_file = self.storage_path / "nodes.json"
        edges_file = self.storage_path / "edges.json"

        if not nodes_file.exists():
            return

        with open(nodes_file, "r") as f:
            nodes_data = json.load(f)
            for node_id, node_data in nodes_data.items():
                node = Node.from_dict(node_data)
                self.nodes[node_id] = node
                self.by_concept[node.concept.lower().strip()] = node_id
                if node.focus:
                    self.by_focus[node.focus].add(node_id)
                self.by_layer[node.layer].add(node_id)
                for tag in node.tags:
                    self.by_tag[tag].add(node_id)

        if edges_file.exists():
            with open(edges_file, "r") as f:
                edges_data = json.load(f)
                for edge_id, edge_data in edges_data.items():
                    edge = Edge.from_dict(edge_data)
                    self.edges[edge_id] = edge
                    self.adjacency[edge.source_id][edge.target_id] = edge.strength
                    if edge.bidirectional:
                        self.adjacency[edge.target_id][edge.source_id] = edge.strength

    def to_dict(self) -> Dict[str, Any]:
        """Export knowledge graph to dictionary."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "stats": self.stats,
        }

    def from_dict(self, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Import knowledge graph from dictionary."""
        for node_id, node_data in data.get("nodes", {}).items():
            node = Node.from_dict(node_data)
            self.nodes[node_id] = node
            self.by_concept[node.concept.lower().strip()] = node_id
            if node.focus:
                self.by_focus[node.focus].add(node_id)
            self.by_layer[node.layer].add(node_id)
            for tag in node.tags:
                self.by_tag[tag].add(node_id)

        for edge_id, edge_data in data.get("edges", {}).items():
            edge = Edge.from_dict(edge_data)
            self.edges[edge_id] = edge
            self.adjacency[edge.source_id][edge.target_id] = edge.strength

        self.stats = data.get("stats", self.stats)
        return self


def create_default_knowledge_graph(
    storage_path: Optional[Path] = None,
) -> KnowledgeGraph:
    """Create a default knowledge graph with basic structure."""
    kg = KnowledgeGraph(storage_path)

    # Add basic SenterOS concepts
    kg.add_concept("SenterOS", focus="system", layer=3, importance=1.0)
    kg.add_concept("Configuration engine", focus="system", layer=2)
    kg.add_concept("Knowledge graph", focus="system", layer=2)
    kg.add_concept("Living memory", focus="system", layer=2)
    kg.add_concept("Evolution engine", focus="system", layer=2)
    kg.add_concept("Genome", focus="system", layer=2)
    kg.add_concept("Symbiotic partnership", focus="vision", layer=3, importance=0.9)

    # Connect them
    kg.connect("SenterOS", "Configuration engine", "has_component")
    kg.connect("SenterOS", "Knowledge graph", "has_component")
    kg.connect("SenterOS", "Living memory", "has_component")
    kg.connect("SenterOS", "Evolution engine", "has_component")
    kg.connect("Configuration engine", "Genome", "uses")
    kg.connect("Knowledge graph", "Living memory", "is")
    kg.connect("Evolution engine", "Knowledge graph", "optimizes")

    return kg
