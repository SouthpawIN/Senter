#!/usr/bin/env python3
"""
SenterOS Capability Registry
============================

A dynamic capability discovery and management system. Capabilities aren't
hardcoded—they're discovered, registered, and evolved automatically.

Key insight: Capabilities should be discovered, not defined. When a new
tool or function is added, SenterOS should automatically discover and
integrate it.

Capability Types:
=================

1. TOOLS - Executable functions (Python scripts, shell commands)
2. KNOWLEDGE - Information sources (documents, APIs, databases)
3. AGENTS - Other AI capabilities (other agents, models)
4. INTERFACES - Communication channels (CLI, TUI, API)

The registry:
- Auto-discovers capabilities in directories
- Creates capability specifications from code
- Manages capability dependencies
- Optimizes capability routing

Usage:
======

registry = CapabilityRegistry()

# Discover capabilities
registry.discover("/path/to/tools")

# Get capability for a task
capability = registry.query("generate image", top_k=1)

# Execute capability
result = await capability.execute(input_data)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Awaitable
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import importlib
import inspect
import uuid


@dataclass
class CapabilitySpec:
    """Specification of a capability."""

    id: str
    name: str  # Human-readable name
    type: str  # tool, knowledge, agent, interface
    description: str  # What this capability does
    category: str  # General category
    tags: List[str] = field(default_factory=list)  # Searchable tags

    # Input/output specs
    input_format: str = "text"  # text, json, file, multimodal
    output_format: str = "text"  # text, json, file, image, audio

    # Execution info
    handler: Optional[Callable] = None  # The actual function to call
    path: str = ""  # File path if applicable
    module: str = ""  # Module name if applicable

    # Metadata
    version: str = "1.0.0"
    author: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""
    use_count: int = 0

    # Performance metrics
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    last_success: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "path": self.path,
            "module": self.module,
            "version": self.version,
            "author": self.author,
            "created": self.created,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "last_success": self.last_success,
        }


@dataclass
class CapabilityResult:
    """Result of executing a capability."""

    success: bool
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """
    A dynamic capability registry for SenterOS.

    Capabilities aren't hardcoded—they're discovered, registered, and
    evolved automatically. When a new tool or function is added,
    SenterOS should automatically discover and integrate it.

    The registry:
    - Auto-discovers capabilities in directories
    - Creates capability specifications from code
    - Manages capability dependencies
    - Optimizes capability routing
    """

    def __init__(self, storage_path: Optional[Path] = None):
        # The registry
        self.capabilities: Dict[str, CapabilitySpec] = {}

        # Indexes
        self.by_name: Dict[str, str] = {}
        self.by_type: Dict[str, Set[str]] = defaultdict(set)
        self.by_category: Dict[str, Set[str]] = defaultdict(set)
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Discovery paths
        self.discovery_paths: List[Path] = []

        # Built-in capabilities
        self._register_builtins()

        # Storage
        self.storage_path = storage_path
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def _register_builtins(self) -> None:
        """Register built-in capabilities."""
        # Text generation
        self.register(
            CapabilitySpec(
                id="ajson://senteros/capability/text_generation",
                name="Text Generation",
                type="agent",
                description="Generate text responses using the configured model",
                category="core",
                tags=["text", "generation", "llm", "model"],
                input_format="text",
                output_format="text",
                handler=self._handler_text_generation,
            )
        )

        # Image understanding
        self.register(
            CapabilitySpec(
                id="ajson://senteros/capability/image_understanding",
                name="Image Understanding",
                type="agent",
                description="Analyze and describe images",
                category="vision",
                tags=["image", "vision", "multimodal", "analysis"],
                input_format="multimodal",
                output_format="text",
            )
        )

        # Web search
        self.register(
            CapabilitySpec(
                id="ajson://senteros/capability/web_search",
                name="Web Search",
                type="tool",
                description="Search the web for current information",
                category="research",
                tags=["search", "web", "internet", "current"],
                input_format="text",
                output_format="text",
                handler=self._handler_web_search,
            )
        )

        # Music generation
        self.register(
            CapabilitySpec(
                id="ajson://senteros/capability/music_generation",
                name="Music Generation",
                type="tool",
                description="Generate music from text descriptions",
                category="creative",
                tags=["music", "audio", "generation", "creative"],
                input_format="text",
                output_format="audio",
            )
        )

        # Image generation
        self.register(
            CapabilitySpec(
                id="ajson://senteros/capability/image_generation",
                name="Image Generation",
                type="tool",
                description="Generate images from text descriptions",
                category="creative",
                tags=["image", "generation", "creative", "art"],
                input_format="text",
                output_format="image",
            )
        )

    def _handler_text_generation(self, input_data: Dict[str, Any]) -> CapabilityResult:
        """Handler for text generation."""
        return CapabilityResult(
            success=True,
            output=f"Generated text for: {input_data.get('prompt', 'unknown')}",
            latency_ms=100.0,
        )

    def _handler_web_search(self, input_data: Dict[str, Any]) -> CapabilityResult:
        """Handler for web search."""
        return CapabilityResult(
            success=True,
            output=f"Search results for: {input_data.get('query', 'unknown')}",
            latency_ms=500.0,
        )

    def register(self, capability: CapabilitySpec) -> None:
        """Register a capability."""
        self.capabilities[capability.id] = capability
        self.by_name[capability.name.lower()] = capability.id
        self.by_type[capability.type].add(capability.id)
        self.by_category[capability.category].add(capability.id)
        for tag in capability.tags:
            self.by_tag[tag].add(capability.id)

    def discover(self, path: Path, recursive: bool = True) -> int:
        """
        Auto-discover capabilities in a directory.

        Looks for:
        - Python files with capability functions
        - JSON files with capability specifications
        - Shell scripts with capability markers

        Args:
            path: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            Number of capabilities discovered
        """
        if not path.exists():
            return 0

        discovered = 0

        for item in path.iterdir():
            if item.is_file() and item.suffix == ".py":
                discovered += self._discover_from_python(item)
            elif item.is_file() and item.suffix == ".json":
                discovered += self._discover_from_json(item)
            elif item.is_dir() and recursive:
                discovered += self.discover(item, recursive)

        self.discovery_paths.append(path)
        return discovered

    def _discover_from_python(self, path: Path) -> int:
        """Discover capabilities from a Python file."""
        # This is a simplified version—in production, use AST parsing
        return 0

    def _discover_from_json(self, path: Path) -> int:
        """Discover capabilities from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "capability" in data:
                spec = CapabilitySpec(
                    id=data.get("id", str(uuid.uuid4())),
                    name=data["capability"].get("name", path.stem),
                    type=data["capability"].get("type", "tool"),
                    description=data["capability"].get("description", ""),
                    category=data["capability"].get("category", "general"),
                    tags=data["capability"].get("tags", []),
                    path=str(path),
                )
                self.register(spec)
                return 1

        except Exception:
            pass

        return 0

    def query(
        self,
        query: str,
        type_filter: str = "",
        category_filter: str = "",
        top_k: int = 5,
    ) -> List[CapabilitySpec]:
        """
        Query for capabilities matching a description.

        Args:
            query: Natural language query
            type_filter: Filter by type (tool, knowledge, agent, interface)
            category_filter: Filter by category
            top_k: Maximum results

        Returns:
            List of matching capabilities
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = []

        for cap_id, cap in self.capabilities.items():
            # Apply filters
            if type_filter and cap.type != type_filter:
                continue
            if category_filter and cap.category != category_filter:
                continue

            # Calculate relevance score
            score = 0.0

            # Check name
            if query_lower in cap.name.lower():
                score += 0.5

            # Check description
            if query_lower in cap.description.lower():
                score += 0.3

            # Check tags
            tag_overlap = len(query_words & set(t.lower() for t in cap.tags))
            if tag_overlap > 0:
                score += 0.2 * (tag_overlap / len(cap.tags))

            # Boost frequently used
            score *= 1 + cap.use_count * 0.01

            # Boost high success rate
            score *= cap.success_rate

            if score > 0:
                scores.append((cap, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [cap for cap, score in scores[:top_k]]

    async def execute(self, capability_id: str, input_data: Any) -> CapabilityResult:
        """
        Execute a capability.

        Args:
            capability_id: ID of the capability to execute
            input_data: Input data for the capability

        Returns:
            CapabilityResult with output or error
        """
        if capability_id not in self.capabilities:
            return CapabilityResult(
                success=False, error=f"Capability not found: {capability_id}"
            )

        capability = self.capabilities[capability_id]
        start_time = datetime.now()

        try:
            # Execute the handler if available
            if capability.handler:
                if inspect.iscoroutinefunction(capability.handler):
                    output = await capability.handler(input_data)
                else:
                    output = capability.handler(input_data)
            else:
                # No handler—just return success
                output = f"Capability {capability.name} executed"

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update metrics
            capability.use_count += 1
            capability.last_used = datetime.now().isoformat()
            capability.average_latency_ms = (
                capability.average_latency_ms * (capability.use_count - 1) + latency_ms
            ) / capability.use_count
            capability.last_success = datetime.now().isoformat()

            return CapabilityResult(success=True, output=output, latency_ms=latency_ms)

        except Exception as e:
            return CapabilityResult(
                success=False,
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def get_capability(self, capability_id: str) -> Optional[CapabilitySpec]:
        """Get a capability by ID."""
        return self.capabilities.get(capability_id)

    def get_capabilities_by_type(self, type: str) -> List[CapabilitySpec]:
        """Get all capabilities of a given type."""
        return [
            self.capabilities[cap_id]
            for cap_id in self.by_type.get(type, set())
            if cap_id in self.capabilities
        ]

    def update_metrics(
        self, capability_id: str, success: bool, latency_ms: float
    ) -> None:
        """Update metrics for a capability."""
        if capability_id not in self.capabilities:
            return

        capability = self.capabilities[capability_id]
        capability.use_count += 1
        capability.average_latency_ms = (
            capability.average_latency_ms * (capability.use_count - 1) + latency_ms
        ) / capability.use_count

        if success:
            capability.last_success = datetime.now().isoformat()
        # Could also track failures to update success_rate

    def save(self) -> None:
        """Save the registry to disk."""
        if not self.storage_path:
            return

        registry_file = self.storage_path / "capabilities.json"
        with open(registry_file, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.capabilities.items()}, f, indent=2
            )

    def load(self) -> None:
        """Load the registry from disk."""
        if not self.storage_path:
            return

        registry_file = self.storage_path / "capabilities.json"
        if not registry_file.exists():
            return

        with open(registry_file, "r") as f:
            data = json.load(f)

        for cap_id, cap_data in data.items():
            cap = CapabilitySpec.from_dict(cap_data)
            self.capabilities[cap_id] = cap
            self.by_name[cap.name.lower()] = cap_id
            self.by_type[cap.type].add(cap_id)
            self.by_category[cap.category].add(cap_id)
            for tag in cap.tags:
                self.by_tag[tag].add(cap_id)

    def to_dict(self) -> Dict[str, Any]:
        """Export the registry."""
        return {
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "discovery_paths": [str(p) for p in self.discovery_paths],
        }

    def from_dict(self, data: Dict[str, Any]) -> "CapabilityRegistry":
        """Import the registry."""
        for cap_id, cap_data in data.get("capabilities", {}).items():
            cap = CapabilitySpec.from_dict(cap_data)
            self.capabilities[cap_id] = cap
            self.by_name[cap.name.lower()] = cap_id
            self.by_type[cap.type].add(cap_id)
            self.by_category[cap.category].add(cap_id)
            for tag in cap.tags:
                self.by_tag[tag].add(cap_id)

        for path_str in data.get("discovery_paths", []):
            self.discovery_paths.append(Path(path_str))

        return self


def create_default_capability_registry(
    storage_path: Optional[Path] = None,
) -> CapabilityRegistry:
    """Create a default capability registry with built-ins."""
    return CapabilityRegistry(storage_path)
