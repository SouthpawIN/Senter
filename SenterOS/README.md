# Senter v3.0 - Configuration-Driven AI Assistant

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ██████╗ ██████╗ ███████╗ █████╗  ██████╗██╗  ██╗                     ║
║    ██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔════╝██║  ██║                     ║
║    ██║     ██║   ██║█████╗  ███████║██║     ███████║                     ║
║    ██║     ██║   ██║██╔══╝  ██╔══██║██║     ██╔══██║                     ║
║    ╚██████╗╚██████╔╝███████╗██║  ██║╚██████╗██║  ██║                     ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝                     ║
║                                                                           ║
║              Configuration-Driven AI Assistant v3.0                      ║
║                                                                           ║
║              "Configuration is the DNA of an AI system"                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## The Fundamental Insight

**"Configuration is the DNA of an AI system. Code is just the cell membrane."**

Senter v3.0 represents a paradigm shift in AI system design. Instead of hard-coding behaviors, everything is defined declaratively in configuration files called **genomes**. The system:

- **Evolves** its own configuration based on outcomes
- **Discovers** capabilities automatically
- **Learns** from every interaction
- **Adapts** to the individual human it serves

## Comparison: Senter v2.0 vs Senter v3.0

| Aspect | v2.0 (Current) | v3.0 (Perfect) |
|--------|---------------|----------------|
| **Architecture** | 4 layers + 7 agents | 1 Configuration Engine |
| **Lines of Code** | ~10,000 | ~500 |
| **Extensibility** | Add Focus + SENTER.md | Add configuration snippet |
| **Evolution** | Manual updates | Automatic (Evolution Engine) |
| **Complexity** | High | Minimal |
| **Capabilities** | Pre-defined categories | Emergent from configuration |
| **Memory** | SENTER.md files | Living Memory (4-layer) |
| **Model Routing** | Router agent + embeddings | Semantic understanding |
| **Goals** | Goal_Detector agent | Implicit in context |
| **Tools** | Tool_Discovery agent | Dynamic Capability Registry |

## The Six Pillars

### 1. Genome (Configuration as DNA)
Every aspect of behavior is defined in a genome file:
```yaml
meta:
  version: 3.0.0
  type: omniagent
  
phenotype:
  model: {type: gguf, n_ctx: 8192}
  capabilities: [reasoning, conversation, analysis]
  boundaries: [no_harm, privacy_first]

evolution:
  mutation_rate: 0.05
  selection_pressure: user_satisfaction
```

### 2. Knowledge Graph (Semantic Memory)
A living graph that stores knowledge by meaning, not by index:
- Nodes = Concepts, facts, principles
- Edges = Relationships (related, prerequisite, contrast)
- Layers = Abstraction levels (raw → concept → principle → wisdom)

### 3. Living Memory (4-Layer System)
```
Layer 1: SEMANTIC - Facts, concepts, structured knowledge
Layer 2: EPISODIC - Specific conversations and interactions  
Layer 3: PROCEDURAL - How to help this specific human
Layer 4: AFFECTIVE - Emotional context and tone
```

### 4. Evolution Engine (Self-Optimization)
Implements natural selection at the configuration level:
- **Mutation**: Propose changes to prompts, capabilities, memory
- **Selection**: Choose beneficial changes based on fitness
- **Inheritance**: Pass successful changes forward

### 5. Capability Registry (Dynamic Discovery)
Auto-discovers and registers capabilities:
- **Tools**: Executable functions
- **Knowledge**: Information sources
- **Agents**: AI capabilities
- **Interfaces**: Communication channels

### 6. Configuration Engine (The Brain)
The unified engine that replaces 7 separate agents:
1. **UNDERSTAND** → Parse intent and extract meaning
2. **RETRIEVE** → Fetch relevant knowledge and capabilities
3. **COMPOSE** → Assemble prompt from components
4. **EXECUTE** → Generate response
5. **EVOLVE** → Optimize configuration based on outcomes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Senter.git
cd Senter/Senter

# Install dependencies
pip install -r requirements.txt

# Run tests
python Senter.py --test

# Start CLI mode
python Senter.py

# Start TUI mode (requires textual)
python Senter.py --tui

# Start interactive REPL
python Senter.py --interact
```

## Usage

### Basic CLI Usage

```python
from Senter import create_configuration_engine

# Create the engine
engine = create_configuration_engine(
    genome_path=Path("/path/to/genome"),
    user_id="my_user"
)

# Process a user interaction
result = engine.interact("How do I learn Python?")

print(result["response"])
# Output: "To learn Python, I'd recommend starting with..."
```

### Custom Genome

```python
from Senter import Genome, create_default_genome

# Create a custom genome
genome = create_default_genome()
genome.system_prompt = "You are a helpful coding assistant."
genome.interaction.style = "collaborative"
genome.evolution.mutation_rate = 0.1

# Save and use
genome.save(Path("/path/to/custom_genome.yaml"))
```

### Accessing Components

```python
engine = create_configuration_engine()

# Access knowledge graph
kg = engine.knowledge_graph
kg.add_concept("Python programming", focus="coding")

# Access memory
memory = engine.memory
memory.absorb(user_input="Hello", ai_response="Hi there!")

# Access evolution
evolution = engine.evolution
mutations = evolution.evolve(interaction_data, fitness_score=0.9)

# Get user profile
profile = engine.get_user_profile()
```

## Project Structure

```
Senter/
├── __init__.py                 # Package exports
├── Senter.py                 # Main entry point
├── genome/
│   ├── __init__.py
│   └── genome.py              # Genome (DNA) specification
├── knowledge/
│   ├── __init__.py
│   └── knowledge_graph.py     # Semantic knowledge storage
├── memory/
│   ├── __init__.py
│   └── living_memory.py       # 4-layer memory system
├── evolution/
│   ├── __init__.py
│   └── evolution_engine.py    # Self-optimizing engine
├── capabilities/
│   ├── __init__.py
│   └── capability_registry.py # Dynamic capability discovery
├── engine/
│   ├── __init__.py
│   └── configuration_engine.py # Unified brain
├── interface/
│   ├── __init__.py
│   └── tui.py                 # Terminal user interface
└── data/                      # User data (created at runtime)
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `Genome` | The DNA of an AI system |
| `KnowledgeGraph` | Semantic knowledge storage |
| `LivingMemory` | 4-layer memory system |
| `EvolutionEngine` | Self-optimizing configuration |
| `CapabilityRegistry` | Dynamic capability discovery |
| `ConfigurationEngine` | The unified brain |

## The Minimal Processing Pattern

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. UNDERSTAND                                          │
│     - Parse intent                                      │
│     - Extract goals                                     │
│     - Detect emotion                                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. RETRIEVE                                            │
│     - Query knowledge graph                             │
│     - Discover capabilities                             │
│     - Fetch memories                                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. COMPOSE                                             │
│     - Build system prompt                               │
│     - Add retrieved context                             │
│     - Apply interaction style                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  4. EXECUTE                                             │
│     - Generate response                                 │
│     - Stream output                                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  5. EVOLVE                                              │
│     - Absorb into memory                                │
│     - Update knowledge graph                            │
│     - Optimize configuration                            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
                   Response + Evolved Configuration
```

## Contributing

Senter is built on the insight that configuration is DNA. Contributions should follow this principle:

1. **Before adding code, ask**: "Can this be configuration instead?"
2. **All behavior should be declarative**, not imperative
3. **The system should evolve its own configuration**, not be manually tuned

## The Vision

Senter v3.0 is not just an AI assistant—it's a proof of concept for the future of AI systems:

> **"The best AI systems are defined by their relationships, not their code."**

This insight generalizes beyond personal assistants:
- **Multi-agent systems** defined by configuration
- **Organizational AI** that adapts to company culture
- **Research AI** that evolves its own methodologies

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built on the **Insight>Architecture Framework** from the Senter project, which teaches that:

> **"The best research contributions aren't 'we built X' - they're 'we understood Y.'"**

The insight behind Senter is: **Configuration is the DNA of an AI system.**

---

**Senter v3.0** - Configuration-Driven AI Assistant
*"Configuration is the DNA of an AI system"*
