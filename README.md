# Senter - Universal AI Personal Assistant

![Senter v2.0](https://img.shields.io/badge/Senter-2.0.0-00ffaa?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A **model-agnostic AI personal assistant** built on an **async chain of omniagent instances**, each configured via SENTER.md files.

## 🌟 Overview

Senter v2.0 represents a **fundamental architectural shift** from script-heavy systems to a unified omniagent-based architecture where **everything is an omniagent with a unique SENTER.md configuration**.

### Core Philosophy

1. **Universal OmniAgent Pattern**: Every capability, feature, and focus is just an omniagent instance with a different SENTER.md config
2. **Async Chain Architecture**: Multiple omniagents work in parallel for performance
3. **Focus-First Organization**: Dynamic Focus system with unlimited, Focus-specific goals
4. **Self-Contained Configuration**: Focus purpose, system prompt, model, context - all in SENTER.md
5. **Extensibility**: Add new capability = Create Focus directory + SENTER.md (no code changes)

### Key Features

- 🎯 **Dynamic Focus System**: Automatic Focus creation based on user interests
- 🤖 **Model-Agnostic**: Bring your own model (GGUF, OpenAI API, vLLM)
- 📹 **Multimodal Processing**: Text, image, audio, video understanding
- 🧠 **Intelligent Routing**: Embedding-based filtering + LLM selection
- 🎨 **Creative Generation**: Image generation (Qwen) + music composition (ACE-Step)
- 🖥️ **Modern TUI**: Beautiful terminal interface using Textual framework
- 🔄 **Self-Learning**: Continuous evolution through SENTER.md context files
- 🚀 **Async Performance**: Parallel agent calls, non-blocking I/O
- ⚡ **Unlimited Goals**: Focus-specific goals tracked without caps
- 🔒 **Privacy-First**: All processing happens locally

---

## 🏗️ Architecture

### The OmniAgent Chain

**Principle**: Everything is an omniagent with SENTER.md

```
┌─────────────────────────────────────────────────────────┐
│              Senter OmniAgent Chain                     │
└─────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                                │
         ┌────▼────┐                   ┌────▼────┐
         │  Router  │                   │  Chat     │
         │  Agent   │                   │  Agent    │
         └────┬────┘                   └────┬────┘
              │                                │
         ┌────────────────┼────────────────┐
         │                          │         │
    ┌────▼─────┐┌────▼─────┐┌────▼─────┐┌────▼─────┐
│ Goal_      ││ Tool_    ││Context_   ││Plan-     ││Profil-    ││Chat      │
│Detector    ││Discovery ││Gatherer  ││ner       ││er Agent  ││Agent     │
└─────┬─────┘└─────┬─────┘└─────┬─────┘└─────┬─────┘└─────┬─────┘
      │         │         │         │         │         │
   │         │         │         │         │         │
   └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
        │
        │ Updates SENTER.md files
        │
        ▼
```

### Internal Focus Agents

| Agent | Purpose | Location | Output |
|-------|---------|----------|--------|
| **Router** | Route queries to best Focus | `Focuses/internal/Router/SENTER.md` | JSON: focus, reasoning |
| **Goal_Detector** | Extract goals from conversations | `Focuses/internal/Goal_Detector/SENTER.md` | JSON: goals list |
| **Tool_Discovery** | Discover tools in Functions/ | `Focuses/internal/Tool_Discovery/SENTER.md` | Creates Focuses for tools |
| **Context_Gatherer** | Update SENTER.md context | `Focuses/internal/Context_Gatherer/SENTER.md` | Direct file updates |
| **Planner** | Break down goals into tasks | `Focuses/internal/Planner/SENTER.md` | JSON: tasks list |
| **Profiler** | Analyze user patterns | `Focuses/internal/Profiler/SENTER.md` | JSON: preferences + patterns |
| **Chat** | Main conversational agent | `Focuses/internal/Chat/SENTER.md` | Natural responses |

### User Focus Agents

| Focus | Purpose | Features |
|-------|---------|----------|
| **coding** | Programming, debugging | Code examples, best practices |
| **research** | Information gathering | Fact-checking, source verification |
| **creative** | Art, music, writing | Style guidance, creative support |
| **user_personal** | Organization | Scheduling, goals, preferences |
| **general** | Catch-all | Versatile assistance, delegation |

---

## 🚀 Quick Start

### Installation

1. **Clone or navigate** to Senter:
```bash
cd /home/sovthpaw/ai-toolbox/Senter
```

2. **Run setup wizard**:
```bash
python3 scripts/setup_senter.py
```

This will:
- Download infrastructure models (Qwen2.5-Omni-3B + Nomic Embed)
- Configure your central model
- Verify all components work

### Running Senter

**CLI Interface**:
```bash
# Start Senter with async chain
python3 scripts/senter.py

# Available commands in CLI:
/list       - List all Focuses
/focus <name> - Set current Focus
/goals       - Show goals for current Focus
/discover    - Run tool discovery
/exit        - Exit
```

**TUI Interface** (work in progress):
```bash
# TUI with async chain (coming soon)
python3 scripts/senter_app.py
```

---

## 🎯 Focus System

### Focus Types

| Type | Has Wiki | Example | Purpose |
|------|-----------|---------|---------|
| **Conversational** | ✅ Yes | `Bitcoin/`, `AI/`, `Coding/` | Research topics with evolving knowledge |
| **Functional** | ❌ No | `WiFi_Lights/`, `Calendar/` | Single-purpose task execution |
| **Internal** | ❌ No | `Router/`, `Planner/` | Senter's own operation |

### Creating New Focuses

To add ANY new capability to Senter:

1. **Create Focus directory**:
```bash
mkdir -p Focuses/MyNewFocus
```

2. **Create SENTER.md**:
```yaml
---
model:
  type: gguf  # Inherits from user_profile.json
  
system_prompt: |
  You are a MyNewFocus Agent.
  Your job: [describe what it does]
  
focus:
  type: conversational  # or functional
  id: ajson://senter/focuses/mynewfocus
  name: MyNewFocus
  created: 2026-01-03T00:00:00Z
---

## User Preferences

## Patterns Observed

## Goals & Objectives
[Unlimited - no caps!]

## Evolution Notes
```

3. **Restart Senter** - it loads automatically!

That's it. No code changes needed.

---

## 📝 Query Processing Flow

When you ask Senter something:

1. **Router** analyzes your query
2. Selects best matching Focus (coding, research, creative, etc.)
3. **Context_Gatherer** gathers context for that Focus
4. **Goal_Detector** extracts any goals from your query
5. **Chat** agent processes with full context + goals

All steps run in parallel for maximum performance!

---

## 🧩 Adding Capabilities

### Example: Add a Image Generator Tool

```bash
# 1. Create Focus directory
mkdir -p Focuses/ImageGen

# 2. Create SENTER.md
cat > Focuses/ImageGen/SENTER.md <<'EOF'
---
model:
  type: gguf
  
system_prompt: |
  You are an Image Generation Agent.
  Your job: Generate images from text descriptions.
  
focus:
  type: functional
---

## User Preferences

## Patterns Observed

## Goals & Objectives
EOF

# 3. Restart Senter - it's now available!
```

### Example: Add a Calculator Tool

```bash
# 1. Create Focus directory
mkdir -p Focuses/Calculator

# 2. Create SENTER.md
cat > Focuses/Calculator/SENTER.md <<'EOF'
---
model:
  type: gguf
  
system_prompt: |
  You are a Calculator Agent.
  Your job: Perform mathematical calculations accurately.
  Available functions: basic math, scientific operations
  
focus:
  type: functional
  id: ajson://senter/focuses/calculator
  name: Calculator
---

## Available Functions
- add: Addition
- subtract: Subtraction
- multiply: Multiplication
- divide: Division
EOF

# 3. Restart Senter - it's now available!
```

---

## 📁 Directory Structure

```
Senter/
├── Focuses/              # Focus system (replaces Topics/)
│   ├── internal/          # Internal agents (7 Focuses)
│   │   ├── Router/
│   │   ├── Goal_Detector/
│   │   ├── Tool_Discovery/
│   │   ├── Context_Gatherer/
│   │   ├── Planner/
│   │   ├── Profiler/
│   │   └── Chat/
│   ├── coding/             # User Focuses
│   ├── research/
│   ├── creative/
│   ├── user_personal/
│   └── general/
│   ├── __init__.py
│   ├── senter_md_parser.py
│   └── focus_factory.py
│
├── Functions/             # Core AI pipelines
│   ├── omniagent.py           # Model-agnostic orchestrator
│   ├── omniagent_async.py    # Async wrapper (NEW)
│   ├── omniagent_chain.py    # Chain orchestrator (NEW)
│   ├── senter_md_parser.py   # SENTER.md parser
│   ├── embedding_utils.py    # Vector search
│   ├── compose_music.py       # Music generation
│   ├── qwen_image_gguf_generator.py  # Image generation
│   └── ...
│
├── scripts/                # Application code
│   ├── senter.py             # CLI interface (REFACTORED)
│   ├── senter_app.py         # TUI interface
│   ├── senter_widgets.py      # UI components
│   └── .obsolete/            # Old scripts (backup)
│
├── config/                  # Configuration
│   ├── senter_config.json    # Infrastructure models
│   └── user_profile.json      # User model + preferences
│
├── Agents/                  # Agent manifests (legacy, still works)
├── Models/                  # Downloaded models (gitignored)
└── outputs/                 # Generated content
```

---

## 🔧 Configuration

### senter_config.json
```json
{
  "infrastructure_models": {
    "multimodal_decoder": {
      "path": "/path/to/Qwen2.5-Omni-3B.gguf",
      "description": "Omni 3B for multimodal decoding"
    },
    "embedding_model": {
      "path": "/path/to/nomic-embed-text.gguf",
      "description": "Embeddings for intelligent search"
    }
  },
  "focus_creation": {
    "embed_filter_threshold": 4,
    "low_confidence_threshold": 0.5,
    "allow_dynamic_creation": true
  },
  "learning": {
    "senter_md_enabled": true,
    "wiki_enabled": true,
    "goal_detection": true
  }
}
```

### user_profile.json
```json
{
  "central_model": {
    "type": "gguf",
    "path": "/path/to/your-model.gguf",
    "is_vlm": false,
    "settings": {
      "max_tokens": 512,
      "temperature": 0.7
    }
  },
  "preferences": {
    "response_style": "balanced",
    "detail_level": "moderate",
    "creativity_level": 0.7
  }
}
```

---

## 📚 Architecture Documentation

For complete details on the OmniAgent Chain architecture, see:
- **ARCHITECTURE.md** - Full architecture documentation
- Focus system diagrams
- Agent interaction flows
- Adding new capabilities

---

## 🚀 Performance

### Async Benefits
- **Parallel Processing**: Multiple agents run simultaneously
- **Non-Blocking I/O**: File operations don't block responses
- **Responsive UI**: Even during heavy processing
- **Resource Efficiency**: Thread pool management

### Code Reduction
- **Old Architecture**: ~3500 lines across 20+ scripts
- **New Architecture**: ~1200 lines (omniagent_async.py + omniagent_chain.py + configs)
- **Reduction**: 65% less code
- **Maintainability**: Single pattern (omniagent + SENTER.md)

---

## 📋 Requirements

### Core Dependencies
- `llama-cpp-python>=0.3.0` - Model inference
- `textual>=0.50.0` - UI framework
- `Pillow>=10.0.0` - Image processing
- `pyyaml` - YAML parsing for SENTER.md

### Media Dependencies
- `yt-dlp>=2024.1.1` - Video downloads
- `soundfile>=0.13.0` - Audio I/O
- `ffmpeg` - Video/audio processing

### System Requirements
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: CUDA-compatible with 8GB+ VRAM
- **Storage**: 20GB for models + generated content

---

## 🤝 Development

### Adding New Internal Agents

1. Create `Focuses/internal/MyAgent/` directory
2. Create `SENTER.md` with proper structure
3. Add to `OmniAgentChain.initialize()` to load automatically

### Adding New User Focuses

1. Create `Focuses/MyFocus/` directory
2. Create `SENTER.md` with system prompt
3. Router will automatically detect it

### Testing

```bash
# Test async chain
python3 scripts/senter.py

# Test tool discovery
/discover

# List all Focuses
/list
```

---

## 🎓 Philosophy

1. **Everything is OmniAgent**: Every capability = omniagent + SENTER.md config
2. **Async by Default**: Parallel processing for performance
3. **Focus-First**: Organize by user interests, not fixed categories
4. **Unlimited Goals**: Track as many goals as needed, Focus-specific
5. **Self-Contained**: All config in SENTER.md files
6. **Extensible**: Add capability = Create Focus (no code changes)

---

## 🔄 Migration from v1.0

### What Changed
- **Removed**: 20+ specialized scripts (~2300 lines)
- **Added**: omniagent_async.py + omniagent_chain.py (~650 lines)
- **Refactored**: Everything uses async chain architecture
- **Replaced**: Pattern-matching with LLM-based goal detection
- **Simplified**: Tool discovery via omniagent instead of AST

### Migration Guide

1. Backup existing configuration
2. Run `setup_senter.py` to reconfigure
3. Old Focuses still work (SENTER.md structure preserved)
4. Goals now unlimited per Focus (no more cap of 3)
5. Internal agents now in SENTER.md format

---

## 📄 License

MIT License - see LICENSE file for details

---

**Built with ❤️ using llama-cpp-python, textual, and the omniagent pattern**

**Version**: 2.0.0 - Async Chain Architecture
**Status**: ✅ Core Implementation Complete, TUI Integration In Progress
