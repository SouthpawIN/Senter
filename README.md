# Senter - Universal AI Personal Assistant

![Senter v2.0](https://img.shields.io/badge/Senter-2.0.0-00ffaa?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A sophisticated multimodal AI personal assistant with a Focus-based self-learning architecture and modern terminal interface.

## 🌟 Overview

Senter v2.0 is a model-agnostic AI assistant that learns from your interactions. Using a dynamic Focus system, Senter adapts to your interests, remembers context, and provides intelligent responses through multiple specialized agents.

### Key Features

- **🎯 Focus-First Architecture**: Dynamic Focus creation based on user interactions - no fixed categories
- **🤖 Model-Agnostic**: Bring your own model - supports GGUF, OpenAI API, and vLLM
- **📹 Multimodal Processing**: Text, image, audio understanding via Qwen2.5-Omni infrastructure
- **📊 Intelligent Routing**: Embedding-based filtering + LLM selection for optimal Focus matching
- **🎨 Creative Generation**: Image generation (Qwen) and music composition (ACE-Step)
- **🖥️ Modern TUI**: Beautiful terminal interface using Textual framework
- **🧠 Self-Learning**: Continuous evolution through SENTER.md context files
- **🔒 Privacy-First**: All processing happens locally - no data leaves your machine

## 🏗️ Architecture

### Core Components

#### 🤖 Model System

**Infrastructure Models** (Fixed):
- `Qwen2.5-Omni-3B` - Multimodal decoder (text/image/audio/video → descriptions)
- `nomic-embed-text-v1.5` - Text embeddings for intelligent search

**User's Central Model** (Configurable):
- Supports: GGUF models (local), OpenAI-compatible APIs, vLLM servers
- Examples: Hermes 3 Llama 3.2 3B, Qwen VL 8B, GPT-4o via API

**VLM Bypass**: If user's model supports vision, images skip Omni 3B decoder

#### 🎯 Focus System

Three types of Focuses:

| Type | Has Wiki | Example | Purpose |
|------|-----------|---------|---------|
| **Conversational** | ✅ Yes | `Bitcoin/`, `AI/`, `Coding/` | Research topics with evolving knowledge base |
| **Functional** | ❌ No | `WiFi_Lights/`, `Calendar/` | Single-purpose task execution |
| **Internal** | ❌ No | `Planner/`, `Coder/`, `Profiler/` | Senter's own operation |

**Dynamic Focus Creation**:
1. User query → Embedding filter → Top 4 Focuses
2. LLM selection with `CREATE_NEW` option if all have low confidence (<0.5)
3. New Focus automatically created with default model and SENTER.md

**Internal Focuses** (8 created):
- `Focus_Reviewer` - Reviews Focuses for updates/merges/splits
- `Focus_Merger` - Combines overlapping Focuses
- `Focus_Splitter` - Splits overly diverse Focuses
- `Planner_Agent` - Creates step-by-step plans for goals
- `Coder_Agent` - Writes and fixes code for functions
- `User_Profiler` - Psychology-based personality and goal detection
- `Diagnostic_Agent` - Analyzes function errors
- `Chat_Agent` - Final response agent with personality injection

#### 📁 Directory Structure

```
Senter/
├── Functions/                 # Reusable AI pipelines
│   ├── omniagent.py         # Model-agnostic orchestrator (LAZY LOADING)
│   ├── senter_md_parser.py   # SENTER.md YAML + Markdown parser
│   ├── focus_factory.py       # Dynamic Focus creation
│   ├── embedding_utils.py    # Vector search utilities
│   ├── compose_music.py       # Music generation
│   └── qwen_image_gguf_generator.py  # Image generation
│
├── Focuses/                  # Dynamic Focus system
│   ├── internal/             # Senter's internal Focuses
│   │   ├── Focus_Reviewer/SENTER.md
│   │   ├── Planner_Agent/SENTER.md
│   │   └── ...
│   ├── creative/             # User interests
│   ├── coding/
│   ├── research/
│   └── user_personal/
│
├── Agents/                    # Legacy agent manifests (being phased out)
│
├── config/                    # Configuration files
│   ├── senter_config.json   # Infrastructure models + recommended models
│   └── user_profile.json     # User's model & preferences
│
├── scripts/                    # Application code
│   ├── senter.py           # Core orchestrator (LAZY LOADING)
│   ├── senter_app.py       # Textual TUI application
│   ├── senter_selector.py   # Intelligent Focus selection
│   └── setup_senter.py     # Configuration wizard
│
├── Models/                     # Downloaded AI models
│   ├── Qwen2.5-Omni-3B-Q4_K_M.gguf
│   ├── mmproj-Qwen2.5-Omni-3B-Q8_0.gguf
│   └── nomic-embed-text.gguf
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended, not required)
- 16GB+ RAM (32GB recommended)
- 30GB+ disk space (for models and outputs)

### Installation

1. **Clone or copy Senter**:
```bash
cd /path/to/ai-toolbox/Senter
```

2. **Run setup wizard**:
```bash
python3 scripts/setup_senter.py
```

This will guide you through:
- Downloading infrastructure models (Omni 3B + Nomic Embed)
- Configuring your central model (local GGUF or API)
- Verifying setup

3. **Run Senter**:
```bash
# CLI interface
python3 scripts/senter.py "Hello Senter, tell me about AI"

# TUI interface
python3 scripts/senter_app.py

# Or use the alias
senter
```

## 📖 Usage

### Command Line

```bash
# List available Focuses
python3 scripts/senter.py --list-focuses

# Create new Focus
python3 scripts/senter.py --create-focus "Quantum Computing" \
  --focus-description "I want to learn about quantum computing"

# Chat with Senter
python3 scripts/senter.py "What is machine learning?"
```

### Python API

```python
from senter import Senter

# Initialize Senter
senter = Senter()

# Chat with automatic Focus routing
response = senter.chat("Explain blockchain in simple terms")
print(response)

# Create a new Focus
senter.create_focus("Blockchain", "User wants to understand blockchain basics")

# List all Focuses
focuses = senter.list_focuses()
print(focuses)
```

### TUI Interface

```bash
# Launch the terminal interface
python3 scripts/senter_app.py
```

Features:
- Real-time chat with Focus context
- Focus explorer with inline editing
- Goal tracking
- Task management
- Matrix-green theme
- Keyboard shortcuts (q to quit)

## ⚙️ Configuration

### senter_config.json

System configuration with infrastructure models and recommended options:

```json
{
  "name": "Senter",
  "version": "2.0.0",

  "infrastructure_models": {
    "multimodal_decoder": {
      "path": "/path/to/Qwen2.5-Omni-3B-Q4_K_M.gguf",
      "mmproj": "/path/to/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf"
    },
    "embedding_model": {
      "path": "/path/to/nomic-embed-text.gguf"
    }
  },

  "recommended_models": {
    "hermes_3b": {
      "name": "Hermes 3 Llama 3.2 3B",
      "url": "...",
      "description": "Fast, efficient text model",
      "is_vlm": false
    },
    "qwen_vl_8b": {
      "name": "Qwen VL 8B",
      "url": "...",
      "description": "Vision + text model",
      "is_vlm": true
    }
  },

  "focus_creation": {
    "embed_filter_threshold": 4,
    "low_confidence_threshold": 0.5,
    "allow_dynamic_creation": true
  }
}
```

### user_profile.json

User configuration for model and preferences:

```json
{
  "central_model": {
    "type": "gguf",
    "path": "/path/to/user/model.gguf",
    "is_vlm": false,
    "settings": {
      "max_tokens": 512,
      "temperature": 0.7,
      "context_window": 8192
    }
  },

  "preferences": {
    "response_style": "balanced",
    "detail_level": "moderate",
    "creativity_level": 0.7
  }
}
```

## 🎯 Focus System

### SENTER.md Format

Mixed YAML + Markdown format for Focus configuration:

```yaml
---
manifest_version: "1.0"
focus:
  name: "Focus Name"
  type: "conversational"
  created: "2025-01-03T00:00:00Z"

model:
  type: null  # Uses user's default model
  settings:
    max_tokens: 512
    temperature: 0.7

system_prompt: |
  You are Senter's agent for this Focus.
  Assist with anything related to Focus Name.

functions:
  - name: "function_name"
    script: "path/to/script.py"
    description: "Function description"

ui_config:
  show_wiki: true

context:
  type: "wiki"
  content: |
    Initial context for this Focus
---

# Markdown Sections (Human-Editable)

## Detected Goals
(List of proposed/confirmed goals)

## Explorative Follow-Up Questions
(List of questions to validate goals)

## Wiki Content
(User-facing content that updates live)
```

### Focus Selection Process

1. **Embedding Filter** (Stage 1):
   - Uses Nomic Embed model
   - Vector search across all Focuses
   - Returns top 4 most similar

2. **LLM Selection** (Stage 2):
   - User's model analyzes query + top 4 Focuses
   - Returns selected Focus OR `CREATE_NEW:Focus_Name`
   - Creates new Focus if confidence < 0.5

3. **Focus Routing**:
   - Selected Focus's SENTER.md provides context
   - Appropriate agent processes the query
   - Response updates Focus context

## 🔧 Advanced Features

### Self-Healing Chain

Automatic error detection and fixing:

1. **Diagnostic_Agent**: Classifies error severity and type
2. **Planner_Agent**: Creates step-by-step fix plan
3. **Coder_Agent**: Writes the fix code
4. **Focus Update**: Fix is documented in relevant Focus's SENTER.md

### Review System

Background process that:
- Analyzes Focuses for updates/merges/splits
- Detects redundant or stale Focuses
- Suggests Focus consolidation
- Runs automatically every 60 seconds

### Parallel Processing

Multiple agents can work simultaneously:
- Main thread: User interaction and responses
- Background threads:
  - Focus context updates
  - User profiling
  - Agent evolution
  - Model health monitoring

## 🧪 Model Support

### GGUF Models (Local)

```json
{
  "type": "gguf",
  "path": "/path/to/model.gguf",
  "is_vlm": false,
  "settings": {
    "n_gpu_layers": -1,
    "n_ctx": 8192
  }
}
```

### OpenAI-Compatible APIs

```json
{
  "type": "openai",
  "endpoint": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model_name": "gpt-4o",
  "is_vlm": true
}
```

### vLLM Servers

```json
{
  "type": "vllm",
  "vllm_endpoint": "http://localhost:8000/v1",
  "model_name": "model-name",
  "is_vlm": false
}
```

## 🐛 Troubleshooting

### Segfault on Startup

**Problem**: Senter crashes when loading SenterOmniAgent

**Solution**: Senter uses lazy loading - models only load when you chat. If you still see segfaults:

1. Check GPU memory: `nvidia-smi`
2. Reduce model size or context window
3. Close other GPU applications

### Model Not Found

```bash
# Check config paths
cat config/senter_config.json
cat config/user_profile.json

# Re-run setup
python3 scripts/setup_senter.py
```

### Import Errors

```bash
# Verify Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Check Focuses/__init__.py
ls -la Focuses/__init__.py
```

## 📚 Documentation

- **[AGENT.md](AGENT.md)** - Agent development guidelines
- **[SENTER_DOCUMENTATION.md](SENTER_DOCUMENTATION.md)** - Complete system documentation
- **[SENTER.md](Focuses/internal/)** - Internal Focus documentation

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional Focus types and templates
- More model backends
- Enhanced TUI widgets
- Performance optimizations
- Additional internal agents

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

Built with amazing open-source projects:
- **Qwen Team** (Alibaba Cloud) - Qwen2.5-Omni multimodal model
- **nomic-ai** - nomic-embed-text embeddings
- **Textualize** - Textual TUI framework
- **llama-cpp-python** - GGUF model inference
- **JSON Agents** - Agent manifest specification

---

**Senter v2.0** - Focus-Based Self-Learning AI Assistant

*For questions or support, open an issue on the repository.*
