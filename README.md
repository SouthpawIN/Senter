# Senter - Universal AI Personal Assistant

![Senter v2.0](https://img.shields.io/badge/Senter-2.0.0-00ffaa?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**An open-source life assistant building a symbiotic future where AI and humans collaborate to unlock their full potential.**

---

## 🌟 The Vision: Symbiotic AI-Human Partnership

Senter is more than an AI assistant - it's a **manifesto for how AI and humans can work together**.

**What Senter ultimately is designed to do:**

Senter harnesses the power of Large Language Models to:
1. **Process natural language into ordered data and intelligent actions** - Transform your messy, unstructured thoughts into clear, actionable insights
2. **Pick up new functionality that the user can call upon automatically when Senter encounters a script, function, or command line tool** - Seamlessly integrate any tool you write
3. **Update its knowledge about the user's interests** - Learn from every conversation, building rich context around what matters to you
4. **Answer the user's questions** - Provide helpful, context-aware responses using all available information

### The Four Pillars of Symbiotic Partnership

#### 1. **Knowledge Evolution** (Focuses)
Every interest you have (Bitcoin, AI, coding, creative writing, research, etc.) becomes a **Focus** - a dynamic, living knowledge base:
- Senter learns what you care about through conversations
- Each Focus has its own evolving knowledge stored in SENTER.md
- Focuses can be conversational (with wiki.md knowledge) or functional (single-purpose tools)
- **No predefined templates** - Focuses grow organically based on your actual interests and goals

#### 2. **Tool Auto-Discovery** (Functions/)
Write any Python script, shell command, or tool, and Senter will:
- Automatically discover it in your Functions/ directory
- Create a Focus for it with a unique SENTER.md configuration
- Integrate it seamlessly into conversations
- Route relevant queries to that tool's Focus automatically
- **No manual configuration** - just code, and Senter handles the rest

#### 3. **Goal & Action Tracking** (Background Processes)
Senter's internal agents continuously work in the background:
- **Goal_Detector**: Extracts goals from your conversations, unlimited and Focus-specific
- **Planner**: Breaks down complex goals into actionable steps
- **Profiler**: Analyzes your patterns, preferences, and interaction style
- **Context_Gatherer**: Updates SENTER.md files with conversation summaries
- **Tool_Discovery**: Scans for new tools and creates Focuses for them

#### 4. **Intelligent Query Processing** (OmniAgent Chain)
When you ask Senter something, an async chain of agents processes your request:
- **Router**: Selects the best Focus for your query
- **Context_Gatherer**: Pulls relevant context from that Focus
- **Goal_Detector**: Extracts any goals relevant to your query
- **Tool_Discovery**: Finds tools that can help
- **Chat Agent**: Provides a response with full context awareness

### The Human's Role

Senter is **your partner in learning and creating**, not your replacement:
- You provide the creativity, goals, direction, and tools
- Senter provides the knowledge, capabilities, organization, and synthesis
- Together, you both become more effective than either alone

### What Makes Senter Unique?

1. **Everything is OmniAgent + SENTER.md**: No complex scripts, no hidden logic - every capability is just an omniagent with a configuration file
2. **Async Chain Architecture**: Multiple agents work in parallel for maximum performance
3. **Model-Agnostic**: Bring your own model (GGUF, OpenAI API, vLLM) - Senter adapts to what you have
4. **Privacy-First**: All processing happens locally, your data never leaves your machine
5. **Truly Extensible**: Add any capability by creating a Focus directory with SENTER.md - no code changes needed
6. **Self-Organizing**: Senter learns what matters to you, creating Focuses and tracking goals automatically
7. **Markdown-First Configuration**: All configuration is in human-readable SENTER.md files, easy to understand and modify

### Real-World Examples

**Example 1: Bitcoin Trading Focus**
```
You: "I want to learn about Bitcoin trading strategies"

Senter [Goal_Detector]: Goal detected: "Learn Bitcoin trading strategies"
Senter [Planner]: Breaking down into steps:
  1. Research different trading approaches
  2. Understand risk management
  3. Learn about technical analysis
  4. Practice with paper trading first

You: "What's the current BTC price?"

Senter [Router]: Routes to Bitcoin Focus
Senter [Bitcoin Focus]: [Performs web search] Current BTC: $67,432.50

Senter [Context_Gatherer]: Updates Bitcoin Focus SENTER.md with:
  - Current interests: Trading strategies, technical analysis
  - Web sources checked
```

**Example 2: Automatically Discovered Tool**
```
# User writes a Python script
cat > Functions/encrypt_file.py <<'EOF'
import os
from cryptography.fernet import Fernet

def encrypt_file(filepath, key):
    with open(filepath, 'rb') as f:
        data = f.read()
    fernet = Fernet(key)
        encrypted = fernet.encrypt(data)
    
    with open(filepath + '.enc', 'wb') as f:
        f.write(encrypted)
    
    return "File encrypted successfully"
EOF

# Senter automatically discovers this
Senter [Tool_Discovery]: Found new tool: encrypt_file
Senter [Tool_Discovery]: Creates Focuses/encrypt_file/SENTER.md:
  system_prompt: |
    You are an Encryption Agent.
    Your job: Encrypt files securely.
    Available functions: encrypt_file

# Now user can use it
You: "Encrypt my notes.txt with key 'secret'"
Senter [Router]: Routes to encrypt_file Focus
Senter [encrypt_file Focus]: Notes.txt.enc created
```

**Example 3: Goal Progression**
```
You: "I'm working on a music album"

Senter [Goal_Detector]: Goal detected: "Complete music album"
Senter [Planner]: Breaks down into steps:
  1. Write 5-7 songs
  2. Record each song
  3. Mix and master
  4. Design album art
  5. Release on streaming platforms

[Week 1]
You: "I've written 3 songs"
Senter [Planner]: Updates goals in creative Focus SENTER.md
  - ✅ Write 5-7 songs (progress: 3/7)
  - Next: Record songs

[Week 2]
You: "I need help mixing the songs"
Senter [Context_Gatherer]: Adds note to creative Focus: "User needs help with audio mixing"
Senter [Chat]: Suggests audio mixing tools from Functions/
```

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
    ┌────▼─────┐┌────▼─────┐┌────▼─────┐
│ Goal_      ││ Tool_    ││Context_   ││Plan-     ││Profil-    │
│Detector    ││Discovery ││Gatherer  ││ner       ││er Agent  │
└─────┬─────┘└─────┬─────┘└─────┬─────┘└─────┬─────┘
      │         │         │         │         │         │
      │         │         │         │         │
   └─────────┴─────────┴─────────┴─────────┴─────────┘
        │
        │ Updates SENTER.md files with:
        │ - Goals, Context, Patterns
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│         User Focuses (omniagents)               │
├─────────────────────────────────────────────────────────┤
│  coding  │  research  │  creative  │  general  │
└─────────────────────────────────────────────────────────┘
        Each Focus has SENTER.md config
```

### Internal Focus Agents

| Agent | Purpose | Location | Output |
|-------|---------|----------|--------|
| **Router** | Route queries to best Focus | `Focuses/internal/Router/SENTER.md` | JSON: focus, reasoning |
| **Goal_Detector** | Extract goals from conversations | `Focuses/internal/Goal_Detector/SENTER.md` | JSON: goals list |
| **Tool_Discovery** | Discover tools in Functions/ | `Focuses/internal/Tool_Discovery/SENTER.md` | Creates Focuses for tools |
| **Context_Gatherer** | Update SENTER.md with context | `Focuses/internal/Context_Gatherer/SENTER.md` | Direct file updates |
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

1. **Clone Senter:**
   ```bash
   git clone https://github.com/SouthpawIN/Senter
   cd Senter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup wizard:**
   ```bash
   python3 scripts/setup_senter.py
   ```

This will:
- Download infrastructure models (Qwen2.5-Omni-3B + Nomic Embed)
- Configure your central model
- Create initial Focuses
- Verify all components work

### Running Senter

**CLI Mode:**
```bash
# Start Senter with async chain
python3 scripts/senter.py

# Available commands:
/list       - List all Focuses
/focus <name> - Set current Focus
/goals       - Show goals for current Focus
/discover    - Run tool discovery
/exit       - Exit
```

**TUI Mode** (in progress):
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

1. **Create Focus directory:**
   ```bash
   mkdir -p Focuses/MyNewFocus
   ```

2. **Create SENTER.md:**
   ```bash
   cat > Focuses/MyNewFocus/SENTER.md <<'EOF'
   ---
   model:
     type: gguf  # Inherit from user_profile.json
   
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
   
   ## Evolution Notes
   EOF
   ```

3. **Restart Senter:**
   ```bash
   python3 scripts/senter.py
   ```

That's it! No code changes needed.

---

## 📝 Query Processing Flow

When you ask Senter something, an async chain of agents processes your request:

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Senter OmniAgent Chain             │
└─────────────────────────────────────────────────────┘
    │
    ▼
[1. Router] → Selects target Focus (coding, research, etc.)
    │
    ├───┴─────────────┬──────────────────┐
    │                  │                  │
[2. Parallel Execution]                    │
    │                  │                  │
┌──▼──────┐    ┌──▼───────┐    ┌──▼───────────┐
│Context_  │    │Goal_     │    │Tool_        │
│Gatherer │    │Detector  │    │Discovery    │
└────┬────┘    └────┬─────┘    └────┬─────┘
     │              │              │
     │              │              └─────┬─────┐
     │                             │        │
     ▼                             ▼        ▼
All three gather information & update SENTER.md files
     │              │
     │              ▼
          Updates Focus's SENTER.md with:
          - Goals & Objectives
          - Context (conversations summary)
          - Patterns Observed
          - Evolution Notes
     │
     ▼
[3. Chat Agent] → Processes query with full context
    │
    ├─ Reads Focus's SENTER.md
    ├─ Considers goals & patterns
    ├─ Uses available tools if needed
    └─ Provides intelligent response
```

### Goal & Action Integration

Senter doesn't just detect goals - it tracks them through your entire journey:

1. **Detection**: Goal_Detector extracts goals from conversations
2. **Planning**: Planner breaks goals into actionable steps
3. **Execution**: Chat agent helps you complete each step
4. **Discovery**: Tool_Discovery finds tools to help
5. **Progression**: Context_Gatherer tracks your progress
6. **Feedback**: Profiler analyzes what works for you

All of this happens continuously, in parallel, in the background.

---

## 📁 Directory Structure

```
Senter/
├── Focuses/              # Focus system (replaces Topics/)
│   ├── internal/          # Internal agents (markdown writers)
│   │   ├── Router/           # Routes queries to best Focus
│   │   ├── Goal_Detector/   # Extracts goals from conversations
│   │   ├── Tool_Discovery/   # Discovers tools in Functions/
│   │   ├── Context_Gatherer/  # Updates SENTER.md files
│   │   ├── Planner/          # Breaks down goals into tasks
│   │   ├── Profiler/         # Analyzes user patterns
│   │   └── Chat/           # Main conversational agent
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
│   └── ...                # Other Functions
│
├── scripts/                # Application code
│   ├── senter.py             # CLI interface (REFACTORED)
│   ├── senter_app.py         # TUI interface (in progress)
│   ├── senter_widgets.py      # UI components
│   ├── .obsolete/            # Old scripts (backup)
│   └── ...                 # Other scripts
│
├── config/                  # Configuration
│   ├── senter_config.json    # Infrastructure models
│   └── user_profile.json      # User model + preferences
│
├── ARCHITECTURE.md          # Comprehensive architecture documentation
├── README.md                # This file
└── ...                      # Other files
```

---

## 🧩 Configuration

### senter_config.json

```json
{
  "infrastructure_models": {
    "multimodal_decoder": {
      "path": "/path/to/Qwen2.5-Omni-3B.gguf",
      "description": "Omni 3B for multimodal decoding ONLY"
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
    "goal_detection": true,
    "tool_discovery": true
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
    "creativity_level": 0.7,
    "technical_level": "intermediate",
    "language": "en",
    "timezone": "UTC"
  }
}
```

---

## 🎓 Philosophy

### The OmniAgent Pattern

**Everything is an omniagent with a unique SENTER.md configuration.**

This simple principle enables:
1. **Universal extensibility** - Add any capability = Create Focus + SENTER.md
2. **Self-documentation** - Each agent's logic is in its own config file
3. **Radical simplicity** - 65% less code, easier to understand and modify
4. **Model flexibility** - Switch models by changing one line in user_profile.json
5. **Focus-driven organization** - Knowledge organized by what matters to you
6. **Async performance** - Parallel agent calls for maximum speed

### Four Pillars

1. **Focuses are Living Knowledge Bases**: 
   - They evolve based on your conversations
   - They track goals and progress
   - They're not static categories
   - They can be created dynamically

2. **Tools are Automatically Discovered**:
   - Write code in Functions/
   - Senter discovers and creates Focuses
   - No manual configuration needed
   - Seamless integration into conversations

3. **Goals are Unlimited and Focus-Specific**:
   - No arbitrary caps
   - Each Focus has its own goals
   - Goals are tracked through completion
   - Progress is automatically updated

4. **Context is Continuously Gathered**:
   - Conversations are analyzed
   - Patterns are detected
   - Knowledge is consolidated
   - SENTER.md files are updated

### The Symbiotic Partnership

**AI brings:**
- Knowledge synthesis and organization
- Pattern recognition and goal tracking
- Tool discovery and integration
- Context awareness and memory
- Intelligent query routing

**Human brings:**
- Creativity and direction
- Goals and purpose
- Tools and capabilities
- Domain expertise
- Values and preferences

**Together:**
- More effective than either alone
- Continuous mutual improvement
- Shared understanding of goals
- Collaborative problem-solving

---

## 📊 Performance

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

```txt
llama-cpp-python>=0.3.0    # Model inference
textual>=0.50.0           # UI framework
Pillow>=10.0.0             # Image processing
pyyaml                        # YAML parsing for SENTER.md
```

### Media Dependencies

```txt
yt-dlp>=2024.1.1          # Video downloads
soundfile>=0.13.0          # Audio I/O
ffmpeg                        # Video/audio processing
```

### System Requirements

- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: CUDA-compatible with 8GB+ VRAM
- **Storage**: 20GB for models + generated content

---

## 🤝 Development

### Adding New Internal Agents

Each internal agent is a markdown writer with specific sections:

**Router** (routes queries):
- Output format section
- Routing patterns section
- Evolution notes

**Goal_Detector** (extracts goals):
- Detection patterns section
- Output format section
- Examples section

**Tool_Discovery** (discovers tools):
- Discovery process section
- Focus creation section
- Integration guidelines

**Context_Gatherer** (gathers context):
- Update strategies section
- SENTER.md sections to update
- Retention policies

**Planner** (breaks down goals):
- Planning principles section
- Task creation section
- Progress tracking section

**Profiler** (analyzes patterns):
- Analysis guidelines section
- Profile sections to update
- Confidence thresholds

**Chat** (conversational agent):
- Conversation style section
- Context usage section
- Goal awareness section

### Adding New User Focuses

Create directories with SENTER.md files that:
- Define the Focus's purpose
- List available functions/tools
- Specify output format (if any)
- Include initial context
- Leave room for evolution

---

## 🔄 Migration from v1.0

### What Changed

- **Removed**: 20+ specialized scripts (~2300 lines)
- **Added**: omniagent_async.py + omniagent_chain.py (~650 lines)
- **Refactored**: Everything uses async chain architecture
- **Replaced**: Pattern-matching with LLM-based goal detection
- **Simplified**: Tool discovery via omniagent instead of AST
- **Enhanced**: Unlimited, Focus-specific goals

### Migration Guide

1. **Backup existing configuration**
2. **Run `setup_senter.py`** to reconfigure
3. **Old Focuses still work** (SENTER.md structure preserved)
4. **Goals now unlimited** per Focus (no more cap of 3)
5. **Internal agents** now in SENTER.md format

---

## 📄 License

MIT License - see LICENSE file for details

---

**Built with ❤️ as a symbiotic AI-human partnership**
**Version**: 2.0.0 - Async Chain Architecture
**Status**: ✅ Core Implementation Complete, Documentation Complete
**Repository**: https://github.com/SouthpawIN/Senter
**Inspiring**: "Building a future where AI and humans unlock their full potential together"
