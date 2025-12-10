# Senter AI Personal Assistant

A sophisticated multimodal AI personal assistant built with a JSON-native agent architecture, topic-based learning, and a modern terminal user interface.

![Senter Logo](https://img.shields.io/badge/Senter-AI%20Assistant-00ffaa?style=for-the-badge&logo=ai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## 🌟 Overview

Senter is a universal AI personal assistant that combines multimodal capabilities (text, vision, audio) with intelligent agent orchestration. Built on the JSON Agents specification, Senter features self-learning through topic-based contexts and provides a seamless terminal experience.

### Key Features
- 🤖 **Agent-First Architecture**: Everything is an agent - defined in JSON for maximum flexibility
- 🎯 **Multimodal Processing**: Text, image, audio, and video understanding via Qwen2.5-Omni
- 📚 **Topic-Based Learning**: Self-evolving knowledge through SENTER.md context files
- 🎨 **Creative Generation**: Built-in image and music generation capabilities
- 🖥️ **Modern TUI**: Beautiful terminal interface with real-time interactions
- 🔄 **Parallel Processing**: Multiple agents work simultaneously without blocking
- 📊 **Self-Improvement**: Continuous learning from user interactions

## 🏗️ Architecture

### Core Components

#### 🤖 Agent System (`Agents/`)
JSON-defined agents following the JSON Agents specification. Each agent has:
- **Identity & Capabilities**: What the agent can do
- **Tools & Functions**: Available actions
- **Runtime Configuration**: Execution environment
- **Self-Learning**: Associated SENTER.md for evolution

**Core Agents:**
- `senter.json` - Main orchestrator
- `analyzer.json` - Content analysis and classification
- `creative_writer.json` - Content generation
- `researcher.json` - Web research and information gathering
- `summarizer.json` - Content condensation
- `qwen_image_gguf_generator.json` - Image generation
- `compose_music.json` - Music composition

#### 🛠️ Functions (`Functions/`)
Python implementations for specialized capabilities:
- `qwen_image_gguf_generator.py` - GGUF-based image generation
- `compose_music.py` - Audio composition using ACE-Step

#### 📁 Topics (`Topics/`)
Self-learning knowledge base with topic-specific contexts:
- `agents/` - Agent evolution notes
- `coding/` - Programming preferences and patterns
- `creative/` - Artistic preferences and outputs
- `general/` - General conversation context
- `research/` - Research methodologies
- `user_personal/` - User preferences and goals

Each topic contains a `SENTER.md` file that evolves with user interactions.

#### ⚙️ Configuration (`config/`)
- `senter_config.json` - System settings and model configurations
- `topic_agent_map.json` - Maps topics to their primary agents
- `user_profile.json` - User preferences and learning profile

#### 📜 Scripts (`scripts/`)
Core application components:
- `senter_app.py` - Main Textual TUI application
- `senter.py` - Core agent orchestration logic
- `qwen25_omni_agent.py` - Multimodal model interface
- `background_processor.py` - Parallel task management
- `function_agent_generator.py` - Auto-generates agents from functions
- `model_server_manager.py` - Manages vLLM servers

#### 🎨 Interface (`senter.tcss`)
Textual CSS styling for the terminal interface with matrix-inspired green theme.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 16GB+ RAM (for Qwen2.5-Omni-3B)
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SouthpawIN/Senter.git
cd Senter
```

2. **Download models:**
```bash
python setup.py
```
This downloads:
- Qwen2.5-Omni-3B (transformers model)
- Qwen-Image-0.5B (GGUF for image generation)
- nomic-embed-text-v1.5 (embeddings)

3. **Install dependencies:**
```bash
pip install torch transformers textual huggingface_hub
# Additional dependencies for full functionality
pip install vllm llama-cpp-python diffusers librosa
```

4. **Launch Senter:**
```bash
python scripts/senter_app.py
```

## 🎯 Usage

### Basic Interaction
- Type messages in the main chat area
- Use `/commands` for available slash commands
- The system automatically routes queries to appropriate agents

### Agent System
Senter uses intelligent routing to select the best agent for each query:
- **General Chat** → Senter orchestrator
- **Creative Tasks** → Creative Writer agent
- **Analysis** → Analyzer agent
- **Research** → Researcher agent
- **Image Generation** → Qwen Image Generator
- **Music Creation** → Music Composer

### Topic Learning
Each conversation contributes to topic-specific knowledge:
- User preferences are learned and stored
- Patterns are recognized across interactions
- Context evolves through SENTER.md files

### Advanced Features
- **Parallel Processing**: Multiple agents work simultaneously
- **Background Learning**: Continuous context analysis
- **Goal Tracking**: Automatic objective detection and monitoring
- **Model Switching**: Dynamic model assignment based on tasks

## 🔧 Development

### Project Structure
```
Senter/
├── Agents/           # JSON agent definitions
├── Functions/        # Specialized Python functions
├── Topics/           # Learning contexts (SENTER.md files)
├── config/           # Configuration files
├── scripts/          # Core application code
├── Models/           # Downloaded models (gitignored)
├── AGENT.md          # Agent development guidelines
├── AGENTS.md         # Code style guidelines
├── senter.tcss       # TUI styling
├── setup.py          # Model download script
└── README.md         # This file
```

### Adding New Agents

1. **Create JSON Manifest** in `Agents/`:
```json
{
  "manifest_version": "1.0",
  "agent": {
    "id": "ajson://ai-toolbox/agents/my_agent",
    "name": "My Agent",
    "description": "What it does",
    "version": "1.0.0"
  },
  "capabilities": [...],
  "tools": [...],
  "runtime": {...}
}
```

2. **Implement Functions** in `Functions/` if needed

3. **Create Topic** in `Topics/my_agent/SENTER.md`

4. **Update Mapping** in `config/topic_agent_map.json`

### Code Guidelines
- Follow PEP 8 with 4-space indentation
- Use type hints throughout
- Comprehensive docstrings required
- JSON Agents specification compliance
- Textual framework for UI components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the agent development guidelines in `AGENT.md`
4. Test thoroughly with `python scripts/test_components.py`
5. Submit a pull request

## 📋 Requirements

### Core Dependencies
- `torch>=2.0.0`
- `transformers>=4.40.0`
- `textual>=0.50.0`
- `huggingface_hub`

### Optional (Enhanced Features)
- `vllm` - High-performance model serving
- `llama-cpp-python` - GGUF model inference
- `diffusers` - Advanced image generation
- `librosa` - Audio processing
- `soundfile` - Audio I/O

### System Requirements
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: CUDA-compatible with 8GB+ VRAM
- **Storage**: 20GB for models + generated content

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **No Telemetry**: No data collection or external communications
- **Isolated Execution**: Agents run in controlled environments
- **Input Validation**: All inputs sanitized and validated

## 📈 Roadmap

- [ ] Third-party agent marketplace
- [ ] Plugin system for custom integrations
- [ ] Multi-user collaboration features
- [ ] Advanced workflow automation
- [ ] API for external applications

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built using amazing open-source projects:
- **Qwen Models** by Alibaba Cloud
- **Textual** for terminal UI
- **vLLM** for model serving
- **JSON Agents** specification
- **ACE-Step** for music generation

---

**Senter**: Your intelligent, evolving AI companion. 🤖✨</content>
<parameter name="filePath">Senter/README.md