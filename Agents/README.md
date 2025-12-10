# AI Toolbox Agents

Structured agents built using the JSON Agents specification, powered by various AI models including Qwen Omni, Qwen Image, and ACE-Step.

## Directory Structure

```
Agents/
├── image/              # Image generation agents
│   └── image_generator.json
├── music/              # Music composition agents
│   └── music_composer.json
├── summarization/      # Text summarization agents
│   └── summarizer.json
├── routing/            # Request routing agents
│   └── router.json
├── analysis/           # Content analysis agents
│   └── analyzer.json
└── README.md          # This file
```

## Agent Types

### 🎨 Image Generation Agent
- **Model**: Qwen Image GGUF
- **Capabilities**: Text-to-image generation
- **Features**: High-quality image creation, customizable dimensions, various styles
- **Entry Point**: `Functions/qwen_image_gguf_generator.py`

### 🎵 Music Composer Agent
- **Model**: ACE-Step diffusion models
- **Capabilities**: Music generation with optional lyrics
- **Features**: Multiple genres, instrumental/vocal, variable duration
- **Entry Point**: `Functions/compose_music.py`

### 📝 Summarizer Agent
- **Model**: Qwen2.5-Omni-3B
- **Capabilities**: Text summarization, multimodal summarization
- **Features**: Multiple summary styles, document structure preservation
- **Entry Point**: `qwen25_omni_agent.py`

### 🚦 Router Agent
- **Model**: Qwen2.5-Omni-3B
- **Capabilities**: Intelligent request routing, intent classification
- **Features**: Multi-agent orchestration, context-aware routing
- **Entry Point**: `qwen25_omni_agent.py`

### 🔍 Analyzer Agent
- **Model**: Qwen2.5-Omni-3B
- **Capabilities**: Content analysis, sentiment analysis, quality assessment
- **Features**: Multimodal analysis, classification, detailed insights
- **Entry Point**: `qwen25_omni_agent.py`

## JSON Agents Specification

All agents follow the [JSON Agents](https://github.com/Agents-Json/Standard) specification with these profiles:

- **Core**: Basic agent identity, capabilities, and tools
- **Exec**: Runtime metadata and resource requirements
- **Graph**: Multi-agent orchestration (router only)

## Usage Examples

### Loading an Agent

```python
import json

# Load agent manifest
with open('Agents/image/image_generator.json', 'r') as f:
    agent_manifest = json.load(f)

print(f"Agent: {agent_manifest['agent']['name']}")
print(f"Capabilities: {len(agent_manifest['capabilities'])}")
```

### Agent Communication

```python
# Example: Using the router agent
from qwen25_omni_agent import QwenOmniAgent

router = QwenOmniAgent()
available_agents = ["image-generator", "music-composer", "summarizer"]

result = router.generate_response([{
    "role": "user",
    "content": [{"type": "text", "text": f"Route this request: 'Create an image of a sunset' to one of: {available_agents}"}]
}])

print(f"Routed to: {result}")
```

### Multimodal Agent Usage

```python
# Example: Using summarizer with multimodal input
summarizer = QwenOmniAgent()

messages = [
    {"role": "system", "content": "You are a summarization expert."},
    {"role": "user", "content": [
        {"type": "text", "text": "Summarize this document and analyze the included image"},
        {"type": "image", "image": "document_figure.jpg"}
    ]}
]

summary = summarizer.generate_response(messages)
```

## Agent Development Guidelines

### 1. Manifest Structure
- Follow JSON Agents v1.0 specification
- Include all required fields for chosen profiles
- Use descriptive IDs with `ajson://ai-toolbox/agents/` prefix

### 2. Capability Definition
- Define clear, specific capabilities
- Include JSON Schema for parameters
- Document expected inputs/outputs

### 3. Tool Specification
- Define function interfaces
- Include parameter validation
- Document tool purposes

### 4. Runtime Requirements
- Specify accurate resource needs
- List all dependencies
- Include environment variables

### 5. Extension Points
- Use `x-*` fields for custom extensions
- Document model-specific features
- Include performance characteristics

## Integration with AI Toolbox

### Function Registry
Agents integrate with existing Functions:
- `qwen_image_gguf_generator.py` - Image generation
- `compose_music.py` - Music composition
- `qwen25_omni_agent.py` - Multimodal analysis

### Model Registry
Agents utilize downloaded models:
- `Qwen2.5-Omni-3B` - Multimodal analysis
- `Qwen_Image-Q6_K.gguf` - Image generation
- `ACE-Step` - Music generation

### Output Integration
Generated content goes to `outputs/` directory:
- Images: `qwen_images/` subdirectory
- Music: `*.wav` files
- Analysis: Text responses

## Testing Agents

### Validation
```bash
# Validate agent manifests
cd Standard/validators/python/
python -m jsonagents.cli validate ../../Agents/*/*.json
```

### Functional Testing
```python
# Test agent loading and basic functionality
from Functions.qwen_image_gguf_generator import QwenImageGGUFGenerator

generator = QwenImageGGUFGenerator()
image = generator.generate_image_from_prompt("A beautiful sunset")
print(f"Generated image: {image.size}")
```

## Future Extensions

### Planned Agents
- **Video Editor**: Video processing and editing
- **Code Assistant**: Programming help and code generation
- **Research Assistant**: Academic paper analysis
- **Creative Writer**: Story and content generation

### Enhanced Capabilities
- **Multi-agent workflows**: Complex agent chains
- **Real-time processing**: Streaming responses
- **Plugin system**: Extensible tool integration
- **Performance monitoring**: Usage analytics

## Contributing

1. Follow the JSON Agents specification
2. Test agents thoroughly
3. Document capabilities clearly
4. Include example usage
5. Update this README

---

**Built with ❤️ using the JSON Agents specification**