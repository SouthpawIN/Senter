# Senter

> **The Markdown OS for AI Agents**

Senter is a minimalist, self-aware AI agent orchestrated entirely through Markdown. Unlike traditional agents with hardcoded logic, Senter's personality, tools, and lifecycle are defined within an Obsidian-compatible vault.

## 🚀 Key Features

- **Markdown-as-Code:** Configure agents, skills, and hooks simply by editing `.md` files.
- **Senter Select Engine:** High-performance two-stage retrieval (Embedding Top 4 ➔ LLM Decision Top 1) for both skills and memory.
- **Terminal-First:** A dedicated persistent terminal (via Zellij) for executing complex background tasks.
- **Obsidian Integration:** Use your personal vault as the "Live OS" for your agent.
- **Progressive Disclosure:** Skills and memories are loaded only when needed, keeping the context laser-focused.

## 🏗️ Architecture

- **`server.py`:** A unified multi-model server (FastAPI) that manages local GGUF models via `llama-server`. Optimized for GLM 4.7 Flash.
- **`senter.py` (The Kernel):** A lightweight "bootloader" that indexes your Markdown vault and handles the LLM selection loop.
- **The Vault:**
  - `AGENTS/`: Persona and goal definitions.
  - `SKILLS/`: Capability definitions with bash tool blocks.
  - `HOOKS/`: Event-driven lifecycle triggers (e.g., `on_startup`).
  - `STATE/`: Persistent chat history and long-term memory.

## 🛠️ Quick Start

### 1. Start the Server
```bash
python3 server.py
```

### 2. Configure the Vault
Senter looks for its brain at `~/.senter/vault`. You can symlink this to your Obsidian vault:
```bash
ln -s ~/.senter/vault "~/Documents/Obsidian Vault/Senter"
```

### 3. Launch Senter
```bash
python3 senter.py
```

## 🧠 Senter Select
Senter handles "unlimited skills" by performing a dual-layer search:
1. **Embedding Stage:** Uses `nomic-embed-text` to find the 4 most relevant tools/memories.
2. **LLM Stage:** Performs a hidden inference where the model picks the #1 best context.
3. **Final Response:** Only the selected context is injected, ensuring maximum accuracy and minimum token waste.

## 🛡️ License
MIT