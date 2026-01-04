#!/usr/bin/env python3
"""
Senter - Universal AI Personal Assistant
JSON-driven agent system with Focus-based organization and SENTER.md context files
LAZY LOADING - Models load only when needed
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add directories to path (order matters)
senter_root = Path(__file__).parent.parent
sys.path.insert(0, str(senter_root))
sys.path.insert(1, str(senter_root / "scripts"))

from Focuses.senter_md_parser import SenterMdParser
from Focuses.focus_factory import FocusFactory


class Senter:
    """Universal AI Personal Assistant with lazy model loading"""

    def __init__(self):
        self.focuses_dir = Path("Focuses")
        self.agents_dir = Path("Agents")
        self.outputs_dir = Path("outputs")
        self.senter_root = senter_root

        self.focuses_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)

        # Core components (lightweight)
        self.parser = SenterMdParser(self.senter_root)
        self.factory = FocusFactory(self.senter_root)

        # LAZY: Only load when needed
        self.omni_agent = None
        self.image_generator = None
        self.music_initialized = False

        # Agent registry
        self.agents = self._load_agent_manifests()
        self.focus_agents = self._load_focus_agents()
        self.focus_agent_map = self._load_focus_agent_map()
        self.locked_agent = None

        print("Senter initialized (models load on demand)")

    def _load_agent_manifests(self) -> Dict[str, Dict]:
        agents = {}
        if self.agents_dir.exists():
            for json_file in self.agents_dir.rglob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        agent_data = json.load(f)
                        agent_id = agent_data["agent"]["id"].split("/")[-1]
                        agents[agent_id] = agent_data
                except Exception as e:
                    print(f"Failed to load agent {json_file}: {e}")
        return agents

    def _load_focus_agents(self) -> Dict[str, Dict]:
        focus_agents = {}
        for focus_dir in self.focuses_dir.iterdir():
            if focus_dir.is_dir():
                senter_file = focus_dir / "SENTER.md"
                if senter_file.exists():
                    try:
                        config = self.parser.load_focus_config(focus_dir.name)
                        focus_agents[focus_dir.name] = config
                    except Exception as e:
                        print(f"Failed to load Focus {focus_dir.name}: {e}")
        return focus_agents

    def _load_focus_agent_map(self) -> Dict[str, str]:
        map_file = self.senter_root / "config" / "focus_agent_map.json"
        if map_file.exists():
            try:
                with open(map_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load focus_agent_map: {e}")
        default_map = {}
        for focus_name, config in self.focus_agents.items():
            agent_id = config.get("agent", {}).get("id", "senter")
            if agent_id:
                default_map[focus_name] = agent_id
        return default_map

    def _ensure_omni_agent(self):
        """Lazy load SenterOmniAgent only when needed"""
        if self.omni_agent is None:
            print("Loading SenterOmniAgent...")
            try:
                from Functions.omniagent import SenterOmniAgent
                self.omni_agent = SenterOmniAgent(senter_root=self.senter_root)
                print("SenterOmniAgent ready!")
            except Exception as e:
                print(f"Failed to load SenterOmniAgent: {e}")
                self.omni_agent = None

    def chat(self, message: str, **kwargs) -> str:
        focus, agent = self._route_request(message, **kwargs)
        self._update_focus_context(focus, "user", message, **kwargs)
        response = self._get_agent_response(agent, message, **kwargs)
        self._update_focus_context(focus, "senter", response, **kwargs)
        return response

    def _route_request(self, message: str, **kwargs) -> Tuple[str, str]:
        available_focuses = self.parser.list_all_focuses()
        from senter_selector import select_topic_and_agent
        selected_focus, agent, reasoning = select_topic_and_agent(
            query=message,
            available_topics=available_focuses,
            available_agents=list(self.agents.keys()),
            context="",
        )
        print(f"Focus: {selected_focus} | Agent: {agent}")
        return selected_focus, agent

    def _update_focus_context(self, focus: str, speaker: str, content: str, **kwargs):
        focus_dir = self.focuses_dir / focus
        senter_md_path = focus_dir / "SENTER.md"
        if not senter_md_path.exists():
            focus_dir.mkdir(exist_ok=True)
            senter_md_path.write_text("", encoding="utf-8")
        existing_content = senter_md_path.read_text(encoding="utf-8")
        update_entry = f"\n## {speaker.title()} Interaction\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nContent: {content}\n"
        senter_md_path.write_text(existing_content + update_entry, encoding="utf-8")

    def _get_agent_response(self, agent_type: str, message: str, **kwargs) -> str:
        self._ensure_omni_agent()
        if self.omni_agent is None:
            return "SenterOmniAgent not available"
        focus = kwargs.get("focus", "general")
        focus_config = self.parser.load_focus_config(focus)
        focus_system_prompt = focus_config.get("system_prompt", "")
        combined_prompt = f"{focus_system_prompt}\n\nUser: {message}"
        try:
            response = self.omni_agent.process_text(combined_prompt, max_tokens=512)
            return response
        except Exception as e:
            return f"Error from agent: {e}"

    def list_focuses(self) -> List[str]:
        return self.parser.list_all_focuses()

    def create_focus(self, focus_name: str, initial_context: str = "") -> bool:
        try:
            focus_dir = self.factory.create_focus(focus_name, initial_context)
            print(f"Created Focus: {focus_name}")
            return True
        except Exception as e:
            print(f"Failed to create Focus {focus_name}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Senter - Universal AI Personal Assistant")
    parser.add_argument("message", help="Your message to Senter")
    parser.add_argument("--list-focuses", action="store_true", help="List available Focuses")
    parser.add_argument("--create-focus", help="Create a new Focus")
    parser.add_argument("--focus-description", help="Description for new Focus")
    args = parser.parse_args()

    senter = Senter()

    if args.list_focuses:
        print("\nAvailable Focuses:")
        for focus in senter.list_focuses():
            print(f"  - {focus}")

    elif args.create_focus:
        if not args.focus_description:
            print("Error: --focus-description required")
        else:
            senter.create_focus(args.focus_description, args.focus_description)

    elif args.message:
        response = senter.chat(args.message)
        print(f"\nResponse:\n{response}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
