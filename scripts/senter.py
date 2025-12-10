#!/usr/bin/env python3
"""
Senter - Universal AI Personal Assistant
JSON-driven agent system with topic-based organization and SENTER.md context files
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from qwen25_omni_agent import QwenOmniAgent
from Functions.qwen_image_gguf_generator import QwenImageGGUFGenerator
from Functions.compose_music import compose_music

class Senter:
    """Universal AI Personal Assistant with JSON-driven agent system"""

    def __init__(self):
        # Core directories
        self.topics_dir = Path("Topics")
        self.agents_dir = Path("Agents")
        self.outputs_dir = Path("outputs")

        self.topics_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)

        # Core components
        self.omni_agent = None
        self.image_generator = None
        self.music_initialized = False

        # Agent registry from JSON manifests
        self.agents = self._load_agent_manifests()
        self.topic_agents = self._load_topic_agents()
        self.topic_agent_map = self._load_topic_agent_map()
        self.locked_agent = None  # For user-locked agent selection

        # Initialize core agent
        self._init_omni_agent()



    def _load_agent_manifests(self) -> Dict[str, Dict]:
        """Load all available agents from Agents directory"""
        agents = {}

        if self.agents_dir.exists():
            for json_file in self.agents_dir.rglob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        agent_data = json.load(f)
                        agent_id = agent_data['agent']['id'].split('/')[-1]
                        agents[agent_id] = agent_data
                except Exception as e:
                    print(f"Failed to load agent {json_file}: {e}")

        return agents

    def _load_topic_agents(self) -> Dict[str, Dict]:
        """Load topic-specific agents"""
        topic_agents = {}

        for topic_dir in self.topics_dir.iterdir():
            if topic_dir.is_dir():
                agent_file = topic_dir / f"{topic_dir.name}.json"
                if agent_file.exists():
                    try:
                        with open(agent_file, 'r') as f:
                            agent_data = json.load(f)
                            topic_agents[topic_dir.name] = agent_data
                    except Exception as e:
                        print(f"Failed to load topic agent {agent_file}: {e}")

        return topic_agents

    def _load_topic_agent_map(self) -> Dict[str, str]:
        """Load topic to agent mapping"""
        map_file = Path("config/topic_agent_map.json")
        if map_file.exists():
            try:
                with open(map_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load topic_agent_map: {e}")
        return {}

    def _init_omni_agent(self):
        """Initialize the core Qwen Omni agent"""
        if self.omni_agent is None:
            print("🤖 Initializing Senter's multimodal core...")
            self.omni_agent = QwenOmniAgent()
            print("✅ Senter core ready!")

    def _init_image_generator(self):
        """Initialize the image generator"""
        if self.image_generator is None:
            print("🎨 Initializing image generation capabilities...")
            self.image_generator = QwenImageGGUFGenerator()
            print("✅ Image generation ready!")

    def _ensure_music_setup(self):
        """Ensure music generation is set up"""
        if not self.music_initialized:
            print("🎵 Music composition capabilities ready!")
            self.music_initialized = True

    def chat(self, message: str, **kwargs) -> str:
        """
        Main chat interface - routes to appropriate topic/agent
        """
        # Determine topic and agent
        topic, agent = self._route_request(message, **kwargs)

        # Update topic context
        self._update_topic_context(topic, "user", message, **kwargs)

        # Get response from appropriate agent
        response = self._get_agent_response(agent, message, **kwargs)

        # Update topic context with response
        self._update_topic_context(topic, "senter", response, **kwargs)

        return response

    def _route_request(self, message: str, **kwargs) -> Tuple[str, str]:
        """
        Route request to appropriate topic-agent pair
        Uses combined selection for efficiency, respects agent locking
        """
        if self.locked_agent:
            # Use locked agent, still select topic
            available_topics = list(self.topic_agent_map.keys())
            topic = self._llm_select_topic_for_locked_agent(message, available_topics, **kwargs)
            return topic, self.locked_agent

        available_pairs = list(self.topic_agent_map.items())

        # Use LLM to select best topic-agent pair
        return self._llm_select_topic_agent_pair(message, available_pairs, **kwargs)

    def _embed_filter_topics(self, message: str, topics: List[str], **kwargs) -> List[str]:
        """
        Use nomic embed to filter topics down to top 4 most similar
        """
        # TODO: Implement nomic embed filtering
        # For now, return first 4
        return topics[:4]

    def _llm_select_topic_agent_pair(self, message: str, pairs: List[Tuple[str, str]], **kwargs) -> Tuple[str, str]:
        """
        Use LLM to select best topic-agent pair from available mappings
        """
        pair_options = [f"{topic} (agent: {agent})" for topic, agent in pairs]

        pair_prompt = f"""
        Analyze this user message and select the most appropriate topic-agent pair from the available options.

        User Message: "{message}"

        Available Topic-Agent Pairs:
        {', '.join(pair_options)}

        Respond with JSON:
        {{
            "selected_pair": "topic_name",
            "reasoning": "brief explanation"
        }}
        """

        # Use omni agent for routing decision
        if self.omni_agent is None:
            self._init_omni_agent()
        routing_response = self.omni_agent.generate_response([{
            "role": "user",
            "content": [{"type": "text", "text": pair_prompt}]
        }])

        try:
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', routing_response, re.DOTALL)
            if json_match:
                routing_data = json.loads(json_match.group())
                selected_topic = routing_data.get("selected_pair", "general")
                # Get agent from map
                selected_agent = self.topic_agent_map.get(selected_topic, "senter")
                return selected_topic, selected_agent
        except:
            pass

        # Fallback
        return "general", "senter"

    def _llm_select_topic_for_locked_agent(self, message: str, topics: List[str], **kwargs) -> str:
        """
        Use LLM to select best topic when agent is locked
        """
        topic_prompt = f"""
        Analyze this user message and select the most appropriate topic from the available topics.
        The agent is already selected.

        User Message: "{message}"

        Available Topics: {', '.join(topics)}

        Respond with JSON:
        {{
            "topic": "selected_topic",
            "reasoning": "brief explanation"
        }}
        """

        # Use omni agent for routing decision
        if self.omni_agent is None:
            self._init_omni_agent()
        routing_response = self.omni_agent.generate_response([{
            "role": "user",
            "content": [{"type": "text", "text": topic_prompt}]
        }])

        try:
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', routing_response, re.DOTALL)
            if json_match:
                routing_data = json.loads(json_match.group())
                topic = routing_data.get("topic", "general")
                return topic
        except:
            pass

        # Fallback
        return "general"

    def _get_agent_response(self, agent_type: str, message: str, **kwargs) -> str:
        """Get response from specified agent type"""
        if agent_type == "image_generator":
            return self._handle_image_generation(message, **kwargs)
        elif agent_type == "music_composer":
            return self._handle_music_composition(message, **kwargs)
        elif agent_type == "creative":
            return self._handle_creative_task(message, **kwargs)
        else:
            # Use omni agent for general/analytical tasks
            if self.omni_agent is None:
                self._init_omni_agent()
            return self.omni_agent.chat(message, **kwargs)

    def _handle_image_generation(self, request: str, **kwargs) -> str:
        """Handle image generation requests"""
        self._init_image_generator()

        prompt = self._extract_image_prompt(request)

        try:
            print(f"🎨 Generating image: {prompt}")
            image = self.image_generator.generate_image_from_prompt(prompt)
            if image:
                output_path = self.image_generator.save_and_open_image(image, f"senter_{prompt[:30].replace(' ', '_')}")
                return f"🎨 Image generated successfully! Saved to: {output_path}"
            else:
                return "❌ Image generation failed"
        except Exception as e:
            return f"❌ Image generation error: {e}"

    def _handle_music_composition(self, request: str, **kwargs) -> str:
        """Handle music composition requests"""
        self._ensure_music_setup()

        prompt = self._extract_music_prompt(request)

        try:
            print(f"🎵 Composing music: {prompt}")
            generated_files = compose_music(prompt=prompt, duration_seconds=30)
            if generated_files:
                return f"🎵 Music composed successfully! Files: {', '.join(generated_files)}"
            else:
                return "❌ Music composition failed"
        except Exception as e:
            return f"❌ Music composition error: {e}"

    def _handle_creative_task(self, request: str, **kwargs) -> str:
        """Handle creative writing and content generation"""
        if self.omni_agent is None:
            self._init_omni_agent()

        creative_messages = [
            {"role": "system", "content": "You are a creative assistant specializing in writing, storytelling, and content creation."},
            {"role": "user", "content": [
                {"type": "text", "text": request}
            ]}
        ]

        return self.omni_agent.generate_response(creative_messages)

    def _update_topic_context(self, topic: str, speaker: str, content: str, **kwargs):
        """Update the SENTER.md file for the topic with new context"""
        topic_dir = self.topics_dir / topic
        topic_dir.mkdir(exist_ok=True)

        senter_md_path = topic_dir / "SENTER.md"

        # Load existing content
        existing_content = ""
        if senter_md_path.exists():
            with open(senter_md_path, 'r') as f:
                existing_content = f.read()

        # Create update entry
        update_entry = f"\n## {speaker.title()} Interaction\n**Timestamp:** {self._get_timestamp()}\n**Content:** {content}\n"

        # Add media context if present
        if kwargs.get('image_path'):
            update_entry += f"**Image:** {kwargs['image_path']}\n"
        if kwargs.get('audio_path'):
            update_entry += f"**Audio:** {kwargs['audio_path']}\n"

        # Use summarizer to update context
        updated_content = self._summarize_and_update_context(existing_content, update_entry, topic)

        # Write updated content
        with open(senter_md_path, 'w') as f:
            f.write(updated_content)

    def _summarize_and_update_context(self, existing_content: str, new_entry: str, topic: str) -> str:
        """Use summarizer agent to maintain concise, relevant context"""
        self._init_omni_agent()

        summary_prompt = f"""
        Update the context summary for topic "{topic}".

        Existing Context Summary:
        {existing_content}

        New Entry to Integrate:
        {new_entry}

        Create an updated, concise summary that:
        1. Maintains key learnings about user preferences
        2. Tracks successful tool usage patterns
        3. Preserves important context without redundancy
        4. Keeps the summary focused and actionable

        Updated Context Summary:
        """

        if self.omni_agent is None:
            self._init_omni_agent()
        summary_response = self.omni_agent.generate_response([{
            "role": "user",
            "content": [{"type": "text", "text": summary_prompt}]
        }])

        return summary_response.strip()

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _extract_image_prompt(self, request: str) -> str:
        """Extract image description from user request"""
        prompt_keywords = ['draw', 'create image', 'generate image', 'make picture']
        for keyword in prompt_keywords:
            if keyword in request.lower():
                return request.replace(keyword, '').strip()
        return request

    def _extract_music_prompt(self, request: str) -> str:
        """Extract music description from user request"""
        music_keywords = ['compose', 'create music', 'generate music', 'make song']
        for keyword in music_keywords:
            if keyword in request.lower():
                return request.replace(keyword, '').strip()
        return request

    def create_topic_agent(self, topic_name: str, description: str) -> bool:
        """Create a new topic with its own agent and SENTER.md"""
        topic_dir = self.topics_dir / topic_name
        topic_dir.mkdir(exist_ok=True)

        # Create topic agent manifest
        agent_manifest = {
            "manifest_version": "1.0",
            "profiles": ["core", "exec"],
            "agent": {
                "id": f"ajson://ai-toolbox/topics/{topic_name}",
                "name": f"{topic_name.title()} Agent",
                "version": "1.0.0",
                "description": description,
                "author": "Senter",
                "tags": ["topic-specific", "contextual", topic_name]
            },
            "capabilities": [
                {
                    "id": "topic-expertise",
                    "description": f"Specialized knowledge and context for {topic_name}",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"Question or request related to {topic_name}"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ],
            "tools": [
                {
                    "id": "topic_chat",
                    "type": "function",
                    "description": f"Chat with specialized knowledge about {topic_name}",
                    "function": {
                        "name": "topic_chat",
                        "description": f"Provide contextual responses about {topic_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "User message"
                                }
                            },
                            "required": ["message"]
                        }
                    }
                }
            ],
            "modalities": {
                "input": ["text", "image", "audio"],
                "output": ["text"]
            },
            "runtime": {
                "type": "python",
                "version": ">=3.8",
                "entrypoint": "qwen25_omni_agent.py",
                "dependencies": [
                    "torch>=2.0.0",
                    "transformers>=4.40.0",
                    "accelerate",
                    "numpy",
                    "soundfile",
                    "scipy"
                ],
                "environment": {
                    "MODEL_PATH": "/home/sovthpaw/Models/Qwen2.5-Omni-3B",
                    "TOPIC": topic_name
                }
            },
            "extensions": {
                "x-topic": {
                    "topic_name": topic_name,
                    "context_file": "SENTER.md",
                    "learning_enabled": True,
                    "specialization": description
                }
            }
        }

        # Save agent manifest
        agent_file = topic_dir / f"{topic_name}.json"
        with open(agent_file, 'w') as f:
            json.dump(agent_manifest, f, indent=2)

        # Create initial SENTER.md
        senter_md_path = topic_dir / "SENTER.md"
        initial_content = f"""# {topic_name.title()} - Senter Context Summary

## Overview
{description}

## Key Learnings
- Topic created: {self._get_timestamp()}
- Context will be learned through interactions

## User Preferences
- To be determined through usage

## Tool Usage Patterns
- To be determined through usage

## Important Context
- Maintains specialized knowledge for {topic_name}
- Updates automatically through summarization
"""

        with open(senter_md_path, 'w') as f:
            f.write(initial_content)

        # Reload topic agents
        self.topic_agents = self._load_topic_agents()

        return True

def main():
    parser = argparse.ArgumentParser(description="Senter - Universal AI Personal Assistant")
    parser.add_argument("message", nargs="?", help="Your message or request to Senter")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--topic", help="Specific topic to use")
    parser.add_argument("--create-topic", help="Create a new topic with description")
    parser.add_argument("--list-topics", action="store_true", help="List available topics")

    args = parser.parse_args()

    senter = Senter()

    if args.list_topics:
        print("Available Topics:")
        for topic in senter.topic_agents.keys():
            print(f"  - {topic}")
        print(f"\nAvailable Agents: {list(senter.agents.keys())}")

    elif args.create_topic:
        if not args.message:
            print("Error: Provide topic description with message")
            return
        success = senter.create_topic_agent(args.topic or "new_topic", args.message)
        if success:
            print(f"✅ Created topic agent: {args.topic}")
        else:
            print("❌ Failed to create topic agent")

    elif args.message:
        kwargs = {}
        if args.image:
            kwargs['image_path'] = args.image
        if args.audio:
            kwargs['audio_path'] = args.audio

        response = senter.chat(args.message, **kwargs)
        print(response)

    else:
        print("Senter - Universal AI Personal Assistant")
        print("Use --help for usage information")

if __name__ == "__main__":
    main()