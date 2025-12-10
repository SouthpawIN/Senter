#!/usr/bin/env python3
"""
Senter Chat Interface
Main terminal user interface for Senter AI Personal Assistant
"""

import os
import sys
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Input, Button, Static, TextArea,
    ListView, ListItem, Label, ProgressBar, Tabs, Tab
)
from textual import events
from textual.css.query import NoMatches

# Import Senter components
try:
    from senter_selector import SenterSelector
    from qwen25_omni_agent import QwenOmniAgent
except ImportError as e:
    print(f"Warning: Could not import Senter components: {e}")
    SenterSelector = None
    QwenOmniAgent = None

class SenterChat(App):
    """Main Senter chat interface using Textual"""

    CSS = """
    Screen {
        background: ansi_black;
    }

    #chat-container {
        height: 1fr;
        border: solid ansi_green;
        margin: 1;
    }

    #sidebar {
        width: 35;
        border: solid ansi_bright_green;
        margin: 1 0 1 1;
        padding: 1;
        background: ansi_black;
    }

    #main-chat {
        height: 1fr;
        margin: 1;
        border: solid ansi_green;
        background: ansi_black;
    }

    #input-container {
        height: 3;
        border: solid ansi_green;
        margin: 0 1 1 1;
        padding: 0 1;
        background: ansi_black;
    }

    #message-input {
        width: 1fr;
        margin: 0.5 0;
        background: ansi_black;
        color: ansi_bright_green;
        border: solid ansi_green;
    }

    #message-input:focus {
        border: solid ansi_bright_green;
    }

    .message {
        margin: 0.5 0;
        padding: 0.5;
        border-radius: 0.25;
    }

    .user-message {
        background: ansi_green;
        color: ansi_black;
        text-align: right;
        margin-left: 20;
    }

    .agent-message {
        background: ansi_black;
        border: solid ansi_bright_green;
        color: ansi_bright_green;
        margin-right: 20;
    }

    .topic-info {
        background: ansi_green;
        color: ansi_black;
        padding: 0.5;
        margin: 0.5 0;
        border-radius: 0.25;
        text-align: center;
        text-style: bold;
    }

    .goal-item {
        margin: 0.25 0;
        padding: 0.25;
        background: ansi_bright_green;
        color: ansi_black;
        border-radius: 0.25;
    }

    .status-indicator {
        color: ansi_bright_green;
        text-align: center;
        margin: 0.5 0;
        text-style: bold;
    }

    #title {
        content-align: center middle;
        height: 6;
        margin: 1;
        background: ansi_green;
        color: ansi_black;
        text-style: bold;
    }

    Header {
        background: ansi_green;
        color: ansi_black;
    }

    Footer {
        background: ansi_green;
        color: ansi_black;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear Chat"),
        ("ctrl+t", "show_topics", "Show Topics"),
    ]

    def __init__(self):
        super().__init__()
        self.selector = SenterSelector() if SenterSelector else None
        self.current_topic = "general"
        self.conversation_history = []
        self.background_tasks = []
        self.user_profile = self._load_user_profile()
        self.config = self._load_config()

        # Start background processing
        self._start_background_processing()

    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile"""
        profile_path = Path(__file__).parent.parent / "config" / "user_profile.json"
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except:
            return {"goals": [], "preferences": {}}

    def _load_config(self) -> Dict[str, Any]:
        """Load Senter configuration"""
        config_path = Path(__file__).parent.parent / "config" / "senter_config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {"parallel_processing": True}

    def _start_background_processing(self):
        """Start background processing threads"""
        if self.config.get("parallel_processing", True):
            # Context analysis thread
            context_thread = threading.Thread(
                target=self._background_context_analysis,
                daemon=True
            )
            context_thread.start()
            self.background_tasks.append(context_thread)

            # User profiling thread
            profile_thread = threading.Thread(
                target=self._background_user_profiling,
                daemon=True
            )
            profile_thread.start()
            self.background_tasks.append(profile_thread)

    def _background_context_analysis(self):
        """Background context analysis and SENTER.md updates"""
        while True:
            try:
                # Analyze recent conversations for topic patterns
                if len(self.conversation_history) > 5:
                    self._analyze_conversation_context()

                # Update SENTER.md files
                self._update_topic_context()

                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                print(f"Context analysis error: {e}")
                time.sleep(60)

    def _background_user_profiling(self):
        """Background user profiling and goal detection"""
        while True:
            try:
                if len(self.conversation_history) > 10:
                    self._analyze_user_patterns()
                    self._detect_potential_goals()

                time.sleep(60)  # Analyze every minute
            except Exception as e:
                print(f"User profiling error: {e}")
                time.sleep(120)

    def _analyze_conversation_context(self):
        """Analyze conversation for topic context"""
        # Simple topic detection based on keywords
        recent_messages = self.conversation_history[-10:]
        topics_detected = []

        topic_keywords = {
            "coding": ["code", "programming", "python", "function", "class", "debug"],
            "creative": ["music", "image", "art", "design", "create", "generate"],
            "research": ["search", "find", "information", "learn", "study"],
            "personal": ["schedule", "reminder", "task", "goal", "plan"]
        }

        for message in recent_messages:
            content = message.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    if topic not in topics_detected:
                        topics_detected.append(topic)

        if topics_detected:
            self.current_topic = topics_detected[0]  # Use first detected topic
            self._update_sidebar_topic()

    def _analyze_user_patterns(self):
        """Analyze user interaction patterns"""
        # This would implement more sophisticated user profiling
        pass

    def _detect_potential_goals(self):
        """Detect potential user goals from conversation"""
        # This would implement goal detection logic
        pass

    def _update_topic_context(self):
        """Update SENTER.md files with new context"""
        topic_dir = Path(__file__).parent.parent / "Topics" / self.current_topic
        senter_file = topic_dir / "SENTER.md"

        if senter_file.exists():
            try:
                with open(senter_file, 'a') as f:
                    f.write(f"\n## Context Update ({datetime.now().isoformat()})\n")
                    f.write(f"- Current conversation length: {len(self.conversation_history)}\n")
                    f.write(f"- Active topic: {self.current_topic}\n")
                    f.write("- Background analysis active\n\n")
            except Exception as e:
                print(f"Error updating SENTER.md: {e}")

    def _update_sidebar_topic(self):
        """Update the sidebar with current topic"""
        try:
            topic_label = self.query_one("#current-topic", Static)
            topic_label.update(f"📁 {self.current_topic.title()}")
        except NoMatches:
            pass

    def compose(self) -> ComposeResult:
        """Compose the UI layout"""
        yield Header()

        # ASCII Art Title
        title = """
███████╗███████╗███╗   ██╗████████╗███████╗██████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
        """
        yield Static(title, id="title")

        with Horizontal():
            # Main chat area
            with Vertical(id="chat-container"):
                # Chat messages area
                yield TextArea("", id="chat-messages", readonly=True)

                # Input area
                with Container(id="input-container"):
                    yield Input(placeholder="Type your message to Senter...", id="message-input")
                    yield Button("Send", id="send-button", variant="primary")

            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("🎯 Senter AI Assistant", classes="topic-info")

                # Current topic
                yield Static(f"📁 {self.current_topic.title()}", id="current-topic", classes="topic-info")

                # Goals section
                yield Static("🎯 Active Goals", classes="topic-info")
                goals_container = Container(id="goals-container")
                for goal in self.user_profile.get("goals", [])[:3]:  # Show top 3 goals
                    goals_container.mount(Static(f"• {goal}", classes="goal-item"))
                yield goals_container

                # Status indicators
                yield Static("🔄 Background Processing: Active", classes="status-indicator")
                yield Static("🤖 Agents: 10 Available", classes="status-indicator")
                yield Static("🧠 Models: Ready", classes="status-indicator")

        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "send-button":
            await self._send_message()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "message-input":
            await self._send_message()

    async def _send_message(self):
        """Send user message and get response"""
        input_widget = self.query_one("#message-input", Input)
        message = input_widget.value.strip()

        if not message:
            return

        # Clear input
        input_widget.value = ""

        # Add user message to chat
        await self._add_message("user", message)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "topic": self.current_topic
        })

        # Process message and get response
        response = await self._process_message(message)

        # Add agent response to chat
        await self._add_message("agent", response)

        # Add to conversation history
        self.conversation_history.append({
            "role": "agent",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "topic": self.current_topic
        })

    async def _add_message(self, sender: str, content: str):
        """Add message to chat display"""
        chat_widget = self.query_one("#chat-messages", TextArea)

        # Format message
        if sender == "user":
            formatted_message = f"\n👤 You: {content}\n"
        else:
            formatted_message = f"\n🤖 Senter: {content}\n"

        # Append to chat
        current_content = chat_widget.text
        chat_widget.text = current_content + formatted_message

        # Auto-scroll to bottom
        chat_widget.scroll_to_bottom()

    async def _process_message(self, message: str) -> str:
        """Process user message and generate response"""
        try:
            # Use agent selection if available
            if self.selector:
                # Simple agent selection based on message content
                if any(word in message.lower() for word in ["analyze", "understand", "explain"]):
                    agent_id = "ajson://ai-toolbox/agents/analyzer"
                elif any(word in message.lower() for word in ["summarize", "shorten", "brief"]):
                    agent_id = "ajson://ai-toolbox/agents/summarizer"
                elif any(word in message.lower() for word in ["create", "generate", "make"]):
                    agent_id = "ajson://ai-toolbox/agents/creative_writer"
                elif any(word in message.lower() for word in ["search", "find", "research"]):
                    agent_id = "ajson://ai-toolbox/agents/researcher"
                else:
                    agent_id = "ajson://ai-toolbox/agents/senter"

                # For now, return a simple response
                return f"I understand you want to: {message[:50]}... I'll help with that using the {agent_id.split('/')[-1]} agent."

            else:
                # Fallback response
                return f"I received your message: '{message}'. Senter is still initializing its agent system."

        except Exception as e:
            return f"Sorry, I encountered an error processing your message: {str(e)}"

    async def on_mount(self) -> None:
        """Called when the app is mounted"""
        # Set focus to input
        input_widget = self.query_one("#message-input", Input)
        input_widget.focus()

        # Show welcome message
        welcome_msg = """
🎉 Welcome to Senter AI Personal Assistant!

I'm your intelligent companion for:
• 💬 Natural conversation and assistance
• 🎯 Goal tracking and progress monitoring
• 📁 Topic-based organization of your work
• 🤖 Access to specialized AI agents
• 🔄 Continuous learning from our interactions

Type your message below to get started!
        """.strip()

        await self._add_message("agent", welcome_msg)

    async def action_clear_chat(self) -> None:
        """Clear the chat log"""
        chat_widget = self.query_one("#chat-messages", TextArea)
        chat_widget.text = ""

        # Re-add welcome message
        welcome_msg = "🤖 Chat cleared! Welcome back to Senter."
        await self._add_message("agent", welcome_msg)

    async def action_show_topics(self) -> None:
        """Show available topics"""
        topics_dir = Path(__file__).parent.parent / "Topics"
        topics = [d.name for d in topics_dir.iterdir() if d.is_dir()]

        topic_msg = f"📁 Available topics: {', '.join(topics)}\n📁 Current topic: {self.current_topic}"
        await self._add_message("agent", topic_msg)


def main():
    """Launch Senter Chat Interface"""
    app = SenterChat()
    app.run()


if __name__ == "__main__":
    main()