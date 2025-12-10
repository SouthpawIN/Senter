#!/usr/bin/env python3
"""
Senter TUI - Advanced Terminal User Interface
Modular sidebar with interactive lists for Topics, Goals, Tasks, To-Do, and Calendar
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
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Static, Input, Button, TextArea,
    ListView, ListItem, Label, Header, Footer, Collapsible, Switch, TabbedContent, TabPane
)
from textual.containers import ScrollableContainer
from textual import events
from textual.binding import Binding

# Import Senter components
try:
    from senter_selector import SenterSelector
    from background_processor import get_background_manager
    from senter_widgets import SidebarContainer
    from model_server_manager import ModelServerManager
except ImportError as e:
    print(f"Warning: Could not import Senter components: {e}")
    SenterSelector = None
    get_background_manager = None
    SidebarContainer = None
    ModelServerManager = None

def create_gradient_ascii():
    """Create ASCII art with diagonal gradient from dark green to mint"""
    lines = [
        "███████╗███████╗███╗   ██╗████████╗███████╗██████╗",
        "██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗",
        "███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝",
        "╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗",
        "███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║",
        "╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝"
    ]

    start_color = (0x00, 0x80, 0x80)  # #008080 teal
    end_color = (0x00, 0xff, 0xaa)    # #00ffaa mint

    max_row = len(lines) - 1
    max_col = max(len(line) for line in lines) - 1

    # Pad all lines to the same width for proper centering
    max_width = max(len(line) for line in lines)
    padded_lines = [line.ljust(max_width) for line in lines]

    result = []
    for row, line in enumerate(padded_lines):
        colored_line = []
        for col, char in enumerate(line):
            # Diagonal gradient: average of row and column progress
            t = (row / max_row + col / max_col) / 2
            r = int(start_color[0] + t * (end_color[0] - start_color[0]))
            g = int(start_color[1] + t * (end_color[1] - start_color[1]))
            b = int(start_color[2] + t * (end_color[2] - start_color[2]))
            hex_color = f"{r:02x}{g:02x}{b:02x}"
            colored_line.append(f"[#{hex_color}]{char}[/#{hex_color}]")
        result.append("".join(colored_line))
    return "\n".join(result)

# ASCII Art Title with diagonal gradient
ASCII_TITLE = create_gradient_ascii()

class ChatArea(ScrollableContainer):
    """Main chat display area"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[str] = []
        self.show_ascii = True

    def compose(self):
        yield Static(ASCII_TITLE, id="chat-content", markup=True)

    def add_message(self, sender: str, message: str):
        """Add a message to the chat"""
        if sender == "user":
            formatted = f"You: {message}"
        else:
            formatted = f"Senter: {message}"

        self.messages.append(formatted)
        self.show_ascii = False
        # Keep last 50 messages
        display_messages = "\n\n".join(self.messages[-50:])
        content = self.query_one("#chat-content", Static)
        content.update(display_messages)
        # Scroll to bottom
        self.scroll_end()

    def clear_chat(self):
        """Clear all messages"""
        self.messages = []
        self.show_ascii = True
        content = self.query_one("#chat-content", Static)
        content.update(ASCII_TITLE)


class InputBar(Static):
    """Input bar that matches chat area width"""

    def compose(self) -> ComposeResult:
        with Container(id="input-container"):
            yield Input(placeholder="Type your message or /command...", id="message-input")
            yield Button("Send", id="send-button", variant="primary")


class SenterApp(App):
    """Advanced Senter TUI with modular sidebar"""

    CSS_PATH = "../senter.tcss"

    # Reactive state
    active_sidebar = "topics"
    sidebar_notifications = {"goals": 0, "tasks": 0, "todo": 0}

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "delete_tab", "Delete Tab"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+t", "toggle_sidebar", "Toggle Sidebar"),
        Binding("f1", "show_topics", "Show Topics"),
        Binding("f2", "show_goals", "Show Goals"),
        Binding("f3", "show_tasks", "Show Tasks"),
        Binding("f4", "show_todo", "Show To-Do"),
        Binding("f5", "show_calendar", "Show Calendar"),
    ]

    def __init__(self):
        super().__init__()
        self.selector = SenterSelector() if SenterSelector else None
        self.background_manager = get_background_manager() if get_background_manager else None
        self.model_manager = ModelServerManager() if ModelServerManager else None

        # Data stores
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_topic = "general"

        # UI state
        self.sidebar_visible = True
        self.model_switches: Dict[str, Switch] = {}
        self.switch_to_model: Dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Compose the main UI layout"""
        with TabbedContent(id="main-tabs"):
            with TabPane("Senter", id="main-tab"):
                with Horizontal():
                    # Left panel: content area (2/3 width)
                    with Vertical(id="left-panel"):
                        yield ChatArea(id="chat-area")
                        yield InputBar(id="input-bar")
                        # Model info bar below input
                        with Horizontal(id="snack-bar"):
                            yield Static("", id="model-info", classes="snack-bar-left")
                            yield Static('type "/" to see commands', id="command-hint", classes="snack-bar-right")

                    # Right panel: sidebar (1/3 width, full height)
                    if self.sidebar_visible:
                        with Vertical(id="sidebar"):
                            # Expandable sections
                            with Collapsible(title="Goals", collapsed=True):
                                yield Static("• Active goal 1\n• Active goal 2", classes="collapsible-content")
                            with Collapsible(title="Tasks", collapsed=True):
                                yield Static("• Proposed task 1", classes="collapsible-content")
                            with Collapsible(title="To-Do", collapsed=True):
                                yield Static("• Pending item 1", classes="collapsible-content")
                            with Collapsible(title="Calendar", collapsed=True):
                                yield Static("• Today: No events\n• Tomorrow: Meeting", classes="collapsible-content")
                            # Models section
                            with Collapsible(title="Models", collapsed=True):
                                if self.model_manager:
                                    models_dir = Path(__file__).parent.parent / "Models"
                                    config_file = models_dir.parent / "config" / "senter_config.json"
                                    default_model = "Qwen2.5-Omni-3B"  # fallback
                                    try:
                                        with open(config_file, 'r') as f:
                                            config = json.load(f)
                                            default_model = config.get("default_model", default_model)
                                    except:
                                        pass

                                    if models_dir.exists():
                                        for model_file in models_dir.glob("*.gguf"):
                                            model_name = model_file.name
                                            # Sanitize model name for ID (replace invalid chars with underscores)
                                            safe_id = model_name.replace(".", "_").replace("-", "_")
                                            is_default = model_name.startswith(default_model.split("/")[-1]) or default_model in model_name
                                            item_classes = "model-item" + (" default-model" if is_default else "")
                                            with Vertical(classes=item_classes):
                                                yield Static(model_name, classes="model-name")
                                                switch = Switch(value=False, id=f"switch-{safe_id}")
                                                self.model_switches[model_name] = switch
                                                if switch.id:
                                                    self.switch_to_model[switch.id] = model_name
                                                yield switch
                            # Sidebar footer
                            with Horizontal(id="sidebar-footer"):
                                yield Static("v 0.1", id="sidebar-topic")
                                yield Static("", id="sidebar-datetime")  # Will be updated with datetime
            with TabPane("➕", id="plus-tab"):
                with Horizontal():
                    # Left panel: content area (2/3 width)
                    with Vertical(id="left-panel-plus"):
                        yield ChatArea(id="chat-area-plus")
                        yield InputBar(id="input-bar-plus")
                        # Model info bar below input
                        with Horizontal(id="snack-bar-plus"):
                            yield Static("", id="model-info-plus", classes="snack-bar-left")
                            yield Static('type "/" to see commands', id="command-hint-plus", classes="snack-bar-right")

                    # Right panel: sidebar (1/3 width, full height)
                    if self.sidebar_visible:
                        with Vertical(id="sidebar-plus"):
                            # Expandable sections
                            with Collapsible(title="Goals", collapsed=True):
                                yield Static("• Active goal 1\n• Active goal 2", classes="collapsible-content")
                            with Collapsible(title="Tasks", collapsed=True):
                                yield Static("• Proposed task 1", classes="collapsible-content")
                            with Collapsible(title="To-Do", collapsed=True):
                                yield Static("• Pending item 1", classes="collapsible-content")
                            with Collapsible(title="Calendar", collapsed=True):
                                yield Static("• Today: No events\n• Tomorrow: Meeting", classes="collapsible-content")
                            # Models section
                            with Collapsible(title="Models", collapsed=True):
                                if self.model_manager:
                                    models_dir = Path(__file__).parent.parent / "Models"
                                    config_file = models_dir.parent / "config" / "senter_config.json"
                                    default_model = "Qwen2.5-Omni-3B"  # fallback
                                    try:
                                        with open(config_file, 'r') as f:
                                            config = json.load(f)
                                            default_model = config.get("default_model", default_model)
                                    except:
                                        pass

                                    if models_dir.exists():
                                        for model_file in models_dir.glob("*.gguf"):
                                            model_name = model_file.name
                                            # Sanitize model name for ID (replace invalid chars with underscores)
                                            safe_id = model_name.replace(".", "_").replace("-", "_")
                                            is_default = model_name.startswith(default_model.split("/")[-1]) or default_model in model_name
                                            item_classes = "model-item" + (" default-model" if is_default else "")
                                            with Vertical(classes=item_classes):
                                                yield Static(model_name, classes="model-name")
                                                switch = Switch(value=False, id=f"switch-plus-{safe_id}")
                                                self.model_switches[model_name] = switch
                                                self.switch_to_model[switch.id or ""] = model_name
                            # Sidebar footer
                            with Horizontal(id="sidebar-footer-plus"):
                                yield Static("v 0.1", id="sidebar-topic-plus")
                                yield Static("", id="sidebar-datetime-plus")  # Will be updated with datetime

    async def on_mount(self) -> None:
        """Initialize the app"""
        # Focus input
        try:
            input_widget = self.query_one("#message-input", Input)
            input_widget.focus()
        except:
            pass

        # Update datetime and model info
        self.update_sidebar_info()
        # Update datetime every minute
        self.set_interval(60, self.update_sidebar_info)

    def update_sidebar_info(self):
        """Update sidebar datetime and model info"""
        # Update datetime
        try:
            datetime_widget = self.query_one("#sidebar-datetime", Static)
            now = datetime.now().strftime("%H:%M")
            datetime_widget.update(now)
        except:
            pass

        # Update model info
        try:
            model_info = self.query_one("#model-info", Static)
            default_model = "Qwen2.5-Omni-3B"  # This should come from config
            model_text = f"[#00ffaa]Model:[/#00ffaa] {default_model}"
            model_info.update(model_text)
        except:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id and event.input.id.startswith("message-input"):
            await self._handle_message(event.value)

    async def on_button_pressed(self, event):
        """Handle button presses"""
        if event.button.id and event.button.id.startswith("send-button"):
            # Find the corresponding input
            tab_suffix = event.button.id.replace("send-button", "")
            input_id = f"message-input{tab_suffix}"
            input_widget = self.query_one(f"#{input_id}", Input)
            message = input_widget.value.strip()
            if message:
                await self._handle_message(message)

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle model server toggle switches"""
        if not self.model_manager:
            return

        switch_id = event.switch.id
        if switch_id:
            model_name = self.switch_to_model.get(switch_id, switch_id.replace("switch-", "").replace("_", "."))
            is_on = event.value

            if is_on:
                # Start server
                success = await self._start_model_server(model_name)
                if not success:
                    # Revert switch if failed
                    event.switch.value = False
            else:
                # Stop server
                await self._stop_model_server(model_name)

    async def on_tab_activated(self, event):
        """Handle tab activation"""
        if event.tab.id == "add-tab":
            # Add a new tab
            tabs = self.query_one("#main-tabs", TabbedContent)
            tab_count = len(tabs.query("TabPane")) - 1  # Subtract the add tab
            new_tab_id = f"tab-{tab_count + 1}"
            new_tab_title = f"Senter {tab_count + 1}"

            # Create new tab content (simplified for now)
            new_pane = TabPane(new_tab_title, id=new_tab_id)
            with new_pane:
                yield Static(f"New Senter instance #{tab_count + 1}", classes="new-tab-content")

            # Add the new tab before the add tab
            tabs.mount(new_pane, before="#add-tab")

            # Switch to the new tab
            tabs.active = new_tab_id

    async def _start_model_server(self, model_name: str) -> bool:
        """Start a model server in background"""
        if not self.model_manager:
            return False
        # Run in thread to avoid blocking UI
        import asyncio
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, self.model_manager.start_model_server, model_name, "text", 8000)
        return success

    async def _stop_model_server(self, model_name: str) -> bool:
        """Stop a model server"""
        if not self.model_manager:
            return False
        # Run in thread to avoid blocking UI
        import asyncio
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, self.model_manager.stop_model_server, "text")
        return success

    async def _add_new_tab(self):
        """Add a new + tab"""
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Generate unique ID
        existing_tabs = len(tabs.query("TabPane"))
        new_id = f"tab-{existing_tabs}"

        # Add new + tab
        new_pane = TabPane("➕", id=new_id)
        with new_pane:
            with Horizontal():
                # Left panel: content area (2/3 width)
                with Vertical(id=f"left-panel-{new_id}"):
                    yield ChatArea(id=f"chat-area-{new_id}")
                    yield InputBar(id=f"input-bar-{new_id}")
                    # Model info bar below input
                    yield Static("", id=f"snack-bar-{new_id}", classes="snack-bar")

                # Right panel: sidebar (1/3 width, full height)
                if self.sidebar_visible:
                    with Vertical(id=f"sidebar-{new_id}"):
                        # Expandable sections
                        with Collapsible(title="Goals", collapsed=True):
                            yield Static("• Active goal 1\n• Active goal 2", classes="collapsible-content")
                        with Collapsible(title="Tasks", collapsed=True):
                            yield Static("• Proposed task 1", classes="collapsible-content")
                        with Collapsible(title="To-Do", collapsed=True):
                            yield Static("• Pending item 1", classes="collapsible-content")
                        with Collapsible(title="Calendar", collapsed=True):
                            yield Static("• Today: No events\n• Tomorrow: Meeting", classes="collapsible-content")
                        # Models section
                        with Collapsible(title="Models", collapsed=True):
                            if self.model_manager:
                                models_dir = Path(__file__).parent.parent / "Models"
                                if models_dir.exists():
                                    for model_file in models_dir.glob("*.gguf"):
                                        model_name = model_file.name
                                        safe_id = model_name.replace(".", "_").replace("-", "_")
                                        with Vertical(classes="model-item"):
                                            yield Static(model_name, classes="model-name")
                                            switch = Switch(value=False, id=f"switch-{new_id}-{safe_id}")
                                            yield switch
                        # Sidebar footer
                        with Horizontal(id=f"sidebar-footer-{new_id}"):
                            yield Static("v 0.1", id=f"sidebar-topic-{new_id}")
                            yield Static("", id=f"sidebar-datetime-{new_id}")

        tabs.mount(new_pane)

    async def _handle_message(self, message: str):
        """Process user message"""
        if not message.strip():
            return

        # Initialize variables
        is_plus_tab = False
        input_id = "#message-input"

        # Clear input
        try:
            input_widget = self.query_one("#message-input", Input)
            input_widget.value = ""
        except:
            pass

        # Handle slash commands
        if message.startswith("/"):
            response = self._handle_slash_command(message)
        else:
            # Check if this is in the plus tab
            tabs = self.query_one("#main-tabs", TabbedContent)
            active_tab = tabs.active_pane
            is_plus_tab = active_tab and active_tab.id == "plus-tab"

            # Process as regular message
            response = await self._process_chat_message(message)

            # If in plus tab, add new plus tab after message
            if is_plus_tab:
                self._add_new_plus_tab()

        # Add to conversation
        try:
            if is_plus_tab:
                chat_area = self.query_one("#chat-area-plus", ChatArea)
                input_id = "#message-input-plus"
            else:
                chat_area = self.query_one("#chat-area", ChatArea)
                input_id = "#message-input"
            chat_area.add_message("user", message)
            chat_area.add_message("agent", response)
        except:
            pass

        # Clear input
        try:
            input_widget = self.query_one(input_id, Input)
            input_widget.value = ""
        except:
            pass

    def _handle_slash_command(self, command: str) -> str:
        """Handle slash commands"""
        parts = command.split()
        cmd = parts[0].lower()

        commands = {
            "/topics": lambda: f"Switched to Topics sidebar (F1)",
            "/goals": lambda: f"Switched to Goals sidebar (F2)",
            "/tasks": lambda: f"Switched to Tasks sidebar (F3)",
            "/todo": lambda: f"Switched to To-Do sidebar (F4)",
            "/calendar": lambda: f"Switched to Calendar sidebar (F5)",
            "/hide": lambda: f"Sidebar {'hidden' if self.sidebar_visible else 'shown'}",
            "/clear": lambda: "Notifications cleared",
            "/help": lambda: """Available commands:
/topics, /goals, /tasks, /todo, /calendar - Switch sidebar
/hide - Toggle sidebar
/clear - Clear notifications
/help - Show this help

Keyboard shortcuts:
F1-F5 - Switch sidebar modules
Ctrl+L - Clear chat
Ctrl+T - Toggle sidebar
Ctrl+C - Quit""",
        }

        if cmd in commands:
            if cmd == "/hide":
                self.sidebar_visible = not self.sidebar_visible
                self.refresh()
            return commands[cmd]()
        else:
            return f"Unknown command: {cmd}. Type /help for available commands."

    async def _process_chat_message(self, message: str) -> str:
        """Process regular chat message"""
        # Simple response logic (would integrate with agent system)
        message_lower = message.lower()

        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm Senter, your AI personal assistant. How can I help you today?"

        elif "status" in message_lower:
            return "✅ All systems operational! The modular sidebar system is active."

        elif "goal" in message_lower:
            return "🎯 You have active goals! Check the sidebar with /goals to see them."

        elif "task" in message_lower:
            return "✅ You have AI-proposed tasks! Use /tasks to see them."

        else:
            return f"I understand you said: '{message}'. The sidebar shows your Topics, Goals, Tasks, To-Do items, and Calendar. Use /help for commands!"

    # Sidebar switching actions
    def action_show_topics(self):
        self.active_sidebar = "topics"
        self._update_sidebar_display()

    def action_show_goals(self):
        self.active_sidebar = "goals"
        self._update_sidebar_display()

    def action_show_tasks(self):
        self.active_sidebar = "tasks"
        self._update_sidebar_display()

    def action_show_todo(self):
        self.active_sidebar = "todo"
        self._update_sidebar_display()

    def action_show_calendar(self):
        self.active_sidebar = "calendar"
        self._update_sidebar_display()

    def action_toggle_sidebar(self):
        self.sidebar_visible = not self.sidebar_visible
        self.refresh()

    def _update_sidebar_display(self):
        """Update sidebar display based on active module"""
        # This would update the sidebar content - simplified for now
        pass

    async def action_clear_chat(self):
        """Clear the chat area"""
        try:
            chat_area = self.query_one("#chat-area", ChatArea)
            chat_area.clear_chat()
        except:
            pass

    async def action_delete_tab(self):
        """Delete the current tab or prompt to exit if last tab"""
        tabs = self.query_one("#main-tabs", TabbedContent)
        active_pane = tabs.active_pane
        all_panes = tabs.query("TabPane")

        if len(all_panes) > 1 and active_pane:
            # Remove the active tab
            active_pane.remove()
            # Switch to the last tab
            remaining = tabs.query("TabPane")
            if remaining:
                tabs.active = remaining[-1].id or ""
        else:
            # Last tab, prompt to exit
            self._show_exit_prompt()

    def _show_exit_prompt(self):
        """Show exit prompt that disappears after 2 seconds"""
        # For now, just quit immediately. To add prompt, could add a temporary widget.
        self.exit()

    def _determine_topic(self, message: str) -> str:
        """Determine the most relevant topic for the message"""
        # Simple heuristic for now
        message_lower = message.lower()
        if "code" in message_lower or "programming" in message_lower:
            return "Coding"
        elif "music" in message_lower or "song" in message_lower:
            return "Music"
        elif "image" in message_lower or "picture" in message_lower:
            return "Images"
        elif "goal" in message_lower or "task" in message_lower:
            return "Goals"
        else:
            return "General"

    def _add_new_plus_tab(self):
        """Add a new plus tab"""
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Generate unique ID
        existing_tabs = len(tabs.query("TabPane"))
        new_id = f"plus-tab-{existing_tabs}"

        # Add new + tab
        new_pane = TabPane("➕", id=new_id)
        with new_pane:
            with Horizontal():
                # Left panel: content area (2/3 width)
                with Vertical(id=f"left-panel-{new_id}"):
                    yield ChatArea(id=f"chat-area-{new_id}")
                    yield InputBar(id=f"input-bar-{new_id}")
                    # Model info bar below input
                    yield Static("", id=f"snack-bar-{new_id}", classes="snack-bar")

                # Right panel: sidebar (1/3 width, full height)
                if self.sidebar_visible:
                    with Vertical(id=f"sidebar-{new_id}"):
                        # Expandable sections
                        with Collapsible(title="Goals", collapsed=True):
                            yield Static("• Active goal 1\n• Active goal 2", classes="collapsible-content")
                        with Collapsible(title="Tasks", collapsed=True):
                            yield Static("• Proposed task 1", classes="collapsible-content")
                        with Collapsible(title="To-Do", collapsed=True):
                            yield Static("• Pending item 1", classes="collapsible-content")
                        with Collapsible(title="Calendar", collapsed=True):
                            yield Static("• Today: No events\n• Tomorrow: Meeting", classes="collapsible-content")
                        # Models section
                        with Collapsible(title="Models", collapsed=True):
                            if self.model_manager:
                                models_dir = Path(__file__).parent.parent / "Models"
                                if models_dir.exists():
                                    for model_file in models_dir.glob("*.gguf"):
                                        model_name = model_file.name
                                        safe_id = model_name.replace(".", "_").replace("-", "_")
                                        with Vertical(classes="model-item"):
                                            yield Static(model_name, classes="model-name")
                                            switch = Switch(value=False, id=f"switch-{new_id}-{safe_id}")
                                            yield switch
                        # Sidebar footer
                        with Horizontal(id=f"sidebar-footer-{new_id}"):
                            yield Static("v 0.1", id=f"sidebar-topic-{new_id}")
                            yield Static("", id=f"sidebar-datetime-{new_id}")

        tabs.mount(new_pane)


def main():
    """Launch Senter TUI"""
    app = SenterApp()
    app.run()


if __name__ == "__main__":
    main()