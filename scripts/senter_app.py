#!/usr/bin/env python3
"""
Senter TUI - Advanced Terminal User Interface
LAZY LOADING - Models load only when needed
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Static,
    Input,
    Button,
    TextArea,
    ListView,
    ListItem,
    Label,
    Header,
    Footer,
)
from textual.containers import ScrollableContainer
from textual import events
from textual.binding import Binding

# Focus system (lightweight imports only)
from Focuses.senter_md_parser import SenterMdParser

# SenterOmniAgent will be loaded lazily when needed
omni_agent = None


def create_gradient_ascii():
    lines = [
        "Sent - Universal AI Assistant",
        "",
    ]
    return "\n".join(lines)


class SenterApp(App):
    """Senter TUI Application"""

    CSS = """
    Screen {
        background: #0a0f0a;
    }
    .sidebar {
        background: #1a1a1a;
        width: 30;
    }
    .chat-panel {
        background: #0a0f0a;
    }
    """

    def __init__(self):
        super().__init__()
        self.senter_root = Path(__file__).parent.parent
        self.parser = SenterMdParser(self.senter_root)
        self.current_focus = "general"

    def compose(self) -> ComposeResult:
        return Vertical(
            Header(),
            Horizontal(
                Static(self.sidebar_content(), id="sidebar"),
                Container(Static(self.chat_content(), id="chat"), id="main"),
            ),
            Footer(),
        )

    def sidebar_content(self) -> str:
        return f"""FOCUSES
{'-' * 20}
"""

    def chat_content(self) -> str:
        return f"""[bold green]Senter v2.0[/bold green] - Focus-Based AI Assistant
{'=' * 50}

[italic]Ready for your input...[/italic]
{'=' * 50}
"""

    def action_quit(self):
        self.exit()


if __name__ == "__main__":
    app = SenterApp()
    app.run()
