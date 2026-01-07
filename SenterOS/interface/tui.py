#!/usr/bin/env python3
"""
Senter TUI Interface
====================

A simple but powerful terminal user interface for Senter v3.0.
Built with Textual for a modern, responsive experience.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    Header,
    Footer,
    Input,
    Button,
    Static,
    ListView,
    ListItem,
    Label,
)
from textual import events
from pathlib import Path
import sys
import asyncio


class ChatMessage(Static):
    """A chat message display."""

    def __init__(self, text: str, is_user: bool = False):
        super().__init__(text)
        self.is_user = is_user
        self.add_class("user-message" if is_user else "ai-message")


class ChatPanel(Container):
    """Main chat display area."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def compose(self) -> ComposeResult:
        yield Vertical(id="chat-messages")

    def add_message(self, text: str, is_user: bool = False):
        """Add a message to the chat."""
        message = ChatMessage(text, is_user)
        chat_container = self.query_one("#chat-messages", Vertical)
        chat_container.mount(message)
        self.messages.append((text, is_user))

        # Auto-scroll to bottom
        self.app.scroll_to_bottom()


class SenterTUI(App):
    """Senter Terminal User Interface."""

    TITLE = "Senter v3.0"
    SUBTITLE = "Configuration-Driven AI Assistant"

    CSS = """
    ChatPanel {
        height: 1fr;
        overflow-y: auto;
    }
    
    ChatMessage {
        height: auto;
        margin: 1;
        padding: 1;
        border: solid;
    }
    
    .user-message {
        color: white;
        background: $accent-darken-2;
        border-color: $accent;
        align: right;
    }
    
    .ai-message {
        color: white;
        background: $surface-darken-1;
        border-color: $success;
        align: left;
    }
    
    #input-area {
        height: auto;
        dock: bottom;
        margin-top: 1;
    }
    
    #user-input {
        width: 1fr;
    }
    
    #send-button {
        width: auto;
    }
    """

    BINDINGS = [
        ("enter", "send_message", "Send"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        yield ChatPanel()
        
        yield Horizontal(
            Input(placeholder="Type your message...", id="user-input"),
            Button("Send →", id="send-button"),
            id="input-area"
        )
        
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the chat with a welcome message."""
        chat_panel = self.query_one(ChatPanel)
        welcome = """ ███████╗███████╗███╗   ██╗████████╗███████╗██████╗ 
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

👋 Welcome to Senter v3.0!

I'm your symbiotic AI assistant, powered by configuration-driven architecture.
Type anything to begin our conversation..."""
        chat_panel.add_message(welcome, is_user=False)

    def action_send_message(self) -> None:
        """Send the user's message."""
        input_widget = self.query_one("#user-input", Input)
        user_input = input_widget.value.strip()

        if not user_input:
            return

        # Add user message
        chat_panel = self.query_one(ChatPanel)
        chat_panel.add_message(user_input, is_user=True)

        # Clear input
        input_widget.value = ""

        # Process with engine (in a task to avoid blocking)
        self.run_async(self._process_input, user_input)

    async def _process_input(self, user_input: str):
        """Process user input through the engine."""
        try:
            result = self.engine.interact(user_input)
            response = result.get("response", "I couldn't generate a response.")

            chat_panel = self.query_one(ChatPanel)
            chat_panel.add_message(response, is_user=False)

        except Exception as e:
            chat_panel = self.query_one(ChatPanel)
            chat_panel.add_message(f"Error: {str(e)}", is_user=False)

    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        chat_panel = self.query_one(ChatPanel)
        chat_panel.scroll_visible(animate=True)


async def run_tui():
    """Run the TUI interface."""
    from engine.configuration_engine import create_configuration_engine

    engine = create_configuration_engine(
        genome_path=Path("/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"),
        user_id="default"
    )

    app = SenterTUI(engine)
    await app.run_async()


def run_cli():
    """Run the CLI interface (simpler, non-interactive)."""
    from engine.configuration_engine import create_configuration_engine

    engine = create_configuration_engine(
        genome_path=Path("/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"),
        user_id="default"
    )

    print("""
 ███████╗███████╗███╗   ██╗████████╗███████╗██████╗ 
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

Configuration-Driven AI Assistant v3.0
"Configuration is the DNA of an AI system"
""")

    print("\nType 'quit' to exit.\n")

    while True:
        try:
            user_input = input("👤 You: ")

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not user_input.strip():
                continue

            result = engine.interact(user_input)
            response = result.get("response", "I couldn't generate a response.")

            print(f"\n🤖 Senter: {response}\n")

            # Print stats
            stats = engine.get_status()
            print(f"   [Latency: {result.get('latency_ms', 0):.0f}ms | "
                  f"Interactions: {stats['stats']['total_interactions']}]")
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tui":
        asyncio.run(run_tui())
    else:
        run_cli()
