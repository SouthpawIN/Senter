#!/usr/bin/env python3
"""
Senter - Universal AI Personal Assistant (Async Chain Version)
Everything is omniagent with SENTER.md configs
"""

import asyncio
import sys
from pathlib import Path

# Import Senter utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent))

from Functions.omniagent_chain import OmniAgentChain


async def main_async():
    """Async main function"""
    print("=" * 60)
    print("🚀 SENTER v2.0 - Async OmniAgent Chain")
    print("=" * 60)

    # Initialize chain
    senter_root = Path(__file__).parent.parent
    chain = OmniAgentChain(senter_root)
    await chain.initialize()

    # Run background discovery (blocking for now, could be background task)
    await chain.run_background_tasks()

    # Show available Focuses
    print("\n📁 Available Focuses:")
    for focus in chain.list_user_focuses():
        print(f"   - {focus}")

    # Interactive loop
    print("\n✅ Senter is ready!")
    print("\n📝 Commands:")
    print("  /list         - List all Focuses")
    print("  /focus <name> - Set Focus")
    print("  /goals        - Show goals for current Focus")
    print("  /discover      - Run tool discovery")
    print("  /exit         - Exit\n")

    current_focus = None

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/list":
                    print("\n📁 Available Focuses:")
                    for focus in chain.list_user_focuses():
                        print(f"   - {focus}")
                elif user_input.startswith("/focus "):
                    current_focus = user_input.split(" ", 1)[1]
                    print(f"\n🎯 Focus set to: {current_focus}")
                elif user_input == "/goals" and current_focus:
                    print(f"\n🎯 Goals for {current_focus}:")
                    senter_file = senter_root / "Focuses" / current_focus / "SENTER.md"
                    if senter_file.exists():
                        with open(senter_file) as f:
                            content = f.read()
                            if "Goals & Objectives" in content:
                                start = content.find("Goals & Objectives")
                                print(f"   {content[start : start + 200]}...")
                    else:
                        print("   No goals yet for this Focus")
                elif user_input == "/discover":
                    print("\n🔍 Running tool discovery...")
                    await chain.discover_tools()
                else:
                    print("⚠️  Unknown command")
                continue

            # Regular query
            response = await chain.process_query(
                user_input,
                context="",  # Could load from history
                focus_hint=current_focus,
            )
            print(f"\nSenter: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

    # Cleanup
    await chain.close()


def main():
    """Sync entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
