#!/usr/bin/env python3
"""
SenterOS v3.0 - Main Entry Point
=================================

The perfect AI assistant, built on the insight that configuration is DNA.

Usage:
======

    python senteros.py                 # CLI mode
    python senteros.py --tui          # TUI mode
    python senteros.py --test         # Run tests
    python senteros.py --interact     # Interactive REPL
"""

import sys
import asyncio
from pathlib import Path


def print_banner():
    """Print the SenterOS banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     ██████╗ ██████╗ ███████╗ █████╗  ██████╗██╗  ██╗            ║
    ║    ██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔════╝██║  ██║            ║
    ║    ██║     ██║   ██║█████╗  ███████║██║     ███████║            ║
    ║    ██║     ██║   ██║██╔══╝  ██╔══██║██║     ██╔══██║            ║
    ║    ╚██████╗╚██████╔╝███████╗██║  ██║╚██████╗██║  ██║            ║
    ║     ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝            ║
    ║                                                                   ║
    ║              Configuration-Driven AI Assistant v3.0              ║
    ║                                                                   ║
    ║              "Configuration is the DNA of an AI system"          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_cli():
    """Run the CLI interface."""
    from SenterOS.engine.configuration_engine import create_configuration_engine

    engine = create_configuration_engine(
        genome_path=Path("/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"),
        user_id="default",
    )

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

            print(f"\n🤖 SenterOS: {response}\n")

            # Print stats
            stats = engine.get_status()
            print(
                f"   [Latency: {result.get('latency_ms', 0):.0f}ms | "
                f"Interactions: {stats['stats']['total_interactions']}]"
            )
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


async def run_tui():
    """Run the TUI interface."""
    from SenterOS.interface.tui import SenterOSTUI
    from SenterOS.engine.configuration_engine import create_configuration_engine

    engine = create_configuration_engine(
        genome_path=Path("/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"),
        user_id="default",
    )

    app = SenterOSTUI(engine)
    await app.run_async()


def run_interactive():
    """Run an interactive REPL."""
    from SenterOS.engine.configuration_engine import create_configuration_engine

    engine = create_configuration_engine(
        genome_path=Path("/home/sovthpaw/ai-toolbox/Senter/SenterOS/genome"),
        user_id="default",
    )

    print("\n" + "=" * 60)
    print("   SenterOS v3.0 - Interactive REPL")
    print("=" * 60)
    print("\nType 'quit' to exit, 'status' for system status.\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "status":
                status = engine.get_status()
                print(f"\n📊 Status:")
                print(f"   Uptime: {status['uptime_seconds']:.0f}s")
                print(f"   Interactions: {status['stats']['total_interactions']}")
                print(f"   Avg Latency: {status['stats']['average_latency_ms']:.0f}ms")
                print(f"   Genome: {status['genome_version']}")
                print()
                continue

            result = engine.interact(user_input)
            response = result.get("response", "I couldn't generate a response.")

            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def run_tests():
    """Run component tests."""
    print("\n🧪 Running SenterOS v3.0 Component Tests\n")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Genome
    print("Testing Genome...")
    try:
        from SenterOS.genome import Genome, create_default_genome

        genome = create_default_genome()
        assert genome.meta.version == "3.0.0"
        assert genome.system_prompt != ""
        print("   ✅ Genome test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Genome test failed: {e}")
        tests_failed += 1

    # Test 2: Knowledge Graph
    print("Testing Knowledge Graph...")
    try:
        from SenterOS.knowledge import KnowledgeGraph, create_default_knowledge_graph

        kg = create_default_knowledge_graph()
        assert kg is not None
        print("   ✅ Knowledge Graph test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Knowledge Graph test failed: {e}")
        tests_failed += 1

    # Test 3: Living Memory
    print("Testing Living Memory...")
    try:
        from SenterOS.memory import LivingMemory, create_default_memory

        memory = create_default_memory()
        assert memory is not None
        print("   ✅ Living Memory test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Living Memory test failed: {e}")
        tests_failed += 1

    # Test 4: Evolution Engine
    print("Testing Evolution Engine...")
    try:
        from SenterOS.evolution import EvolutionEngine, create_default_evolution_engine

        evolution = create_default_evolution_engine()
        assert evolution is not None
        print("   ✅ Evolution Engine test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Evolution Engine test failed: {e}")
        tests_failed += 1

    # Test 5: Capability Registry
    print("Testing Capability Registry...")
    try:
        from SenterOS.capabilities import (
            CapabilityRegistry,
            create_default_capability_registry,
        )

        registry = create_default_capability_registry()
        assert registry is not None
        print("   ✅ Capability Registry test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Capability Registry test failed: {e}")
        tests_failed += 1

    # Test 6: Configuration Engine
    print("Testing Configuration Engine...")
    try:
        from SenterOS.engine import create_configuration_engine

        engine = create_configuration_engine()
        assert engine is not None
        print("   ✅ Configuration Engine test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Configuration Engine test failed: {e}")
        tests_failed += 1

    # Test 7: Interaction
    print("Testing Interaction...")
    try:
        from SenterOS.engine import create_configuration_engine

        engine = create_configuration_engine()
        result = engine.interact("Hello, SenterOS!")
        assert result is not None
        assert "response" in result
        print("   ✅ Interaction test passed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Interaction test failed: {e}")
        tests_failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"   Tests Passed: {tests_passed}")
    print(f"   Tests Failed: {tests_failed}")
    print(f"{'=' * 60}\n")

    return tests_failed == 0


def main():
    """Main entry point."""
    print_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "--tui":
            print("🚀 Starting TUI mode...\n")
            asyncio.run(run_tui())
        elif command == "--test":
            success = run_tests()
            sys.exit(0 if success else 1)
        elif command == "--interact":
            print("🚀 Starting Interactive REPL...\n")
            run_interactive()
        elif command == "--help":
            print("""
SenterOS v3.0 - Configuration-Driven AI Assistant

Usage:
    python senteros.py                 # CLI mode (default)
    python senteros.py --tui          # TUI mode (requires textual)
    python senteros.py --interact     # Interactive REPL
    python senteros.py --test         # Run component tests
    python senteros.py --help         # Show this help

The perfect AI assistant, built on the insight that configuration is DNA.
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information.")
            sys.exit(1)
    else:
        print("🚀 Starting SenterOS v3.0...\n")
        run_cli()


if __name__ == "__main__":
    main()
