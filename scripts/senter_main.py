#!/usr/bin/env python3
"""
Senter Main Orchestrator
Coordinates all agents and manages the Senter AI Personal Assistant system
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import Senter components
from background_processor import get_background_manager
from model_server_manager import ModelServerManager

class SenterOrchestrator:
    """Main orchestrator for the Senter AI Personal Assistant"""

    def __init__(self):
        self.senter_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.user_profile = self._load_user_profile()

        # Initialize components
        self.background_manager = get_background_manager()
        self.model_manager = ModelServerManager()

        # System state
        self.initialized = False
        self.conversation_history: List[Dict[str, Any]] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load Senter configuration"""
        config_file = self.senter_root / "config" / "senter_config.json"
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return {"parallel_processing": True}

    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile"""
        profile_file = self.senter_root / "config" / "user_profile.json"
        try:
            with open(profile_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"goals": [], "preferences": {}}

    def initialize_system(self) -> bool:
        """Initialize the complete Senter system"""
        print("🚀 Initializing Senter AI Personal Assistant...")

        try:
            # Step 1: Start background processing
            print("1️⃣ Starting background processing...")
            self.background_manager.start_background_processing()

            # Step 2: Start model servers
            print("2️⃣ Starting model servers...")
            if not self.model_manager.start_all_servers():
                print("⚠️ Some model servers failed to start")

            # Step 3: Verify system health
            print("3️⃣ Verifying system health...")
            health_status = self.check_system_health()
            if not health_status["overall_healthy"]:
                print("⚠️ System health issues detected")
                for issue in health_status["issues"]:
                    print(f"   - {issue}")

            # Step 4: Load conversation history
            print("4️⃣ Loading conversation history...")
            self._load_conversation_history()

            self.initialized = True
            print("✅ Senter initialization complete!")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health = {
            "overall_healthy": True,
            "issues": []
        }

        # Check background processing
        bg_status = self.background_manager.get_status()
        if not bg_status["running"]:
            health["issues"].append("Background processing not running")
            health["overall_healthy"] = False

        # Check model servers
        model_status = self.model_manager.get_server_status()
        for model_type, status in model_status.items():
            if not status["healthy"]:
                health["issues"].append(f"{model_type} model server not healthy")
                health["overall_healthy"] = False

        # Check required directories
        required_dirs = ["Agents", "Functions", "Topics", "Models", "config", "scripts"]
        for dir_name in required_dirs:
            if not (self.senter_root / dir_name).exists():
                health["issues"].append(f"Required directory missing: {dir_name}")
                health["overall_healthy"] = False

        return health

    def _load_conversation_history(self):
        """Load conversation history from file"""
        history_file = self.senter_root / "conversation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.conversation_history = json.load(f)
                print(f"📚 Loaded {len(self.conversation_history)} conversation entries")
            except Exception as e:
                print(f"Warning: Could not load conversation history: {e}")
                self.conversation_history = []

    def _save_conversation_history(self):
        """Save conversation history to file"""
        history_file = self.senter_root / "conversation_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process a user message and return response"""
        if not self.initialized:
            return "❌ Senter is not fully initialized. Please run initialization first."

        # Add message to history
        message_entry = {
            "role": "user",
            "content": message,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.conversation_history.append(message_entry)

        # Simple response logic (would be replaced with agent orchestration)
        response = self._generate_response(message, context)

        # Add response to history
        response_entry = {
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.conversation_history.append(response_entry)

        # Save history periodically
        if len(self.conversation_history) % 10 == 0:
            self._save_conversation_history()

        return response

    def _generate_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate a response to user message"""
        # This is a placeholder - would use agent selection and orchestration
        message_lower = message.lower()

        # Simple keyword-based responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Senter, your AI personal assistant. How can I help you today?"

        elif any(word in message_lower for word in ["status", "health", "check"]):
            health = self.check_system_health()
            if health["overall_healthy"]:
                return "✅ All systems operational! Background processing active, model servers running."
            else:
                issues = "\\n".join(f"• {issue}" for issue in health["issues"])
                return f"⚠️ System health issues detected:\\n{issues}"

        elif any(word in message_lower for word in ["analyze", "understand", "explain"]):
            return "🤔 I'll analyze that for you using my analyzer agent. This would provide deep insights into the content."

        elif any(word in message_lower for word in ["summarize", "summary", "brief"]):
            return "📝 I'll create a summary for you using my summarizer agent. This will condense the information effectively."

        elif any(word in message_lower for word in ["create", "generate", "make"]):
            return "🎨 I'll help you create that using my creative writer agent. What would you like to generate?"

        elif any(word in message_lower for word in ["search", "find", "research"]):
            return "🔍 I'll research that for you using my researcher agent with web search capabilities."

        elif any(word in message_lower for word in ["goal", "plan", "objective"]):
            goals = self.user_profile.get("goals", [])
            if goals:
                goal_list = "\\n".join(f"• {goal}" for goal in goals[:3])
                return f"🎯 Your current goals:\\n{goal_list}\\n\\nHow can I help you work towards these?"
            else:
                return "🎯 You haven't set any goals yet. Would you like me to help you identify and track some goals?"

        else:
            return f"I understand you said: '{message}'. I'm still learning how to best assist you. My background systems are analyzing this interaction to improve my responses!"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "initialized": self.initialized,
            "background_processing": self.background_manager.get_status(),
            "model_servers": self.model_manager.get_server_status(),
            "conversation_count": len(self.conversation_history),
            "user_goals": len(self.user_profile.get("goals", [])),
            "system_health": self.check_system_health()
        }

    def shutdown(self):
        """Gracefully shutdown the system"""
        print("🛑 Shutting down Senter...")

        # Stop background processing
        self.background_manager.stop_background_processing()

        # Stop model servers
        self.model_manager.stop_all_servers()

        # Save conversation history
        self._save_conversation_history()

        print("✅ Senter shutdown complete")


def main():
    """Main entry point for Senter"""
    import argparse

    parser = argparse.ArgumentParser(description="Senter AI Personal Assistant")
    parser.add_argument("command", choices=[
        "init", "chat", "status", "shutdown", "test"
    ])
    parser.add_argument("--message", help="Message for chat command")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")

    args = parser.parse_args()

    orchestrator = SenterOrchestrator()

    if args.command == "init":
        if orchestrator.initialize_system():
            print("\\n🎉 Senter is ready! Run 'python scripts/senter_main.py chat --interactive' to start chatting.")
        else:
            print("\\n❌ Senter initialization failed.")
            sys.exit(1)

    elif args.command == "chat":
        if not orchestrator.initialized:
            print("❌ Senter not initialized. Run 'python scripts/senter_main.py init' first.")
            sys.exit(1)

        if args.interactive:
            print("🤖 Senter Interactive Chat (Ctrl+C to exit)")
            print("=" * 50)

            try:
                while True:
                    message = input("\\n👤 You: ").strip()
                    if message:
                        response = orchestrator.process_message(message)
                        print(f"🤖 Senter: {response}")
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                orchestrator.shutdown()

        elif args.message:
            response = orchestrator.process_message(args.message)
            print(f"🤖 Senter: {response}")

        else:
            print("❌ Use --message 'your message' or --interactive")

    elif args.command == "status":
        status = orchestrator.get_system_status()
        print("🔍 Senter System Status")
        print("=" * 30)
        print(f"Initialized: {'✅' if status['initialized'] else '❌'}")
        print(f"Background Workers: {status['background_processing']['workers']}")
        print(f"Conversation Count: {status['conversation_count']}")
        print(f"User Goals: {status['user_goals']}")

        print("\\n🧠 Model Servers:")
        for model_type, model_status in status['model_servers'].items():
            health_icon = "🟢" if model_status["healthy"] else "🔴"
            print(f"  {health_icon} {model_type}: {model_status['status']}")

        health = status['system_health']
        if health['overall_healthy']:
            print("\\n✅ System Health: Good")
        else:
            print("\\n⚠️ System Health Issues:")
            for issue in health['issues']:
                print(f"  - {issue}")

    elif args.command == "shutdown":
        orchestrator.shutdown()

    elif args.command == "test":
        # Quick test of system components
        print("🧪 Running Senter system tests...")

        # Test configuration loading
        config_test = len(orchestrator.config) > 0
        print(f"Config Loading: {'✅' if config_test else '❌'}")

        # Test background manager
        bg_test = orchestrator.background_manager is not None
        print(f"Background Manager: {'✅' if bg_test else '❌'}")

        # Test model manager
        model_test = orchestrator.model_manager is not None
        print(f"Model Manager: {'✅' if model_test else '❌'}")

        print("\\n🎯 Run 'python scripts/senter_main.py init' to fully initialize the system.")


if __name__ == "__main__":
    main()