#!/usr/bin/env python3
"""
SENTER KERNEL v3.5 - The Refined Selection Engine
- History stored in STATE/ChatHistory.md
- Two-stage Progressive Disclosure (Selection Engine)
- Configurable Main Loop from AGENTS.md
"""

import json, os, re, subprocess, urllib.request, sys, hashlib, threading, time, platform, glob, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Kernel Configuration ---
VAULT_ROOT = Path(os.getenv("SENTER_VAULT", "~/.senter/vault")).expanduser()
HISTORY_FILE = VAULT_ROOT / "STATE/ChatHistory.md"
CONFIG = {
    "server_url": os.getenv("SENTER_URL", "http://localhost:8081/v1"),
    "model_name": os.getenv("SENTER_MODEL", "glm-4.7-flash"),
    "embedding_model": "nomic-embed-text",
}

# --- Colors ---
RESET, BOLD, DIM, ITALIC = "\033[0m", "\033[1m", "\033[2m", "\033[3m"
BLUE, CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"

def render_markdown(text):
    text = re.sub(r'^(#{1,3})\s+(.+)$', f'{BOLD}{MAGENTA}\2{RESET}', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}\1{RESET}', text)
    text = re.sub(r'\*(.+?)\*', f'{ITALIC}\1{RESET}', text)
    text = re.sub(r'`([^`]+)`', f'{YELLOW}\1{RESET}', text)
    text = re.sub(r'^(\s*[-*])\s+', f'{DIM}•{RESET} ', text, flags=re.MULTILINE)
    return text

class SenterKernel:
    def __init__(self):
        self.skills, self.agents, self.hooks = {}, {}, []
        self.bg_executor = ThreadPoolExecutor(max_workers=5)
        self.cache_path = Path("~/.senter/embedding_cache.json").expanduser()
        self.embed_cache = self._load_cache()
        VAULT_ROOT.joinpath("STATE").mkdir(parents=True, exist_ok=True)

    def _load_cache(self):
        if self.cache_path.exists():
            try: return json.loads(self.cache_path.read_text())
            except: return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.embed_cache))

    def get_embedding(self, text):
        if not text: return None
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self.embed_cache: return self.embed_cache[h]
        try:
            url = f"{CONFIG['server_url']}/embeddings"
            data = json.dumps({"model": CONFIG["embedding_model"], "input": [text]}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as res:
                v = json.loads(res.read())["data"][0]["embedding"]
                self.embed_cache[h] = v
                return v
        except: return None

    def cosine_similarity(self, v1, v2):
        if not v1 or not v2: return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return dot / (math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2)))

    def index_vault(self):
        for p in VAULT_ROOT.glob("AGENTS/*.md"): self.agents[p.stem] = p.read_text()
        for p in VAULT_ROOT.glob("SKILLS/**/*.md"):
            c = p.read_text(); d = "No description."
            if c.startswith("---"):
                try: d = re.search(r'description:\s*(.*)', c.split("---")[1]).group(1).strip()
                except: pass
            self.skills[p.stem] = {"path": str(p), "content": c, "desc": d, "embed": self.get_embedding(d)}
        self.hooks = []
        for p in VAULT_ROOT.glob("HOOKS/*.md"):
            c = p.read_text()
            if c.startswith("---"):
                try:
                    m = c.split("---")[1]
                    e = re.search(r'event:\s*(.*)', m).group(1).strip()
                    self.hooks.append({"event": e, "content": c.split("---", 2)[2]})
                except: pass
        self._save_cache()

    def run_hooks(self, event_name, context=""):
        for hook in self.hooks:
            if hook["event"] == event_name:
                cmds = re.findall(r'''bash\(cmd=(['"])(.*?)\1\)''', hook["content"], re.DOTALL)
                for _, cmd in cmds: subprocess.run(cmd, shell=True)

    def select_top_one(self, query, items, item_type="Skill"):
        if not items: return None
        if len(items) == 1: return items[0]
        prompt = f"User Query: \"{query}\"\nSelect the single most relevant {item_type} from this list:\n"
        for i, item in enumerate(items):
            desc = item.get("desc") or item.get("text", "No description")[:200]
            name = item.get("path", "").split("/")[-1] if "path" in item else f"Item {i+1}"
            prompt += f"{i+1}. [{name}] {desc}\n"
        prompt += f"\nRespond ONLY with the number (1-{len(items)}). No other text."
        messages = [{"role": "system", "content": "You are a selector agent. Only respond with a number."}, {"role": "user", "content": prompt}]
        full_resp = ""
        for chunk in self.call_llm(messages): full_resp += chunk
        match = re.search(r'(\d)', full_resp)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(items): return items[idx]
        return items[0]

    def get_history_context(self, query):
        if not HISTORY_FILE.exists(): return []
        content = HISTORY_FILE.read_text()
        turns = [t.strip() for t in re.split(r'\n---\n', content) if t.strip()]
        if not turns: return []
        presence = turns[-2:] if len(turns) >= 2 else turns
        q_v = self.get_embedding(query)
        if not q_v or len(turns) <= 2: return self._parse_turns(presence)
        pool = turns[:-2]
        ranked = sorted([{"text": t, "score": self.cosine_similarity(q_v, self.get_embedding(t[:500]))} 
                        for t in pool if self.get_embedding(t[:500])], 
                       key=lambda x: x["score"], reverse=True)[:4]
        best_memory = self.select_top_one(query, ranked, item_type="Memory")
        context_turns = ([best_memory["text"]] if best_memory else []) + presence
        return self._parse_turns(context_turns)

    def _parse_turns(self, turns):
        msgs = []
        for t in turns:
            parts = t.split("\n", 1)
            if len(parts) == 2:
                role = "user" if "USER:" in parts[0] else "assistant"
                msgs.append({"role": role, "content": parts[1].strip()})
        return msgs

    def save_turn(self, query, response):
        with open(HISTORY_FILE, "a") as f: f.write(f"USER:\n{query}\n---\nASSISTANT:\n{response}\n---\n")

    def call_llm(self, messages):
        url = f"{CONFIG['server_url']}/chat/completions"
        data = {
            "model": CONFIG["model_name"], 
            "messages": messages, 
            "stream": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "repeat_penalty": 1.0, # Recommended to disable for GLM 4.7 Flash
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as res:
                for line in res:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            c = delta.get("content") or delta.get("reasoning_content")
                            if c: yield c
        except Exception as e: pass

    def extract_tool(self, text):
        m = re.search(r'(bash|spawn)\((?:cmd|task)=([\'"])(.*?)\2\)', text, re.DOTALL)
        if m: return m.group(1), m.group(3).strip()
        return None, None

def get_main_loop_config():
    p = VAULT_ROOT / "AGENTS.md"
    if not p.exists(): return None
    try:
        content = p.read_text()
        m = re.search(r'<main_loop>(.*?)</main_loop>', content)
        if m: return m.group(1).strip()
    except: pass
    return None

def main():
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", help="Run single task"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(); kernel = SenterKernel(); kernel.index_vault()
    
    if not args.worker:
        term_width = shutil.get_terminal_size((80, 20)).columns
        banner = ["███████╗███████╗███╗   ██╗████████╗███████╗██████╗ ",
                  "██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗",
                  "███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝",
                  "╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗",
                  "███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║",
                  "╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═ "]
        print("\n" + BOLD + GREEN)
        for l in banner: print(" " * ((term_width - len(l)) // 2) + l)
        print(" " * ((term_width - 15) // 2) + f"{DIM}has {len(kernel.skills)} Skills{RESET}")
        print(" " * ((term_width - 25) // 2) + f"{CYAN}Model: {CONFIG['model_name']}{RESET}")
        print(f"\n{DIM}{'─'*term_width}{RESET}")

    kernel.run_hooks("on_startup")
    main_loop_skill = get_main_loop_config()
    first_run = True

    while True:
        try:
            if args.command and first_run: query = args.command; first_run = False
            elif main_loop_skill and first_run:
                print(f"{MAGENTA}Running Main Loop: {main_loop_skill}{RESET}")
                query = f"Execute main loop: {main_loop_skill}"; first_run = False
            else:
                if args.worker: break
                query = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            if not query: continue

            # 1. Skill Selection
            q_v = kernel.get_embedding(query)
            top_4_skills = sorted(kernel.skills.values(), key=lambda x: kernel.cosine_similarity(q_v, x["embed"]), reverse=True)[:4]
            picked_skill = kernel.select_top_one(query, top_4_skills, item_type="Skill")
            skills_p = f"### {picked_skill['path'].split('/')[-1]}\n{picked_skill['content']}" if picked_skill else ""
            
            # 2. History Selection
            history_context = kernel.get_history_context(query)
            
            agent_p = kernel.agents.get("Senter", "You are Senter, a minimal AI agent.")
            sys_p = f"{agent_p}\n\nCAPABILITIES:\n{skills_p}\n\nRULES:\n1. Use <tool_code>bash(cmd='...')</tool_code>\n2. Use <tool_code>spawn(task='...')</tool_code>\n3. PREFER the Terminal skill for running commands."
            
            messages = [{"role": "system", "content": sys_p}] + history_context + [{"role": "user", "content": query}]
            
            for _ in range(10):
                full_response, visible_text, suppressed = "", "", False
                for chunk in kernel.call_llm(messages):
                    full_response += chunk
                    if not args.worker and not suppressed:
                        clean = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
                        clean = re.sub(r'<think>.*', '', clean, flags=re.DOTALL)
                        clean = re.sub(r'<tool_code>.*', '', clean, flags=re.DOTALL)
                        for t in ["<", "bash(", "spawn("]:
                            if t in clean.lower(): clean = clean[:clean.lower().find(t)]; suppressed = True
                        if len(clean) > len(visible_text):
                            print(render_markdown(clean[len(visible_text):]), end="", flush=True)
                            visible_text = clean
                    if kernel.extract_tool(full_response)[0]: break

                t_type, t_payload = kernel.extract_tool(full_response)
                if t_type:
                    print(f"{GREEN}⏺ {t_type.title()}{RESET}({DIM}{t_payload[:60]}{RESET}) ...", end="", flush=True)
                    res = subprocess.getoutput(t_payload) if t_type == "bash" else f"[Agent Spawned: {t_payload}]"
                    if t_type == "spawn": kernel.bg_executor.submit(lambda: subprocess.run([sys.executable, __file__, "-c", t_payload, "--worker"]))
                    print(f"\r{GREEN}⏺ {t_type.title()}{RESET}({DIM}{t_payload[:60]}{RESET}) {GREEN}OK{RESET}")
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": f"<|observation|>\n{res}\n\nContinue."})
                else:
                    kernel.save_turn(query, full_response); print(); break
        except KeyboardInterrupt: break
        except Exception as e: print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__": main()
