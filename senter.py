#!/usr/bin/env python3
"""
Senter - Capability-Discovery Agent
Multi-Model | Skill-Aware | Agentic

A lightweight Python agent that discovers and uses skills via RAG-based selection.
No external dependencies required (standard library only).
"""

import json, os, re, subprocess, urllib.request, sys, math, hashlib, threading, time, platform, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Configuration (Environment Variables) ---
CONFIG = {
    "server_url": os.getenv("SENTER_URL", "http://localhost:8081/v1"),
    "aux_urls": [u.strip() for u in os.getenv("SENTER_AUX_URLS", "").split(",") if u.strip()],
    "model_name": os.getenv("SENTER_MODEL", "glm-4.7-flash"),
    "embedding_model": os.getenv("SENTER_EMBED_MODEL", "nomic-embed-text"),
    "agents_md": os.path.join(os.getcwd(), "AGENTS.md"),
    "todo_md": os.path.join(os.getcwd(), "TODO.md"),
    "skills_roots": [
        os.path.expanduser("~/.opencode/skills"),
        os.path.expanduser("~/.claude/skills"),
        os.path.join(os.getcwd(), ".agent/skills"),
        os.path.join(os.getcwd(), "skills"),
    ],
    "history_limit": int(os.getenv("SENTER_HISTORY", 20)),
    "auto_approve_safe": True,
}

# --- Colors & Formatting ---
RESET, BOLD, DIM, ITALIC = "\033[0m", "\033[1m", "\033[2m", "\033[3m"
BLUE, CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"

# --- Server & Model Registry ---

class ModelRegistry:
    def __init__(self):
        self.routes = {}
        self.discovered = []

    def probe(self):
        """Scans all configured URLs to find available models."""
        urls = [CONFIG["server_url"]] + CONFIG["aux_urls"]
        
        for base_url in urls:
            try:
                url = f"{base_url}/models" if not base_url.endswith("/models") else base_url
                with urllib.request.urlopen(url, timeout=5) as res:
                    data = json.loads(res.read())
                    for m in data.get("data", []):
                        m_id = m["id"]
                        self.routes[m_id] = base_url
                        if m_id not in self.discovered:
                            self.discovered.append(m_id)
            except:
                pass

    def get_url(self, model):
        return self.routes.get(model, CONFIG["server_url"])

registry = ModelRegistry()

# --- Task Tracker ---

class TaskTracker:
    def __init__(self, path):
        self.path = path
    
    def plan(self, goal):
        tasks = [f"- [ ] {line.strip()}" for line in goal.split('\n') if line.strip()]
        with open(self.path, 'w') as f:
            f.write(f"# TODO\n\n" + '\n'.join(tasks))
        return len(tasks)
    
    def get_next_task(self):
        if not os.path.exists(self.path): return None
        with open(self.path, 'r') as f:
            for line in f:
                if re.match(r'^- \[ \]', line):
                    return line[6:].strip()
        return None
    
    def mark_complete(self, task_text):
        if not os.path.exists(self.path): return False
        with open(self.path, 'r') as f: content = f.read()
        new_content = content.replace(f"- [ ] {task_text}", f"- [x] {task_text}", 1)
        with open(self.path, 'w') as f: f.write(new_content)
        return new_content != content

task_tracker = TaskTracker(CONFIG["todo_md"])

# --- Background Task Manager ---

class BackgroundTaskManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.tasks = {}
        self.results = {}
        self.counter = 0
    
    def add_task(self, task_type, query, callable_fn):
        self.counter += 1
        task_id = f"{task_type}_{self.counter}"
        self.tasks[task_id] = {"type": task_type, "query": query[:50], "status": "running"}
        future = self.executor.submit(self._run, task_id, callable_fn)
        return task_id
    
    def _run(self, task_id, fn):
        try:
            result = fn()
            self.results[task_id] = result
            self.tasks[task_id]["status"] = "done"
        except Exception as e:
            self.results[task_id] = f"Error: {e}"
            self.tasks[task_id]["status"] = "error"
    
    def check_completed(self):
        for task_id, task in list(self.tasks.items()):
            if task["status"] == "done":
                print(f"\n{GREEN}✅ {task['type'].upper()} Task Complete (ID: {task_id}){RESET}")
            elif task["status"] == "error":
                print(f"\n{RED}❌ Task {task_id} Error: {self.results.get(task_id)}{RESET}")

bg_manager = BackgroundTaskManager()

# --- RAG & Skill Sync ---

def get_embedding(text):
    try:
        url = f"{CONFIG['server_url']}/embeddings"
        data = json.dumps({"model": CONFIG["embedding_model"], "input": [text]}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as res:
            return json.loads(res.read())["data"][0]["embedding"]
    except:
        return None

def cosine_similarity(v1, v2):
    if not v1 or not v2: return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1, n2 = math.sqrt(sum(a*a for a in v1)), math.sqrt(sum(b*b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0

SKILLS_IDX = []

def sync_agents_md():
    found_skills = []
    for root in CONFIG["skills_roots"]:
        if os.path.exists(root):
            try:
                paths = glob.glob(os.path.join(root, "**", "SKILL.md"), recursive=True)
                for p in paths:
                    try:
                        with open(p, 'r') as f:
                            content = f.read()
                        name = Path(p).parent.name
                        desc, preferred_model = "No description.", None
                        
                        if content.startswith("---"):
                            try:
                                frontmatter = content.split("---")[1]
                                desc_match = re.search(r'description:\s*(.*)', frontmatter)
                                if desc_match: desc = desc_match.group(1).strip()
                                model_match = re.search(r'model:\s*(.*)', frontmatter)
                                if model_match: preferred_model = model_match.group(1).strip()
                            except: pass
                            
                        found_skills.append({
                            "name": name, 
                            "desc": desc, 
                            "path": str(p), 
                            "model": preferred_model
                        })
                    except: pass
            except: pass
    
    # Update AGENTS.md
    xml = "<available_skills>\n"
    for s in found_skills:
        m_tag = f" <model>{s['model']}</model>" if s['model'] else ""
        xml += f"<skill>\n<name>{s['name']}</name>{m_tag}\n<description>{s['desc']}</description>\n</skill>\n"
    xml += "</available_skills>"
    
    if os.path.exists(CONFIG["agents_md"]):
        with open(CONFIG["agents_md"], 'r') as f:
            current_content = f.read()
        if "<skills_system" in current_content:
            new_content = re.sub(r'<skills_system priority="1">.*?</skills_system>', 
                f'<skills_system priority="1">\n\n## Available Skills\n\n{xml}\n\n</skills_system>', 
                current_content, flags=re.DOTALL)
        else:
            new_content = current_content + f"\n\n<skills_system priority=\"1\">\n\n## Available Skills\n\n{xml}\n\n</skills_system>"
    else:
        new_content = f"# AGENTS\n\n> Auto-generated by Senter.\n\n<skills_system priority=\"1\">\n\n## Available Skills\n\n{xml}\n\n</skills_system>"
        
    with open(CONFIG["agents_md"], "w") as f: f.write(new_content)
    return found_skills

def index_all():
    global SKILLS_IDX
    skills = sync_agents_md()
    new_idx = []
    for s in skills:
        new_idx.append({
            "name": s["name"],
            "desc": s["desc"],
            "path": s["path"],
            "model": s["model"],
            "embed": get_embedding(s["desc"])
        })
    SKILLS_IDX = new_idx

# --- API Handling ---

def call(msgs, model_override=None):
    target_model = model_override if model_override else CONFIG["model_name"]
    target_url = registry.get_url(target_model)
    
    url = f"{target_url}/chat/completions"
    data = {"model": target_model, "messages": msgs, "stream": True}
    
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            for line in res:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                    except GeneratorExit:
                        return
                    except:
                        continue
    except GeneratorExit:
        return
    except urllib.error.HTTPError as e:
        yield f"\n{RED}[HTTP Error {e.code}: {e.reason}]{RESET}"
    except urllib.error.URLError as e:
        yield f"\n{RED}[Connection Error: {e.reason}]{RESET}"
    except Exception as e:
        yield f"\n{RED}[Error: {str(e)}]{RESET}"

def unload_model(model_name):
    try:
        api_url = f"{registry.get_url(model_name)}/generate" if "/v1" not in registry.get_url(model_name) else registry.get_url(model_name).replace("/v1", "/api/generate")
        data = json.dumps({"model": model_name, "keep_alive": 0}).encode()
        req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            return True
    except:
        return False

def extract_cmd(full_text):
    m = re.search(r'''bash\(cmd=(['"])(.*?)\1\)''', full_text, re.DOTALL)
    if m: return m.group(2).strip()
    return None

def get_system_context():
    return f"OS: {platform.system()} {platform.release()}\nShell: {os.environ.get('SHELL', 'unknown')}\nCWD: {os.getcwd()}\nUser: {os.environ.get('USER', 'unknown')}"

def render_markdown(text):
    """Simple terminal markdown renderer using ANSI codes."""
    text = re.sub(r'^(#{1,3})\s+(.+)$', f'{BOLD}{CYAN}\\2{RESET}', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}\\1{RESET}', text)
    text = re.sub(r'\*(.+?)\*', f'{DIM}\\1{RESET}', text)
    text = re.sub(r'`([^`]+)`', f'{YELLOW}\\1{RESET}', text)
    text = re.sub(r'^(\s*[-*])\s+', f'{DIM}•{RESET} ', text, flags=re.MULTILINE)
    return text

class Heartbeat:
    def __init__(self):
        self.stop_event = threading.Event(); self.thread = None
    def _run(self):
        dots, increasing = 0, True
        while not self.stop_event.is_set():
            dots = (dots + 1) if increasing else (dots - 1)
            if dots >= 3: increasing = False
            if dots <= 1: increasing = True
            print(f"\r{DIM}Thinking {'⏺' * dots}{'  ' * (3-dots)}{RESET}", end="", flush=True)
            time.sleep(0.4)
        print("\r" + " " * 80 + "\r", end="", flush=True)
    def start(self):
        self.stop_event.clear(); self.thread = threading.Thread(target=self._run); self.thread.daemon = True; self.thread.start()
    def stop(self):
        if self.thread: self.stop_event.set(); self.thread.join(timeout=1)

def interactive_execute(cmd, whitelist):
    is_safe = False
    if any(cmd == a or a in cmd for a in whitelist): is_safe = True
    dangerous = ["rm ", "> ", "mv ", "dd ", "mkfs", ":(){:|:&};:"]
    if any(p in cmd for p in dangerous): is_safe = False
    
    if is_safe: return subprocess.getoutput(cmd)
    
    print(f"\n{YELLOW}⚠️  Confirm: {BOLD}{cmd}{RESET}")
    choice = input(f"{YELLOW}[y/N]{RESET} ").strip().lower()
    return subprocess.getoutput(cmd) if choice == 'y' else "[SYSTEM: Denied]"

def main():
    import shutil
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner = ["███████╗███████╗███╗   ██╗████████╗███████╗██████╗ ",
              "██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗",
              "███████╗█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝",
              "╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗",
              "███████║███████╗██║ ╚████║   ██║   ███████╗██║  ██║",
              "╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═ "]
    print("\n" + BOLD + CYAN)
    for l in banner: print(" " * ((term_width - len(l)) // 2) + l)
    
    # Discovery
    registry.probe()
    if not registry.discovered:
        print(f"{YELLOW}No models found. Check SENTER_URL environment variable.{RESET}")
    
    # Connect
    connected_model = None
    for i in range(5):
        try:
             url = f"{CONFIG['server_url']}/models"
             with urllib.request.urlopen(url, timeout=5) as res:
                data = json.loads(res.read())
                available = [m["id"] for m in data["data"]]
                
                if CONFIG["model_name"] in available:
                    connected_model = CONFIG["model_name"]
                elif any("glm" in m.lower() for m in available):
                    connected_model = next(m for m in available if "glm" in m.lower())
                elif available:
                    connected_model = available[0]
                
                if connected_model:
                    CONFIG["model_name"] = connected_model
                    break
        except: pass
        time.sleep(2)
    
    if not connected_model:
        print(f"{RED}Critical: No models available at {CONFIG['server_url']}{RESET}")
        return
    
    index_all()
    
    # Centered Status
    skill_count_str = f"has {len(SKILLS_IDX)} Skills"
    model_str = f"Model: {CONFIG['model_name']}"
    print(" " * ((term_width - len(skill_count_str)) // 2) + f"{DIM}{skill_count_str}{RESET}")
    print(" " * ((term_width - len(model_str)) // 2) + f"{CYAN}{model_str}{RESET}")
    print("\n")
    
    history = []
    print(f"{DIM}{'─'*term_width}{RESET}")
    
    next_task = task_tracker.get_next_task()
    if next_task: print(f"{MAGENTA}Current Focus: {next_task}{RESET}")

    while True:
        try:
            prompt_symbol = f"{BOLD}{MAGENTA}Focus ❯{RESET} " if next_task else f"{BOLD}{BLUE}❯{RESET} "
            query = input(prompt_symbol).strip()
            if not query: continue
            
            # Meta Commands
            if query in ("/q", "exit"): break
            if query == "/c": history = []; print(f"{GREEN}Cleared.{RESET}"); continue
            if query.startswith("/plan"):
                count = task_tracker.plan(query[5:].strip())
                print(f"{GREEN}Created plan with {count} tasks. Use /next.{RESET}"); continue
            if query == "/next":
                next_task = task_tracker.get_next_task()
                if not next_task: print(f"{GREEN}All tasks complete!{RESET}"); continue
                print(f"{MAGENTA}Focusing: {next_task}{RESET}")
                query = f"Execute: {next_task}"
            if query.startswith("/done"):
                if task_tracker.mark_complete(query[5:].strip()): print(f"{GREEN}Task done.{RESET}")
                continue
            if query.startswith("/unload"):
                m = query[7:].strip()
                if unload_model(m): print(f"{GREEN}Unloaded {m}{RESET}")
                else: print(f"{RED}Failed to unload {m}{RESET}")
                continue

            # RAG
            index_all()  # Hot Reload
            q_embed = get_embedding(query)
            best_s = sorted(SKILLS_IDX, key=lambda x: cosine_similarity(q_embed, x["embed"]), reverse=True)[:4]
            
            selected_names = [s['name'] for s in best_s]
            print(f"{DIM}Selected Skills: {', '.join(selected_names)}{RESET}")
            
            toolbox, whitelist, active_model_override = "", [], None
            
            if best_s and best_s[0]["model"]:
                active_model_override = best_s[0]["model"]
                print(f"{DIM}[Switched to {CYAN}{active_model_override}{RESET}{DIM} for skill '{best_s[0]['name']}']{RESET}")

            for s in best_s:
                try:
                    with open(s["path"], 'r') as f: content = f.read()
                    skill_root = str(Path(s["path"]).parent)
                    if content.startswith("---"): content = content.split("---", 2)[2]
                    doc_content = content.replace('./', skill_root + '/')
                    toolbox += f"### CAPABILITY: {s['name']}\n{doc_content}\n---\n"
                    for m in re.finditer(r'''bash\(cmd=(['"])(.*?)\1\)''', doc_content, re.DOTALL):
                        whitelist.append(m.group(2).strip())
                except: pass

            sys_p = (
                f"You are Senter. {get_system_context()}\n"
                f"Active Model: {active_model_override if active_model_override else CONFIG['model_name']}\n"
                f"Focus: {next_task if next_task else 'User Query'}\n\n"
                "CAPABILITIES:\n"
                f"{toolbox}\n"
                "RULES:\n"
                "1. Format: <tool_code>print(bash(cmd='...'))</tool_code>\n"
                "2. Use ONLY exact commands from CAPABILITIES.\n"
            )
            
            msgs = [{"role": "system", "content": sys_p}] + history + [{"role": "user", "content": query}]
            
            t_count = 0
            cmd_history = []
            while t_count < 15:
                full, visible_text = "", ""
                hb = Heartbeat(); hb.start()
                gen = call(msgs, model_override=active_model_override)
                suppressed = False
                
                try:
                    for chunk in gen:
                        full += chunk
                        if not suppressed:
                            clean = re.sub(r'<think>.*?</think>', '', full, flags=re.DOTALL)
                            clean = re.sub(r'<think>.*', '', clean, flags=re.DOTALL)
                            clean = re.sub(r'<tool_code>.*', '', clean, flags=re.DOTALL)
                            visible = clean
                            if visible.strip() and visible_text == "": hb.stop()
                            for t in ["<", "bash(", "print(", "tool_code"]:
                                if t in visible.lower(): 
                                    visible = visible[:visible.lower().find(t)]; suppressed = True
                            if len(visible) > len(visible_text):
                                new_text = visible[len(visible_text):]
                                print(render_markdown(new_text), end="", flush=True)
                                visible_text = visible
                        if extract_cmd(full): break
                finally:
                    hb.stop() 
                    try: gen.close()
                    except GeneratorExit: pass
                    except: pass
                    print()
                
                cmd = extract_cmd(full)
                if cmd:
                    if cmd in cmd_history:
                         msgs.append({"role": "user", "content": "[SYSTEM: ERROR: Loop detected. CHANGE STRATEGY.]"})
                         t_count += 1; continue
                    cmd_history.append(cmd)
                    
                    print(f"{GREEN}⏺ Bash{RESET}({DIM}{cmd[:60]}{RESET}) {YELLOW}...{RESET}", end="", flush=True)
                    res = interactive_execute(cmd, whitelist)
                    print(f"\r{GREEN}⏺ Bash{RESET}({DIM}{cmd[:60]}{RESET}) {GREEN}OK{RESET}")
                    
                    clean_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()
                    msgs.append({"role": "assistant", "content": full})
                    msgs.append({"role": "user", "content": f"<|observation|>\n{clean_res}\n\n[SYSTEM: Continue processing based on this observation.]"})
                    t_count += 1; continue
                
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": full})
                if len(history) > CONFIG["history_limit"]: history = history[-CONFIG["history_limit"]:]
                break
                
        except KeyboardInterrupt:
            print(f"\n{DIM}Interrupted.{RESET}")
        except EOFError:
            break

    print(f"\n{DIM}Goodbye.{RESET}")

if __name__ == "__main__":
    main()
