"""
virtual_assistant.py
A modern virtual assistant GUI:
 - Chat (with memory saved to memory.json)
 - Speech-to-text (optional, via speech_recognition)
 - Text-to-speech (pyttsx3)
 - Prediction using OpenRouter model tngtech/deepseek-r1t2-chimera:free
 - Local fallback math evaluator if OR_Key missing/unavailable
"""

import os
import json
import threading
import time
import queue
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import customtkinter as ctk
from tkinter import messagebox, filedialog
import requests
from dotenv import load_dotenv
import pyttsx3

# Optional: speech recognition (microphone -> text)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# ---------------------------
# Configuration & Constants
# ---------------------------
APP_TITLE = "Jarvis Virtual Assistant"
MEMORY_FILE = "memory.json"
MAX_MEMORY_MESSAGES = 200  # cap conversation memory to this many messages
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Chat completions compatible
MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"



# place this where your other classes are (replace old TTS)
def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)



# Load env
load_dotenv()
OR_KEY = os.getenv("OR_Key") or os.getenv("OR_KEY") or os.getenv("ORKey")  # be flexible

# ---------------------------
# Utilities
# ---------------------------
def save_memory(messages: List[dict]):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages[-MAX_MEMORY_MESSAGES:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save memory:", e)


def load_memory() -> List[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# Safe math evaluator (fallback) — evaluates arithmetic expressions only.
import ast, operator as op, math

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}

SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_NAMES.update({"abs": abs, "round": round})

def safe_eval(expr: str):
    """
    Evaluate a math expression safely using AST parsing.
    Supports functions from math (sin, cos, log, etc.) and basic operators.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):  # <number>
            return node.n
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant in expression")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Operator {op_type} not allowed")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Unary operator {op_type} not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsafe function call")
            func_name = node.func.id
            if func_name not in SAFE_NAMES:
                raise ValueError(f"Function {func_name} not allowed")
            args = [_eval(arg) for arg in node.args]
            return SAFE_NAMES[func_name](*args)
        raise ValueError("Unsupported expression")
    try:
        node = ast.parse(expr, mode='eval')
        return _eval(node)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

# ---------------------------
# OpenRouter Integration
# ---------------------------

def call_openrouter_chat(messages: List[dict], temperature: float = 0.0, max_tokens: int = 512):
    """
    Robust call to OpenRouter. Returns trimmed reply string or raises RuntimeError.
    Tries several possible response shapes and logs the raw response when things look wrong.
    """
    if not OR_KEY:
        raise RuntimeError("OR_Key not found in environment.")

    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Try the common ChatCompletions shape first
    try:
        choices = data.get("choices") or []
        if isinstance(choices, list) and len(choices) > 0:
            c = choices[0]
            # new-style: choices[0].message.content
            if isinstance(c, dict):
                msg = c.get("message") or {}
                if isinstance(msg, dict) and msg.get("content"):
                    content = msg.get("content")
                elif c.get("text"):
                    content = c.get("text")
                else:
                    # streaming delta style?
                    delta = c.get("delta") or {}
                    content = delta.get("content") or None
                if content:
                    return str(content).strip()
        # fallback top-level fields
        if isinstance(data.get("output"), str) and data.get("output").strip():
            return data.get("output").strip()
        if isinstance(data.get("result"), str) and data.get("result").strip():
            return data.get("result").strip()
    except Exception:
        pass

    # If we reach here, response didn't contain usable text — log for debugging and raise
    try:
        print("[DEBUG] OpenRouter raw response:", json.dumps(data)[:4000])
    except Exception:
        print("[DEBUG] OpenRouter returned non-JSON or very large response")
    # If we cannot extract content, raise to let caller handle fallback
    raise RuntimeError("OpenRouter returned no usable content.")



# ---------------------------
# TTS Engine
# ---------------------------

class TTS:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._engine_loop, daemon=True)
        self.thread.start()

    def speak(self, text: str):
        text = remove_emojis(text or "")
        if text.strip():
            # Put text in queue; it will be spoken in order
            self.queue.put(text)

    def _engine_loop(self):
        """Run in dedicated thread. Try SAPI (win32com) first, fallback to pyttsx3."""
        use_sapi = False
        speaker = None
        engine = None

        # Try to import win32com (SAPI) inside thread to avoid circular/import issues
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()  # initialize COM for this thread
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            use_sapi = True
        except Exception:
            use_sapi = False
            # will use pyttsx3 as fallback; initialize lazily
            engine = None

        while self.running:
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not text:
                self.queue.task_done()
                continue

            try:
                if use_sapi and speaker is not None:
                    # SAPI speak is synchronous and reliable on Windows
                    speaker.Speak(text)
                else:
                    # fallback: pyttsx3 in this same thread to avoid concurrency issues
                    if engine is None:
                        try:
                            engine = pyttsx3.init()
                            rate = engine.getProperty("rate")
                            engine.setProperty("rate", int(rate * 0.95))
                        except Exception:
                            engine = None
                    if engine is not None:
                        engine.say(text)
                        engine.runAndWait()
                    else:
                        # Last resort: just print (so you see response)
                        print("[TTS fallback] " + text)
            except Exception as ex:
                # Try to recover: re-init engines if needed
                try:
                    if use_sapi:
                        # try reinit COM & speaker
                        import pythoncom, win32com.client
                        pythoncom.CoInitialize()
                        speaker = win32com.client.Dispatch("SAPI.SpVoice")
                        speaker.Speak(text)
                    else:
                        engine = pyttsx3.init()
                        engine.say(text)
                        engine.runAndWait()
                except Exception:
                    # give up this utterance but don't crash loop
                    print("TTS error:", ex)
            finally:
                try:
                    self.queue.task_done()
                except Exception:
                    pass

        # cleanup if exiting
        try:
            if use_sapi:
                # Uninitialize COM for this thread
                import pythoncom
                pythoncom.CoUninitialize()
        except Exception:
            pass

    def stop(self):
        """Stop thread: push sentinel and wait briefly"""
        self.running = False
        # Put something to unblock queue.get if waiting
        try:
            self.queue.put("")
        except Exception:
            pass
        # optionally join thread (don't block GUI for long)
        # self.thread.join(timeout=1)

# ---------------------------
# Assistant core
# ---------------------------
@dataclass
class Message:
    role: str  # 'user' | 'assistant' | 'system'
    content: str

class AssistantCore:
    def __init__(self, tts_enabled=True):
        self.memory = load_memory()  # list of dicts {role,content}
        self.tts = TTS() if tts_enabled else None

    def append_memory(self, role: str, content: str):
        """
        Only save non-empty messages to memory.
        User messages are trimmed and saved if non-empty.
        Assistant messages that are empty are NOT saved.
        """
        if content is None:
            return
        content = str(content).strip()
        if not content:
            return
        self.memory.append({"role": role, "content": content})
        save_memory(self.memory)


    def clear_memory(self):
        self.memory = []
        save_memory(self.memory)

    def ask_model(self, user_text: str, system_prompt: Optional[str] = None, temperature=0.0, timeout=30):
        """
        Ask remote model; robustly handle empty responses.
        - Try the primary call
        - If empty or fails, retry once with a very simple system prompt
        - If still empty, attempt safe_eval for math
        - If everything fails, return polite fallback string (and do NOT save empty assistant messages)
        """
        system_prompt = system_prompt or """ You are Jarvis, a smart, friendly, and natural-sounding virtual assistant. Your goal is to help the user quickly and clearly. 
        
        Follow these rules:

        1. Respond conversationally, politely, and concisely to greetings, questions, or casual chat.
        2. If the user asks a math or calculation question, provide the answer clearly in plain text.
        3. Only return JSON with keys "prediction", "confidence", and "explanation" if the user explicitly asks to "predict", "forecast", or "estimate" something. Never use JSON otherwise.
        4. Help the user with useful advice, reminders, scheduling tips, or general knowledge.
        5. Do not expose technical errors, system details, or internal reasoning.
        6. Never explain your thought process unless asked explicitly.
        7. Do not include emojis or decorative symbols.
        8. Understand sequences, patterns, and context from previous messages, but keep your replies short.
        9. Clarify ambiguous requests politely if you cannot understand.
        10. Be proactive: if a question is vague, ask for more details instead of guessing.
        11. Always aim to be helpful, friendly, and accurate, whether for general knowledge, math, predictions, or practical advice.
        12. Don't Use Bold, Italic or Any Stylish Text. 
        13. Never Use Bold Text.
        14. Use Simple and Plain Text.
        15. Never Use Formatted Text.
        """

        # Build context: include last N memory items (already trimmed by append_memory)
        messages = []
        last_mem = self.memory[-20:]
        messages.extend(last_mem)
        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        # Helper to safely append user + assistant to memory
        def save_pair(user_msg, assistant_msg):
            # Always save user (if non-empty) and assistant only if not empty
            if user_msg and str(user_msg).strip():
                self.append_memory("user", user_msg)
            if assistant_msg and str(assistant_msg).strip():
                self.append_memory("assistant", assistant_msg)

        # Try primary API call
        try:
            reply = call_openrouter_chat(messages, temperature=temperature)
            if reply and reply.strip():
                save_pair(user_text, reply)
                return reply
            # Empty reply -> fall through to retry
            print("[DEBUG] OpenRouter returned empty reply; attempting simple-retry.")
        except Exception as e:
            print("[DEBUG] OpenRouter primary call failed:", e)

        # Retry once with a minimal system prompt to provoke a short reply
        try:
            retry_sys = "You are a helpful assistant. Give a short friendly reply to the user's input."
            retry_messages = [{"role": "system", "content": retry_sys}, {"role": "user", "content": user_text}]
            reply = call_openrouter_chat(retry_messages, temperature=max(0.0, min(0.5, temperature)))
            if reply and reply.strip():
                save_pair(user_text, reply)
                return reply
            print("[DEBUG] OpenRouter retry returned empty reply.")
        except Exception as e:
            print("[DEBUG] OpenRouter retry failed:", e)

        # If input looks like math, try safe local eval
        try:
            result = safe_eval(user_text)
            reply = json.dumps({
                "prediction": result,
                "confidence": 0.6,
                "explanation": "Computed locally (safe evaluator)."
            })
            save_pair(user_text, reply)
            return reply
        except Exception as e2:
            print("[DEBUG] Local safe_eval failed:", e2)

        # Final polite fallback (guaranteed non-empty)
        fallback = "Sorry — I'm having trouble forming a response right now. Please try rephrasing or ask something else."
        save_pair(user_text, fallback)
        return fallback

# ---------------------------
# GUI
# ---------------------------
class VirtualAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("dark-blue")
        self.title(APP_TITLE)
        self.geometry("980x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.core = AssistantCore(tts_enabled=True)
        self.audiorec_thread: Optional[threading.Thread] = None
        self.request_queue = queue.Queue()

        self._build_ui()
        self._process_queue_periodically()

    def _build_ui(self):
        # Layout: left = chat, right = controls/prediction + memory buttons
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Chat frame
        chat_frame = ctk.CTkFrame(self, corner_radius=10)
        chat_frame.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(1, weight=0)
        chat_frame.grid_columnconfigure(0, weight=1)

        # Chat display (read-only)
        self.chat_display = ctk.CTkTextbox(chat_frame, wrap="word", state="disabled")
        self.chat_display.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        # Bottom input
        bottom_frame = ctk.CTkFrame(chat_frame)
        bottom_frame.grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        bottom_frame.grid_columnconfigure(1, weight=1)

        self.entry = ctk.CTkEntry(bottom_frame, placeholder_text="Type your message or a math expression (e.g. 2+2, sin(0.5)*100)...")
        self.entry.grid(row=0, column=1, padx=(8,4), pady=6, sticky="ew")
        self.entry.bind("<Return>", lambda e: self.on_send())

        send_btn = ctk.CTkButton(bottom_frame, text="Send", command=self.on_send, width=90)
        send_btn.grid(row=0, column=2, padx=(4,8), pady=6)

        mic_btn = ctk.CTkButton(bottom_frame, text="🎤 Speak", command=self.on_mic_toggle, width=110)
        mic_btn.grid(row=0, column=0, padx=4, pady=6)

        # Right control panel
        ctrl = ctk.CTkFrame(self, corner_radius=10)
        ctrl.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        ctrl.grid_rowconfigure(6, weight=1)

        lbl = ctk.CTkLabel(ctrl, text="Assistant Controls", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=0, column=0, padx=12, pady=(12,6))

        # Prediction input
        self.predict_input = ctk.CTkEntry(ctrl, placeholder_text="Enter a prediction request (e.g. 'predict next 7 days price for item X' or '2+2')")
        self.predict_input.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        predict_btn = ctk.CTkButton(ctrl, text="Predict (Use Model)", command=self.on_predict)
        predict_btn.grid(row=2, column=0, padx=12, pady=(0,12), sticky="ew")

        # Settings area
        self.temp_slider = ctk.CTkSlider(ctrl, from_=0.0, to=1.0, number_of_steps=10)
        self.temp_slider.set(0.0)
        trow = ctk.CTkLabel(ctrl, text="Model Temperature (creativity)")
        trow.grid(row=3, column=0, padx=12, pady=(6,0), sticky="w")
        self.temp_slider.grid(row=4, column=0, padx=12, pady=(0,12), sticky="ew")

        # Memory controls
        mem_label = ctk.CTkLabel(ctrl, text="Memory")
        mem_label.grid(row=5, column=0, padx=12, pady=(6,2), sticky="w")
        mem_view_btn = ctk.CTkButton(ctrl, text="Open Memory File", command=self.open_memory_file)
        mem_view_btn.grid(row=6, column=0, padx=12, pady=6, sticky="ew")

        mem_clear_btn = ctk.CTkButton(ctrl, text="Clear Memory", command=self.clear_memory)
        mem_clear_btn.grid(row=7, column=0, padx=12, pady=6, sticky="ew")

        # Status
        self.status_label = ctk.CTkLabel(ctrl, text=f"Model: {MODEL_NAME}\nOR_Key: {'present' if OR_KEY else 'MISSING'}", anchor="w", justify="left")
        self.status_label.grid(row=8, column=0, padx=12, pady=(12,12), sticky="w")

        # Populate chat from memory
        self._refresh_chat_display_from_memory()

    # -----------------------
    # UI Helpers
    # -----------------------
    def _append_chat(self, role: str, text: str):
        self.chat_display.configure(state="normal")
        if role == "user":
            self.chat_display.insert("end", f"You: {text}\n\n")
        else:
            self.chat_display.insert("end", f"Assistant: {text}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _refresh_chat_display_from_memory(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        for msg in self.core.memory:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "Assistant" if role == "assistant" else ("System" if role == "system" else "You")
            self.chat_display.insert("end", f"{prefix}: {content}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    # -----------------------
    # Actions & Threads
    # -----------------------
    def on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._append_chat("user", text)
        # enqueue request to avoid UI blocking
        self.request_queue.put(("ask", text, float(self.temp_slider.get())))
    
    def on_predict(self):
        text = self.predict_input.get().strip()
        if not text:
            messagebox.showinfo("Predict", "Please enter a prediction request or math expression.")
            return
        self._append_chat("user", f"[PREDICT] {text}")
        self.request_queue.put(("ask", text, float(self.temp_slider.get())))

    def on_mic_toggle(self):
        if not SR_AVAILABLE:
            messagebox.showwarning("Speech Recognition", "speech_recognition package is not available. Install it to enable microphone input.")
            return
        # start a background thread to listen once and transcribe
        t = threading.Thread(target=self._record_and_transcribe_once, daemon=True)
        t.start()

    def _record_and_transcribe_once(self):
        # Single-shot: listen and transcribe
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self._append_chat("assistant", "Listening... speak now.")
            try:
                audio = r.listen(source, timeout=None, phrase_time_limit=None)
            except Exception as e:
                self._append_chat("assistant", f"Microphone error: {e}")
                return
        try:
            self._append_chat("assistant", "Transcribing...")
            text = r.recognize_google(audio)
            self._append_chat("user", text)
            self.request_queue.put(("ask", text, float(self.temp_slider.get())))
        except sr.UnknownValueError:
            self._append_chat("assistant", "Could not understand audio.")
        except sr.RequestError as e:
            self._append_chat("assistant", f"Speech recognition service error: {e}")

    def _process_queue_periodically(self):
        try:
            while True:
                item = self.request_queue.get_nowait()
                if item[0] == "ask":
                    _, text, temp = item
                    threading.Thread(target=self._handle_ask, args=(text, temp), daemon=True).start()
        except queue.Empty:
            pass
        self.after(200, self._process_queue_periodically)

    def _handle_ask(self, text: str, temperature: float):
        self._append_chat("assistant", "Thinking...")
        reply = self.core.ask_model(text, temperature=temperature)
        self._refresh_chat_display_from_memory()
        if self.core.tts:
            self.core.tts.speak(reply)  # queued properly



    # -----------------------
    # Memory & Files
    # -----------------------
    def open_memory_file(self):
        if not os.path.exists(MEMORY_FILE):
            messagebox.showinfo("Memory", "No memory file found.")
            return
        # open in default editor
        try:
            os.startfile(MEMORY_FILE)  # Windows
        except AttributeError:
            # Mac or Linux
            try:
                import subprocess, sys
                if sys.platform == "darwin":
                    subprocess.call(("open", MEMORY_FILE))
                else:
                    subprocess.call(("xdg-open", MEMORY_FILE))
            except Exception as e:
                messagebox.showinfo("Memory", f"Memory file located at: {os.path.abspath(MEMORY_FILE)}")

    def clear_memory(self):
        if messagebox.askyesno("Clear Memory", "Permanently clear conversation memory?"):
            self.core.clear_memory()
            self._refresh_chat_display_from_memory()

    def on_close(self):
        # Stop the TTS engine thread properly before closing the app
        if self.core.tts:
            self.core.tts.stop()
        self.destroy()


# ---------------------------
# Run
# ---------------------------
def main():
    app = VirtualAssistantApp()
    app.mainloop()

if __name__ == "__main__":
    main()

