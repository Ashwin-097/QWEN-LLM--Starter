# =====================================================================
# IMPORTS: Bringing in Python standard libraries & ML dependencies
# =====================================================================
import configparser  # Reads key-value pairs from .ini settings files
import asyncio       # Enables asynchronous execution for streaming APIs
import gc            # Garbage Collector to clear unreferenced Python objects
import json          # Handles persistent file-based JSON long-term memory storage
import os            # System utility to handle local paths and URIs
import re            # Regex filtering for simple rule-based memory extraction
import sys           # Provides access to system-specific functions (like stderr)
import time          # Used to measure time taken for benchmarks
from typing import Union  # Type hinting for multi-type parameter inputs

import torch         # PyTorch core library for tensor computations and GPU management
from PIL import Image # For direct image object manipulation
from qwen_vl_utils import process_vision_info  # Handles vision feature extraction for Qwen3-VL

from transformers import (
    AutoProcessor,                    # Handles text tokenization & vision preprocessing for Qwen-VL
    BitsAndBytesConfig,               # Sets up 4-bit/8-bit GPU quantization parameters
    Qwen3VLForConditionalGeneration,  # The actual Qwen Vision-Language model class
    TextIteratorStreamer,             # Allows thread-safe token streaming (used for async API)
    TextStreamer,                     # Streams generated tokens directly to stdout/terminal in real-time
)

# =====================================================================
# CONFIGURATION PARSING: Reading settings from config.ini
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")  # Load settings file into memory

# Extract file paths & aliases
MODEL_PATH = config.get("PATHS", "model_path", fallback="Qwen/Qwen3-VL-4B-Instruct")
MODEL_ALIAS = config.get("PATHS", "model_alias", fallback="qwen3-vl")

# Hardware acceleration toggle
ALLOW_TF32 = config.getboolean("HARDWARE", "allow_tf32", fallback=True)

# Standard generation parameters
DEFAULT_MAX_TOKENS = config.getint("GENERATION", "default_max_tokens", fallback=2048)
DEFAULT_TEMPERATURE = config.getfloat("GENERATION", "default_temperature", fallback=0.7)
DEFAULT_TOP_P = config.getfloat("GENERATION", "default_top_p", fallback=0.9)
DEFAULT_REPETITION_PENALTY = config.getfloat("GENERATION", "default_repetition_penalty", fallback=1.05)

# Context window size limit
MAX_CONTEXT_TURNS = 20

# Map human-readable text strings from INI to PyTorch data types
dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Construct quantization configuration dictionary for bitsandbytes
QUANTIZATION_CONFIG = {
    "load_in_4bit": config.getboolean("QUANTIZATION", "load_in_4bit", fallback=True),
    "bnb_4bit_compute_dtype": dtype_map.get(
        config.get("QUANTIZATION", "bnb_4bit_compute_dtype", fallback="float16"),
        torch.float16,
    ),
    "bnb_4bit_quant_type": config.get("QUANTIZATION", "bnb_4bit_quant_type", fallback="nf4"),
    "bnb_4bit_use_double_quant": config.getboolean("QUANTIZATION", "bnb_4bit_use_double_quant", fallback=True),
}

# Global variables to hold loaded model, processor, and memory instances in RAM
model = None
processor = None
memory = None


# =====================================================================
# 10,000-FACT LOCAL HUMAN MEMORY ENGINE (Zero External Dependencies)
# =====================================================================
class LocalHumanMemory:
    """Manages up to 10,000 persistent facts stored locally in JSON with fast keyword retrieval."""

    def __init__(self, memory_filepath: str = "user_memories.json", max_facts: int = 10000):
        self.filepath = memory_filepath
        self.max_facts = max_facts
        self.memories = self._load()

    def _load(self) -> dict:
        """Loads structured memory store from local disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        """Persists structured memory store to local disk."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save long-term memory file: {e}", file=sys.stderr)

    def extract_and_store(self, user_id: str, content: Union[str, list]):
        """Rule-based extractor storing up to 10,000 facts per user."""
        if user_id not in self.memories:
            self.memories[user_id] = []

        text_content = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_name = os.path.basename(str(item.get("image")))
                    fact = f"User shared an image file: '{img_name}'"
                    if fact not in self.memories[user_id]:
                        self.memories[user_id].append(fact)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_content += " " + item.get("text", "")
        else:
            text_content = str(content)

        patterns = [
            (r"\bmy name is ([A-Za-z0-9_\s]+)", "User's name is {0}"),
            (r"\bi am ([A-Za-z0-9_\s]+)", "User is {0}"),
            (r"\bi work as a? ([A-Za-z0-9_\s]+)", "User works as a {0}"),
            (r"\bi like ([A-Za-z0-9_\s]+)", "User enjoys {0}"),
            (r"\bi prefer ([A-Za-z0-9_\s]+)", "User prefers {0}"),
            (r"\bmy favorite ([A-Za-z0-9_\s]+) is ([A-Za-z0-9_\s]+)", "User's favorite {0} is {1}"),
            (r"\bi use ([A-Za-z0-9_\s]+)", "User uses {0}"),
            (r"\bi have a? ([A-Za-z0-9_\s]+)", "User has {0}"),
            (r"\bi live in ([A-Za-z0-9_\s]+)", "User lives in {0}"),
        ]

        for pattern, template in patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    fact = template.format(*[x.strip() for x in m])
                else:
                    fact = template.format(m.strip())

                if fact not in self.memories[user_id]:
                    self.memories[user_id].append(fact)

        # Ensure memory stays capped at the 10,000 fact limit
        if len(self.memories[user_id]) > self.max_facts:
            self.memories[user_id] = self.memories[user_id][-self.max_facts:]

        self._save()

    def get_context_block(self, user_id: str, query_text: str = "", max_prompt_facts: int = 20) -> str:
        """
        Retrieves relevant facts from up to 10,000 stored memories.
        Uses keyword relevance matching so prompt length remains fast and lightweight.
        """
        user_facts = self.memories.get(user_id, [])
        if not user_facts:
            return ""

        # If total facts are small, send all of them
        if len(user_facts) <= max_prompt_facts:
            selected_facts = user_facts
        else:
            # Score facts based on word overlap with current user input
            query_words = set(re.findall(r"\w+", query_text.lower()))
            scored_facts = []

            for idx, fact in enumerate(user_facts):
                fact_words = set(re.findall(r"\w+", fact.lower()))
                overlap = len(query_words.intersection(fact_words))
                # Recency bonus for newer facts
                recency_score = idx / len(user_facts)
                total_score = (overlap * 2.0) + recency_score
                scored_facts.append((total_score, fact))

            # Sort by score and take top matches
            scored_facts.sort(key=lambda x: x[0], reverse=True)
            selected_facts = [fact for score, fact in scored_facts[:max_prompt_facts]]

        formatted = "\n".join([f"- {fact}" for fact in selected_facts])
        return (
            "\n\n[Known Long-Term Facts & Context About the User]:\n"
            f"{formatted}\n"
            "(Integrate these details naturally when relevant, as a real human friend would.)"
        )


# =====================================================================
# MODEL INITIALIZATION & VRAM MANAGEMENT
# =====================================================================
def load_model():
    """Loads processor, quantized model, and local JSON memory."""
    global model, processor, memory

    print("🧠 Initializing 10,000-Fact Local Memory Store...")
    memory = LocalHumanMemory(memory_filepath="user_memories.json", max_facts=10000)

    if ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading processor from {MODEL_PATH}...")
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)

    print("Loading model weights into GPU...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
        device_map="auto",
    )
    print("✅ Model & 10,000-Fact Memory System initialized successfully!\n")


def cleanup_vram():
    """Garbage collects memory and flushes CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# PROMPT & VISION INPUT FORMATTING
# =====================================================================
def format_image_source(image_input: Union[str, Image.Image]) -> Union[str, Image.Image]:
    """Ensures local file paths become valid URIs while preserving PIL Images and URLs."""
    if isinstance(image_input, str):
        if not image_input.startswith(("http://", "https://", "file://")) and os.path.exists(image_input):
            return f"file://{os.path.abspath(image_input)}"
    return image_input


def build_multimodal_inputs(messages: list, user_id: str = "default_user"):
    """Formats turn messages, injects long-term memory, and extracts visual inputs for Qwen3-VL."""
    formatted = []

    # Extract latest user message string to perform relevance matching against 10k facts
    latest_query = ""
    if messages:
        last_turn = messages[-1]
        raw_content = last_turn.get("content", "") if isinstance(last_turn, dict) else getattr(last_turn, "content", "")
        if isinstance(raw_content, list):
            latest_query = " ".join([i.get("text", "") for i in raw_content if isinstance(i, dict) and i.get("type") == "text"])
        else:
            latest_query = str(raw_content)

    # Dynamically select relevant facts out of 10,000
    memory_context = memory.get_context_block(user_id, query_text=latest_query) if memory else ""

    # Human-like conversational persona system prompt
    system_instruction = (
        "You are an empathetic, sharp, and natural AI collaborator. "
        "Interact like a knowledgeable peer. Use relevant past details about the user smoothly "
        "without awkwardly citing 'stored records', 'my files', or 'memory banks'."
        f"{memory_context}"
    )

    formatted.append({"role": "system", "content": system_instruction})

    # Recent turns window (20 turns)
    for msg in messages[-MAX_CONTEXT_TURNS:]:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "user")
        else:
            content = getattr(msg, "content", "")
            role = getattr(msg, "role", "user")

        if role == "system":
            continue

        if isinstance(content, list):
            item_list = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_src = format_image_source(item.get("image"))
                        item_list.append({"type": "image", "image": image_src})
                    elif item.get("type") == "text":
                        item_list.append({"type": "text", "text": item.get("text", "")})
            formatted.append({"role": role, "content": item_list})
        else:
            formatted.append({"role": role, "content": str(content)})

    text_prompt = processor.apply_chat_template(
        formatted, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(formatted)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    return inputs


# =====================================================================
# INFERENCE: CLI/Terminal Mode (Synchronous with Metrics)
# =====================================================================
def generate_cli(conversation_history: list, user_id: str = "default_user"):
    """Executes non-async inference for direct terminal interactions with memory extraction."""
    inputs = build_multimodal_inputs(conversation_history, user_id=user_id)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    start_time = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_TOKENS,
            streamer=streamer,
            use_cache=True,
            do_sample=True,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    end_time = time.time()

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    generated_tokens = generated_ids.shape[0]
    elapsed = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0

    reply = processor.decode(generated_ids, skip_special_tokens=True).strip()

    print(
        f"\n\n⏱ Time: {elapsed:.2f}s | ⚡ Speed: {tokens_per_sec:.2f} tok/sec\n"
        + "-" * 50
    )

    # Automatically extract long-term memory from last user input
    if conversation_history:
        last_turn = conversation_history[-1]
        user_content = last_turn.get("content", "") if isinstance(last_turn, dict) else getattr(last_turn, "content", "")
        memory.extract_and_store(user_id, user_content)

    cleanup_vram()
    return reply


# =====================================================================
# INFERENCE: API Server Mode (Asynchronous Streaming)
# =====================================================================
async def generate_stream_api(messages: list, user_id: str = "default_user", max_tokens: int = None, temperature: float = None):
    """Executes async streaming generation for API endpoints with local memory persistence."""
    inputs = build_multimodal_inputs(messages, user_id=user_id)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "do_sample": True,
        "temperature": temperature or DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "streamer": streamer,
    }

    loop = asyncio.get_running_loop()

    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as e:
            print(f"Generation error: {e}", file=sys.stderr)

    task = loop.run_in_executor(None, run_generation)

    for token in streamer:
        yield token

    await task

    if messages:
        last_turn = messages[-1]
        user_content = last_turn.get("content", "") if isinstance(last_turn, dict) else getattr(last_turn, "content", "")
        memory.extract_and_store(user_id, user_content)

    cleanup_vram()


# =====================================================================
# TERMINAL INTERACTIVE LOOP
# =====================================================================
def run_cli_chat():
    """Terminal chat execution loop with local memory support."""
    load_model()
    print("\n🚀 Qwen3-VL (Terminal Chat) is ready! Type 'exit' to quit.")
    print("💡 Tip: Type 'image: <path_to_image>' followed by your question to analyze an image.\n" + "=" * 50)

    history = []
    current_user = "user_alex"

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_text:
            continue
        if user_text.lower() == "exit":
            print("Goodbye!")
            break

        if user_text.lower().startswith("image:"):
            parts = user_text[6:].strip().split(" ", 1)
            img_path = parts[0]
            prompt = parts[1] if len(parts) > 1 else "Describe this image."
            content_payload = [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]
        else:
            content_payload = user_text

        history.append({"role": "user", "content": content_payload})
        print("Qwen: ", end="", flush=True)

        reply = generate_cli(history, user_id=current_user)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_cli_chat()