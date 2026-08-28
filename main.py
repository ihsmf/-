import os
import sys
import time
from pathlib import Path
from datetime import datetime
from urllib import request, parse
import json
# ============================================================
# CONFIGURATION
# ============================================================
INSTRUCTIONS_FILE = Path(__file__).resolve().parent / "instructions.txt"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Можно изменить модель через GitHub Secret GROQ_MODEL.
# Если Secret не задан — используется эта модель.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Telegram ограничивает длину одного сообщения.
TELEGRAM_MAX_MESSAGE_LENGTH = 4000
# Небольшая пауза между сообщениями Telegram.
TELEGRAM_MESSAGE_DELAY = 0.5
# ============================================================
# HELPERS
# ============================================================
def log(message: str) -> None:
    """Simple timestamped logging."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)
def get_required_env(name: str) -> str:
    """Get required environment variable or stop execution."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )
    return value.strip()
def read_instructions() -> str:
    """Read the main AI instructions from instructions.txt."""
    if not INSTRUCTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Instructions file not found: {INSTRUCTIONS_FILE}"
        )
    instructions = INSTRUCTIONS_FILE.read_text(
        encoding="utf-8"
    ).strip()
    if not instructions:
        raise ValueError(
            "instructions.txt exists, but it is empty."
        )
    return instructions
def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH):
    """
    Split a long text into Telegram-compatible chunks.
    Tries to split by paragraphs first, then by lines,
    and finally hard-splits if necessary.
    """
    text = text.strip()
    if len(text) <= max_length:
        return [text]
    chunks = []
    current = ""
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )
        if len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        # If one paragraph itself is too large,
        # split it by lines.
        if len(paragraph) > max_length:
            lines = paragraph.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) > max_length:
                    # Final fallback: hard split.
                    for i in range(0, len(line), max_length):
                        chunks.append(line[i:i + max_length])
                else:
                    if not current:
                        current = line
                    elif len(current) + 1 + len(line) <= max_length:
                        current += "\n" + line
                    else:
                        chunks.append(current)
                        current = line
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
# ============================================================
# GROQ
# ============================================================
def generate_script(
    api_key: str,
    instructions: str,
    model: str
) -> str:
    """
    Generate the daily script using Groq Chat Completions API.
    """
    today = datetime.now().strftime("%d.%m.%Y")
    user_prompt = f"""
Сегодня {today}.
На основе системных инструкций подготовь сегодняшний сценарий.
КРИТИЧЕСКИ ВАЖНО:
1. Верни только готовый сценарий.
2. Не объясняй, как ты его создавал.
3. Не пиши мета-комментарии вроде "вот сценарий", "я предлагаю" и т.п.
4. Не выдавай список идей вместо сценария.
5. Сценарий должен быть полностью пригоден для дальнейшей публикации/озвучки.
6. Соблюдай ВСЕ требования из instructions.txt.
7. Если в инструкции есть требования к структуре, длине, стилю, хукам,
   фактам, CTA или оформлению — соблюдай их буквально.
8. Пиши естественно, интересно и без ощущения текста от ИИ.
9. Не добавляй неподтвержденные конкретные факты, цифры или события,
   если инструкция не разрешает их использовать.
10. Сделай материал максимально сильным с точки зрения удержания внимания.
Ответ должен содержать только финальный сценарий.
""".strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.7,
        "max_completion_tokens": 12000
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "daily-script-generator/1.0"
        },
        method="POST"
    )
    log(f"Sending request to Groq using model: {model}")
    try:
        with request.urlopen(req, timeout=180) as response:
            raw_response = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed: {exc}"
        ) from exc
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Groq returned invalid JSON."
        ) from exc
    # Handle API errors explicitly.
    if "error" in result:
        error = result["error"]
        if isinstance(error, dict):
            message = error.get("message", str(error))
        else:
            message = str(error)
        raise RuntimeError(
            f"Groq API error: {message}"
        )
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Groq response does not contain a valid message."
        ) from exc
    if not content or not content.strip():
        raise RuntimeError(
            "Groq returned an empty script."
        )
    return content.strip()
# ============================================================
# TELEGRAM
# ============================================================
def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str
) -> None:
    """Send one Telegram message."""
    url = (
        f"https://api.telegram.org/bot{bot_token}/sendMessage"
    )
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }
    data = parse.urlencode(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "daily-script-generator/1.0"
        },
        method="POST"
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            raw_response = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Telegram request failed: {exc}"
        ) from exc
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Telegram returned invalid JSON."
        ) from exc
    if not result.get("ok"):
        description = result.get(
            "description",
            "Unknown Telegram error"
        )
        raise RuntimeError(
            f"Telegram API error: {description}"
        )
def send_script_to_telegram(
    bot_token: str,
    chat_id: str,
    script: str
) -> None:
    """Send the complete script, splitting it if necessary."""
    chunks = split_message(script)
    log(
        f"Sending script to Telegram in {len(chunks)} message(s)."
    )
    for index, chunk in enumerate(chunks, start=1):
        send_telegram_message(
            bot_token=bot_token,
            chat_id=chat_id,
            text=chunk
        )
        log(
            f"Telegram message {index}/{len(chunks)} sent."
        )
        if index < len(chunks):
            time.sleep(TELEGRAM_MESSAGE_DELAY)
# ============================================================
# MAIN
# ============================================================
def main() -> None:
    log("========================================")
    log("Daily AI Script Generator started")
    log("========================================")
    # Required secrets.
    groq_api_key = get_required_env("GROQ_API_KEY")
    telegram_bot_token = get_required_env("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = get_required_env("TELEGRAM_CHAT_ID")
    # Optional model override.
    groq_model = os.getenv(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL
    ).strip()
    # Read instructions.
    log(f"Reading instructions from: {INSTRUCTIONS_FILE}")
    instructions = read_instructions()
    log(
        f"Instructions loaded successfully "
        f"({len(instructions)} characters)."
    )
    # Generate script.
    script = generate_script(
        api_key=groq_api_key,
        instructions=instructions,
        model=groq_model
    )
    log(
        f"Script generated successfully "
        f"({len(script)} characters)."
    )
    # Send to Telegram.
    send_script_to_telegram(
        bot_token=telegram_bot_token,
        chat_id=telegram_chat_id,
        script=script
    )
    log("========================================")
    log("DONE — script delivered to Telegram")
    log("========================================")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Execution interrupted.")
        sys.exit(130)
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
