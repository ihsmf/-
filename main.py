import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import parse, request


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INSTRUCTIONS_FILE = BASE_DIR / "instructions.txt"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Default Groq model.
# No GitHub Secret is required for this.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Telegram message limit.
# We use 4000 instead of 4096 to leave a safety margin.
TELEGRAM_MAX_MESSAGE_LENGTH = 4000

# Delay between Telegram messages.
TELEGRAM_MESSAGE_DELAY = 0.5


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    """Print a timestamped log message."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(
        f"[{now}] {message}",
        flush=True
    )


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

def get_required_env(name: str) -> str:
    """
    Get a required environment variable.

    Stops execution with a clear error if the variable
    is missing or empty.
    """

    value = os.getenv(name)

    if not value or not value.strip():
        raise RuntimeError(
            f"Required environment variable '{name}' "
            f"is not set or is empty."
        )

    return value.strip()


# ============================================================
# INSTRUCTIONS
# ============================================================

def read_instructions() -> str:
    """Read instructions.txt from the repository."""

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


# ============================================================
# TEXT PROCESSING
# ============================================================

def split_message(
    text: str,
    max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH
) -> list[str]:
    """
    Split long text into Telegram-compatible messages.

    The function tries to preserve paragraphs and lines
    before falling back to a hard split.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Paragraph fits into current chunk.
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_length:
            current = candidate
            continue

        # Save current chunk.
        if current:
            chunks.append(current)
            current = ""

        # Paragraph itself is too large.
        if len(paragraph) > max_length:

            lines = paragraph.split("\n")

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # Single line is too long.
                if len(line) > max_length:

                    # Save current chunk first.
                    if current:
                        chunks.append(current)
                        current = ""

                    # Hard split the long line.
                    for i in range(
                        0,
                        len(line),
                        max_length
                    ):
                        chunks.append(
                            line[i:i + max_length]
                        )

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
# GROQ API
# ============================================================

def generate_script(
    api_key: str,
    instructions: str
) -> str:
    """
    Generate the daily script using Groq.

    The model is intentionally hardcoded as a default
    so no GROQ_MODEL GitHub Secret is required.
    """

    model = DEFAULT_GROQ_MODEL

    today = datetime.now().strftime("%d.%m.%Y")

    user_prompt = f"""
Сегодня {today}.

Подготовь сегодняшний сценарий на основе системных инструкций.

КРИТИЧЕСКИ ВАЖНО:

1. Верни только готовый сценарий.
2. Не объясняй процесс создания сценария.
3. Не пиши мета-комментарии.
4. Не выдавай список идей вместо полноценного сценария.
5. Сценарий должен быть полностью готов для дальнейшего использования.
6. Соблюдай ВСЕ требования из системных инструкций.
7. Если в инструкциях указаны требования к структуре,
   длине, стилю, хуку, фактам, CTA, повествованию
   или оформлению — соблюдай их.
8. Пиши естественно и живо.
9. Избегай шаблонных фраз и ощущения текста,
   написанного нейросетью.
10. Не придумывай конкретные факты, цифры, события,
    цитаты или статистику, если это не разрешено
    системными инструкциями.
11. Сделай сценарий максимально сильным с точки зрения
    удержания внимания.
12. Не добавляй вступление вроде "Вот ваш сценарий".
13. Не добавляй заключительные комментарии после сценария.

Ответ должен содержать ТОЛЬКО финальный сценарий.
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

    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

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

    log(
        f"Sending request to Groq using model: {model}"
    )

    try:

        with request.urlopen(
            req,
            timeout=180
        ) as response:

            raw_response = response.read().decode(
                "utf-8"
            )

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

    # Handle Groq API errors.
    if "error" in result:

        error = result["error"]

        if isinstance(error, dict):

            message = error.get(
                "message",
                str(error)
            )

        else:
            message = str(error)

        raise RuntimeError(
            f"Groq API error: {message}"
        )

    try:

        content = result["choices"][0]["message"]["content"]

    except (
        KeyError,
        IndexError,
        TypeError
    ) as exc:

        raise RuntimeError(
            "Groq response does not contain "
            "a valid generated message."
        ) from exc

    if not content or not content.strip():

        raise RuntimeError(
            "Groq returned an empty script."
        )

    return content.strip()


# ============================================================
# TELEGRAM API
# ============================================================

def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str
) -> None:
    """Send one message to Telegram."""

    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }

    data = parse.urlencode(
        payload
    ).encode("utf-8")

    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type":
                "application/x-www-form-urlencoded",
            "User-Agent":
                "daily-script-generator/1.0"
        },
        method="POST"
    )

    try:

        with request.urlopen(
            req,
            timeout=60
        ) as response:

            raw_response = response.read().decode(
                "utf-8"
            )

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


# ============================================================
# SEND FULL SCRIPT
# ============================================================

def send_script_to_telegram(
    bot_token: str,
    chat_id: str,
    script: str
) -> None:
    """Send the complete generated script to Telegram."""

    chunks = split_message(script)

    if not chunks:
        raise RuntimeError(
            "Cannot send an empty script to Telegram."
        )

    log(
        f"Sending script to Telegram "
        f"in {len(chunks)} message(s)."
    )

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        send_telegram_message(
            bot_token=bot_token,
            chat_id=chat_id,
            text=chunk
        )

        log(
            f"Telegram message "
            f"{index}/{len(chunks)} sent."
        )

        if index < len(chunks):
            time.sleep(
                TELEGRAM_MESSAGE_DELAY
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    log("========================================")
    log("Daily AI Script Generator started")
    log("========================================")

    # --------------------------------------------------------
    # Get secrets
    # --------------------------------------------------------

    groq_api_key = get_required_env(
        "GROQ_API_KEY"
    )

    telegram_bot_token = get_required_env(
        "TELEGRAM_BOT_TOKEN"
    )

    telegram_chat_id = get_required_env(
        "TELEGRAM_CHAT_ID"
    )

    # --------------------------------------------------------
    # Read instructions
    # --------------------------------------------------------

    log(
        f"Reading instructions from: "
        f"{INSTRUCTIONS_FILE}"
    )

    instructions = read_instructions()

    log(
        f"Instructions loaded successfully "
        f"({len(instructions)} characters)."
    )

    # --------------------------------------------------------
    # Generate script
    # --------------------------------------------------------

    script = generate_script(
        api_key=groq_api_key,
        instructions=instructions
    )

    log(
        f"Script generated successfully "
        f"({len(script)} characters)."
    )

    # --------------------------------------------------------
    # Send to Telegram
    # --------------------------------------------------------

    send_script_to_telegram(
        bot_token=telegram_bot_token,
        chat_id=telegram_chat_id,
        script=script
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    log("========================================")
    log("DONE — script delivered to Telegram")
    log("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log("Execution interrupted.")
        sys.exit(130)

    except Exception as exc:

        log(f"ERROR: {exc}")
        sys.exit(1)
