import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, parse, request


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INSTRUCTIONS_FILE = BASE_DIR / "instructions.txt"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fixed model.
# No GROQ_MODEL secret is required.
GROQ_MODEL = "openai/gpt-oss-120b"

# Groq Free Tier has a relatively small token-per-minute limit.
# We intentionally keep requests conservative.
SAFE_TOTAL_TOKENS = 7200

# Maximum visible + reasoning completion tokens.
INITIAL_MAX_COMPLETION_TOKENS = 4000

# Never allow generation to become ridiculously short.
MIN_COMPLETION_TOKENS = 1800

# Number of retries for temporary API problems.
MAX_GROQ_RETRIES = 4

# Telegram message limit.
TELEGRAM_MAX_MESSAGE_LENGTH = 4000

# Telegram retry count.
MAX_TELEGRAM_RETRIES = 4

# Delay between Telegram chunks.
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
# ENVIRONMENT
# ============================================================

def get_required_env(name: str) -> str:
    """Get a required environment variable."""

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
    """Read instructions.txt."""

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
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.

    For mixed Russian/English text, characters / 4 is a
    reasonable conservative approximation for this use case.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4
    )


def calculate_max_completion_tokens(
    instructions: str
) -> int:
    """
    Calculate a safe completion size.

    Keeps input + output comfortably below the conservative
    Groq Free Tier token budget.
    """

    estimated_input_tokens = (
        estimate_tokens(instructions)
        + 350
    )

    available = (
        SAFE_TOTAL_TOKENS
        - estimated_input_tokens
    )

    calculated = min(
        INITIAL_MAX_COMPLETION_TOKENS,
        available
    )

    calculated = max(
        MIN_COMPLETION_TOKENS,
        calculated
    )

    log(
        "Estimated input tokens: "
        f"{estimated_input_tokens}"
    )

    log(
        "Max completion tokens: "
        f"{calculated}"
    )

    return calculated


# ============================================================
# GROQ ERROR HELPERS
# ============================================================

def extract_http_error(
    exc: error.HTTPError
) -> tuple[int, str]:
    """Read status code and response body from HTTPError."""

    status = exc.code

    try:
        body = exc.read().decode(
            "utf-8",
            errors="replace"
        )
    except Exception:
        body = ""

    return status, body


def parse_error_message(body: str) -> str:
    """Extract a readable API error message."""

    if not body:
        return "No response body."

    try:
        data = json.loads(body)

        if isinstance(data, dict):

            err = data.get("error")

            if isinstance(err, dict):
                return err.get(
                    "message",
                    json.dumps(err, ensure_ascii=False)
                )

            if isinstance(err, str):
                return err

    except json.JSONDecodeError:
        pass

    return body[:1000]


# ============================================================
# GROQ REQUEST
# ============================================================

def make_groq_request(
    api_key: str,
    instructions: str,
    max_completion_tokens: int
) -> str:
    """
    Make one Groq request.

    Raises HTTPError when Groq rejects the request.
    """

    today = datetime.now().strftime("%d.%m.%Y")

    user_prompt = f"""
Сегодня {today}.

Создай сегодняшний готовый сценарий строго по системным
инструкциям.

КРИТИЧЕСКИЕ ТРЕБОВАНИЯ:

- Верни только финальный сценарий.
- Не объясняй процесс создания.
- Не добавляй мета-комментарии.
- Не выдавай список идей вместо сценария.
- Соблюдай все требования из instructions.txt.
- Сценарий должен быть полностью готов к использованию.
- Пиши естественно, живо и профессионально.
- Не придумывай конкретные факты, цифры, события или
  статистику, если это не разрешено инструкциями.
- Не добавляй фразу "Вот ваш сценарий".
- Не добавляй комментарии после сценария.

Ответ должен содержать ТОЛЬКО сценарий.
""".strip()

    payload = {
        "model": GROQ_MODEL,

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

        "max_completion_tokens": max_completion_tokens,

        # GPT-OSS supports low / medium / high.
        # Low is enough for a script-generation task and
        # helps keep token usage predictable.
        "reasoning_effort": "low"
    }

    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")

    log(
        f"Sending request to Groq using model: "
        f"{GROQ_MODEL}"
    )

    log(
        f"Request body size: {len(data)} bytes"
    )

    req = request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "daily-script-generator/2.0"
        },
        method="POST"
    )

    with request.urlopen(
        req,
        timeout=180
    ) as response:

        raw_response = response.read().decode(
            "utf-8",
            errors="replace"
        )

    try:
        result = json.loads(raw_response)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON."
        ) from exc

    if "error" in result:

        message = parse_error_message(
            raw_response
        )

        raise RuntimeError(
            f"Groq API error: {message}"
        )

    try:

        content = (
            result["choices"][0]
            ["message"]["content"]
        )

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
# GROQ WITH RETRIES / FALLBACKS
# ============================================================

def generate_script(
    api_key: str,
    instructions: str
) -> str:

    max_completion_tokens = (
        calculate_max_completion_tokens(
            instructions
        )
    )

    # Fallback sizes for 413 / oversized requests.
    fallback_sizes = [
        max_completion_tokens,
        3500,
        3000,
        2500,
        2000,
    ]

    # Remove duplicates and keep values >= minimum.
    sizes = []

    for value in fallback_sizes:

        value = max(
            MIN_COMPLETION_TOKENS,
            value
        )

        if value not in sizes:
            sizes.append(value)

    last_error = None

    for attempt in range(
        1,
        MAX_GROQ_RETRIES + 1
    ):

        current_size = sizes[
            min(
                attempt - 1,
                len(sizes) - 1
            )
        ]

        log(
            f"Groq attempt "
            f"{attempt}/{MAX_GROQ_RETRIES} "
            f"with max_completion_tokens="
            f"{current_size}"
        )

        try:

            return make_groq_request(
                api_key=api_key,
                instructions=instructions,
                max_completion_tokens=current_size
            )

        except error.HTTPError as exc:

            status, body = extract_http_error(
                exc
            )

            message = parse_error_message(
                body
            )

            last_error = (
                f"HTTP {status}: {message}"
            )

            # ------------------------------------------------
            # 413 — Payload Too Large
            # ------------------------------------------------

            if status == 413:

                log(
                    "Groq returned HTTP 413 "
                    "(Payload Too Large)."
                )

                if attempt < MAX_GROQ_RETRIES:

                    log(
                        "Reducing completion size "
                        "and retrying..."
                    )

                    time.sleep(1)

                    continue

                break

            # ------------------------------------------------
            # 429 — Rate limit
            # ------------------------------------------------

            if status == 429:

                retry_after = (
                    exc.headers.get(
                        "Retry-After"
                    )
                    if exc.headers
                    else None
                )

                try:
                    wait_seconds = int(
                        retry_after
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    wait_seconds = 10

                wait_seconds = min(
                    max(wait_seconds, 3),
                    60
                )

                log(
                    f"Groq rate limit reached. "
                    f"Waiting {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            # ------------------------------------------------
            # Temporary server errors
            # ------------------------------------------------

            if status in {
                500,
                502,
                503,
                504
            }:

                wait_seconds = min(
                    2 ** attempt,
                    30
                )

                log(
                    f"Temporary Groq error "
                    f"{status}. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            # ------------------------------------------------
            # Permanent client errors
            # ------------------------------------------------

            raise RuntimeError(
                f"Groq API returned HTTP "
                f"{status}: {message}"
            ) from exc

        except error.URLError as exc:

            last_error = (
                f"Network error: {exc}"
            )

            if attempt < MAX_GROQ_RETRIES:

                wait_seconds = min(
                    2 ** attempt,
                    30
                )

                log(
                    f"Network error. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            break

        except TimeoutError as exc:

            last_error = (
                f"Timeout: {exc}"
            )

            if attempt < MAX_GROQ_RETRIES:

                time.sleep(
                    min(2 ** attempt, 30)
                )

                continue

            break

        except Exception as exc:

            raise RuntimeError(
                f"Unexpected Groq error: {exc}"
            ) from exc

    raise RuntimeError(
        "Groq request failed after all retries. "
        f"Last error: {last_error}"
    )


# ============================================================
# TELEGRAM
# ============================================================

def split_message(
    text: str,
    max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH
) -> list[str]:
    """
    Split long text into Telegram-compatible chunks.
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

        if len(paragraph) <= max_length:

            current = paragraph
            continue

        # Long paragraph: split by lines.
        lines = paragraph.split("\n")

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if len(line) <= max_length:

                if not current:

                    current = line

                elif (
                    len(current)
                    + 1
                    + len(line)
                    <= max_length
                ):

                    current += "\n" + line

                else:

                    chunks.append(current)
                    current = line

            else:

                if current:

                    chunks.append(current)
                    current = ""

                # Hard split extremely long lines.
                for i in range(
                    0,
                    len(line),
                    max_length
                ):

                    chunks.append(
                        line[
                            i:i + max_length
                        ]
                    )

    if current:
        chunks.append(current)

    return chunks


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str
) -> None:
    """Send one Telegram message."""

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
                "daily-script-generator/2.0"
        },
        method="POST"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_TELEGRAM_RETRIES + 1
    ):

        try:

            with request.urlopen(
                req,
                timeout=60
            ) as response:

                raw_response = (
                    response.read()
                    .decode(
                        "utf-8",
                        errors="replace"
                    )
                )

            result = json.loads(
                raw_response
            )

            if not result.get("ok"):

                description = result.get(
                    "description",
                    "Unknown Telegram error"
                )

                raise RuntimeError(
                    f"Telegram API error: "
                    f"{description}"
                )

            return

        except error.HTTPError as exc:

            status, body = (
                extract_http_error(exc)
            )

            message = parse_error_message(
                body
            )

            last_error = (
                f"HTTP {status}: {message}"
            )

            # Telegram 429 should be retried.
            if status == 429:

                wait_seconds = 5

                try:

                    error_data = json.loads(
                        body
                    )

                    wait_seconds = int(
                        error_data
                        .get("parameters", {})
                        .get(
                            "retry_after",
                            5
                        )
                    )

                except Exception:
                    pass

                wait_seconds = min(
                    max(wait_seconds, 1),
                    60
                )

                log(
                    f"Telegram rate limit. "
                    f"Waiting {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if status in {
                500,
                502,
                503,
                504
            }:

                time.sleep(
                    min(2 ** attempt, 30)
                )

                continue

            raise RuntimeError(
                f"Telegram API returned HTTP "
                f"{status}: {message}"
            ) from exc

        except (
            error.URLError,
            TimeoutError
        ) as exc:

            last_error = str(exc)

            if attempt < MAX_TELEGRAM_RETRIES:

                time.sleep(
                    min(2 ** attempt, 30)
                )

                continue

    raise RuntimeError(
        "Telegram request failed after "
        f"{MAX_TELEGRAM_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def send_script_to_telegram(
    bot_token: str,
    chat_id: str,
    script: str
) -> None:
    """Send the entire script."""

    chunks = split_message(script)

    if not chunks:

        raise RuntimeError(
            "Generated script is empty."
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
    # Secrets
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
    # Instructions
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
    # Generate
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
    # Telegram
    # --------------------------------------------------------

    send_script_to_telegram(
        bot_token=telegram_bot_token,
        chat_id=telegram_chat_id,
        script=script
    )

    # --------------------------------------------------------
    # Done
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

        log(
            f"ERROR: {exc}"
        )

        sys.exit(1)
