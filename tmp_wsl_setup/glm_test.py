#!/usr/bin/env python3
import os
import sys

from openai import OpenAI


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    api_key = require_env("ZAI_API_KEY")
    base_url = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    model = os.getenv("ZAI_MODEL", "glm-5.2")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Please reply with one short sentence in Chinese to confirm the GLM API is working."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
