"""
Chatbot — command-line entry point.

Usage:
    python -m chatbot.motus_native
"""

import asyncio

from .serve import agent


async def session():
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting session.")
            break
        result = await agent(user_input)
        print(f"MotusBot: {result}")


if __name__ == "__main__":
    asyncio.run(session())
