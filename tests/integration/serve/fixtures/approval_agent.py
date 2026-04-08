"""Fake agent that exercises interrupt/resume HITL flows directly."""


async def fake_agent(message, state):
    from motus.models import ChatMessage

    content = (message.content or "").lower()

    if "delete" in content:
        from motus.serve.interrupt import interrupt

        decision = await interrupt(
            {
                "type": "tool_approval",
                "tool_name": "delete_file",
                "tool_args": {"path": "/tmp/test.txt"},
            }
        )
        if decision.get("approved"):
            result = "Deleted /tmp/test.txt"
        else:
            result = "User rejected delete_file"
        response = ChatMessage.assistant_message(content=result)

    elif "ask" in content:
        from motus.serve.interrupt import interrupt

        answer = await interrupt(
            {
                "type": "user_input",
                "questions": [
                    {
                        "question": "Which option?",
                        "header": "Option",
                        "multiSelect": False,
                        "options": [
                            {"label": "A", "description": "Option A"},
                            {"label": "B", "description": "Option B"},
                        ],
                    }
                ],
            }
        )
        response = ChatMessage.assistant_message(content=f"You chose: {answer}")

    else:
        response = ChatMessage.assistant_message(content="No action taken.")

    return response, state + [message, response]
