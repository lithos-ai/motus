"""ask_user_question — built-in tool for structured user questions.

Based on Claude Code's AskUserQuestion schema. The LLM decides when to
call it; the tool internally calls interrupt() to hand control to the
user via the serve/router/frontend stack.

NOTE: The frontend automatically appends a free-text input ("Other") as
the last option of every question, allowing users to type a custom
answer. Do NOT include an "Other" option yourself — the frontend handles
it uniformly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from motus.tools.core.decorators import tool


class OptionModel(BaseModel):
    label: str = Field(..., description="Display text for the option (1-5 words)")
    description: str = Field(..., description="Explanation of this option")
    markdown: str | None = Field(
        None,
        description="Optional preview shown in a monospace box when this option is focused",
    )


class QuestionModel(BaseModel):
    question: str = Field(..., description="Complete question text, ending with '?'")
    header: str = Field(..., description="Short chip label, max 12 chars")
    options: list[OptionModel] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2-4 distinct choices (frontend auto-appends an 'Other' free-text field)",
    )
    multiSelect: bool = Field(
        False,
        description="Allow multiple selections (for non-mutually-exclusive choices)",
    )


class AskUserQuestionSchema(BaseModel):
    questions: list[QuestionModel] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="1-4 questions to ask the user",
    )


@tool(name="ask_user_question", schema=AskUserQuestionSchema)
async def ask_user_question(questions: list[dict]) -> dict[str, Any]:
    """Ask the user structured questions with options.

    Use when you need to:
    1. Gather user preferences or requirements
    2. Clarify ambiguous instructions
    3. Get decisions on implementation choices

    IMPORTANT: The frontend always appends a free-text input as the last
    option of every question, allowing the user to type a custom answer.
    Do not include an "Other" option yourself.
    """
    from motus.serve.interrupt import interrupt

    response = await interrupt(
        {
            "type": "user_input",
            "questions": questions,
        }
    )
    # response shape: {"answers": {question_text: answer_str}, ...}
    return response
