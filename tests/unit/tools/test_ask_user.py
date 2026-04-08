import pytest


def test_ask_user_question_schema_exists():
    from pydantic import BaseModel

    from motus.tools.builtins.ask_user import AskUserQuestionSchema

    assert issubclass(AskUserQuestionSchema, BaseModel)


def test_ask_user_question_schema_validates_valid_input():
    from motus.tools.builtins.ask_user import AskUserQuestionSchema

    valid = {
        "questions": [
            {
                "question": "Which library?",
                "header": "Library",
                "multiSelect": False,
                "options": [
                    {"label": "datetime", "description": "stdlib"},
                    {"label": "arrow", "description": "third-party"},
                ],
            }
        ]
    }
    # Should not raise
    AskUserQuestionSchema.model_validate(valid)


def test_ask_user_question_schema_rejects_too_many_options():
    import pydantic

    from motus.tools.builtins.ask_user import AskUserQuestionSchema

    invalid = {
        "questions": [
            {
                "question": "Pick one",
                "header": "Choice",
                "multiSelect": False,
                "options": [{"label": str(i), "description": ""} for i in range(5)],
            }
        ]
    }
    with pytest.raises(pydantic.ValidationError):
        AskUserQuestionSchema.model_validate(invalid)


def test_ask_user_question_tool_registered():
    from motus.tools.builtins.ask_user import ask_user_question

    # Should be a callable with a __tool_name__ attribute (set by @tool)
    assert getattr(ask_user_question, "__tool_name__", None) == "ask_user_question"
