"""
Abstract base class for agents in the Motus framework.
"""

import inspect
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel

from motus import Tools, normalize_tools
from motus.memory import BasicMemory, CompactionMemory, DatabaseMemory, MemoryScope
from motus.memory.base_memory import BaseMemory
from motus.models import BaseChatClient, ChatMessage, ReasoningConfig, ToolDefinition
from motus.runtime.agent_task import agent_task
from motus.runtime.types import AGENT_CALL
from motus.tools.core.tool import Tool

# Type variable for the return type of __call__
T = TypeVar("T")


class AgentBase(ABC, Generic[T]):
    """
    Abstract base class for agents.

    Provides common functionality for:
    - Managing conversation history via Memory interface
    - Handling tools and tool schemas
    - Client and model configuration
    - System prompt management
    - Short-term and long-term memory capabilities

    Subclasses must implement the `_run` method which contains
    the core agent logic (e.g., ReAct loop, chain-of-thought, etc.)
    """

    def __init__(
        self,
        client: BaseChatClient,
        model_name: str,
        name: str | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        response_format: Optional[type[BaseModel]] = None,
        max_steps: Optional[int] = None,
        memory_type: Literal["basic", "compact", "database"] = "basic",
        memory: Optional[BaseMemory] = None,
        input_guardrails: Optional[list[Callable]] = None,
        output_guardrails: Optional[list[Callable]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
    ) -> None:
        """
        Initialize the agent.

        Args:
            client: The chat client to use for LLM calls
            model_name: The model identifier (e.g., "gpt-4", "claude-3-opus")
            name: Optional agent name. If not provided, inferred from the
                  caller's variable name on first __call__, or falls back
                  to the class name.
            system_prompt: Optional system prompt to set agent behavior
            tools: Optional tools (can be Tools, dict, list, or callable)
            max_steps: Maximum number of reasoning steps (default: None, no limit)
            memory_type: Memory type - "basic" (default), "compact", or "database"
            memory: Optional BaseMemory instance for conversation and memory management.
                   If not provided, a new Memory instance will be created.
                   Can be any BaseMemory subclass (e.g., Memory, CompactionMemory).
            input_guardrails: Optional list of guardrail callables run before the
                agent's ``_run()`` method.  Signature: ``(value: str) -> str | None``
                or ``(value: str, agent) -> str | None``.  Return ``None`` to pass
                through, return ``str`` to replace, or raise to block.
            output_guardrails: Optional list of guardrail callables run after
                ``_run()``.  For plain string results: same signature as input
                guardrails.  For structured output (``response_format`` set):
                declare fields from the BaseModel — ``(field: T, ...) -> dict | None``
                with optional ``agent`` parameter.
        """
        self._client = client
        self._model_name = model_name
        self._name = name  # None means "infer later"
        self._system_prompt = system_prompt
        self._tools: Optional[Tools] = tools
        self._response_format = response_format
        self._max_steps = max_steps
        self._memory_type = memory_type
        self._reasoning = reasoning
        self._input_guardrails = (
            input_guardrails if input_guardrails is not None else []
        )
        self._output_guardrails = (
            output_guardrails if output_guardrails is not None else []
        )

        # Initialize memory before _init_tools() since it may access _memory
        if memory is not None:
            self._memory = memory
        elif memory_type == "basic":
            self._memory = BasicMemory()
        elif memory_type == "compact":
            self._memory = CompactionMemory()
        elif memory_type == "database":
            scope = MemoryScope(
                session_id=str(uuid.uuid4()),
                agent_id=self._name or self.__class__.__name__,
            )
            self._memory = DatabaseMemory(
                scope=scope,
                model_name=model_name,
                system_prompt=self._system_prompt or "",
            )
        else:
            raise ValueError(
                f"Unknown memory_type={memory_type!r}. "
                "Supported values: 'basic', 'compact', 'database'."
            )

        # Inject model/client if the memory supports it
        if hasattr(self._memory, "set_model"):
            self._memory.set_model(
                client=self._client,
                model_name=self._model_name,
            )
        self._memory.set_system_prompt(self._system_prompt or "")

        self._init_tools()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def client(self) -> BaseChatClient:
        """The chat client used for LLM calls."""
        return self._client

    @property
    def model_name(self) -> str:
        """The model identifier."""
        return self._model_name

    @property
    def name(self) -> str:
        """The name of the agent.

        Resolved lazily: explicit name > inferred variable name > class name.
        """
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    @property
    def max_steps(self) -> int:
        """Maximum number of reasoning steps."""
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Set maximum number of reasoning steps."""
        self._max_steps = value

    @property
    def system_prompt(self) -> Optional[str]:
        """The system prompt for the agent."""
        return self.memory.construct_system_prompt().content

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]) -> None:
        """Set the system prompt."""
        self._system_prompt = value
        # Update memory's system prompt as well
        self._memory.set_system_prompt(value or "")

    @property
    def tools(self) -> Optional[Tools]:
        """The tools available to the agent."""
        return self._tools

    @tools.setter
    def tools(self, value: Optional[Any]) -> None:
        """Set the tools (will be normalized)."""
        self._tools = normalize_tools(value) if value else None

    @property
    def memory(self) -> BaseMemory:
        """The BaseMemory instance managing conversation and memory."""
        return self._memory

    @property
    def messages(self) -> List[ChatMessage]:
        """The conversation history (working memory)."""
        return self._memory.get_context()

    @property
    def response_format(self) -> Optional[type[BaseModel]]:
        """The response format for the agent."""
        return self._response_format

    # -------------------------------------------------------------------------
    # Message Management (delegated to Memory)
    # -------------------------------------------------------------------------

    async def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the conversation history.

        Args:
            message: The ChatMessage to add
        """
        await self._memory.add_message(message)

    async def add_user_message(
        self, content: str, base64_image: Optional[str] = None
    ) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The user message content
            base64_image: Optional base64 encoded image
        """
        await self._memory.add_message(ChatMessage.user_message(content, base64_image))

    async def add_assistant_message(
        self, content: Optional[str] = None, tool_calls: Optional[list] = None
    ) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The assistant message content
            tool_calls: Optional tool calls made by the assistant
        """
        await self._memory.add_message(
            ChatMessage.assistant_message(content, tool_calls)
        )

    async def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        name: str,
        base64_image: Optional[str] = None,
    ) -> None:
        """
        Add a tool result message to the conversation.

        Args:
            content: The tool result content
            tool_call_id: The ID of the tool call this responds to
            name: The name of the tool
            base64_image: Optional base64 encoded image
        """
        await self._memory.add_message(
            ChatMessage.tool_message(content, tool_call_id, name, base64_image)
        )

    def clear_messages(self) -> None:
        """Clear all messages from the conversation history."""
        self._memory.clear_messages()

    def reset(self) -> None:
        """
        Reset the agent to its initial state.

        Clears messages and re-adds the system prompt if one exists.
        """
        self._memory.reset()

    def get_context(self) -> List[ChatMessage]:
        """
        Get the full context for the current conversation.

        This includes the system prompt with memory context prepended
        to the working memory messages.

        Returns:
            List of messages ready to be sent to the LLM
        """
        return self._memory.get_context()

    # -------------------------------------------------------------------------
    # Tool Utilities
    # -------------------------------------------------------------------------

    def tools_schema(self) -> Optional[List[ToolDefinition]]:
        """
        Get tools as a list of ToolDefinition objects.

        Returns:
            List of ToolDefinition or None if no tools are configured
        """
        if self._tools is None:
            return None
        return [
            ToolDefinition(
                name=name,
                description=tool.description or "",
                parameters=tool.json_schema,
            )
            for name, tool in self._tools.items()
        ]

    # -------------------------------------------------------------------------
    # Agent Operations
    # -------------------------------------------------------------------------

    def fork(self) -> "AgentBase[T]":
        """
        Create a copy of this agent with the same configuration and history.

        Note: The memory is forked via memory.fork(), creating an
        independent conversation state.

        Returns:
            A new agent instance with copied state
        """
        new_memory = self._memory.fork()

        new_agent = self.__class__(
            client=self._client,
            model_name=self._model_name,
            name=self.name,
            system_prompt=self._system_prompt,
            tools=self._tools,
            max_steps=self._max_steps,
            memory=new_memory,
            input_guardrails=list(self._input_guardrails),
            output_guardrails=list(self._output_guardrails),
        )

        return new_agent

    def as_tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_extractor: Optional[Callable] = None,
        stateful: bool = False,
        max_steps: Optional[int] = None,
        input_guardrails: Optional[list] = None,
        output_guardrails: Optional[list] = None,
    ) -> Tool:
        """Wrap this agent as a Tool for use by another agent.

        Returns an :class:`AgentTool` that delegates to this agent instance.
        """
        from motus.tools.core.agent_tool import AgentTool

        return AgentTool(
            agent=self,
            name=name,
            description=description,
            output_extractor=output_extractor,
            stateful=stateful,
            max_steps=max_steps,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )

    @abstractmethod
    async def _run(
        self,
        user_prompt: Optional[str] = None,
        **kwargs,
    ) -> T:
        """
        Execute the agent's core logic.

        This method must be implemented by subclasses to define the
        agent's behavior (e.g., ReAct loop, chain-of-thought, etc.)

        Args:
            user_prompt: Optional user message to start or continue the conversation
            **kwargs: Additional arguments for the specific agent implementation

        Returns:
            The result of the agent's execution (type depends on implementation)
        """
        raise NotImplementedError("Subclasses must implement the _run method.")

    def __call__(
        self,
        user_prompt: Optional[str] = None,
        **kwargs,
    ) -> T:
        """
        Run the agent with the given user prompt.

        Infers agent name from the caller's variable name on first call
        if no explicit name was provided at __init__ time.

        Args:
            user_prompt: Optional user message to start or continue the conversation
            **kwargs: Additional arguments passed to _run()

        Returns:
            The result of the agent's execution
        """
        # Infer name while still in the caller's thread context
        if self._name is None:
            self._name = self._infer_name()
        return self._execute(user_prompt, **kwargs)

    @agent_task(task_type=AGENT_CALL)
    async def _execute(
        self,
        user_prompt: Optional[str] = None,
        **kwargs,
    ) -> T:
        """
        Internal method decorated with @agent_task for runtime integration.

        This runs as an async coroutine on the runtime event loop.
        Guardrails are executed before/after _run().
        """
        # Connect lazy MCP sessions before first run
        if self._tools is not None and hasattr(self._tools, "_connect_mcp_sessions"):
            await self._tools._connect_mcp_sessions()

        # Input guardrails — fn(value, agent)
        if user_prompt is not None and self._input_guardrails:
            from motus.guardrails import run_guardrails

            user_prompt = await run_guardrails(
                self._input_guardrails, user_prompt, agent=self
            )

        result = await self._run(user_prompt, **kwargs)

        # Output guardrails
        if self._output_guardrails:
            if isinstance(result, BaseModel):
                from motus.guardrails import run_structured_output_guardrails

                updated = await run_structured_output_guardrails(
                    self._output_guardrails, result.model_dump(), agent=self
                )
                result = type(result).model_validate(updated)
            elif isinstance(result, str):
                from motus.guardrails import run_guardrails

                result = await run_guardrails(
                    self._output_guardrails, result, agent=self
                )

        return result

    def _infer_name(self) -> str:
        """Infer agent name from the caller's variable name.

        Walks up the call stack to find a local variable referencing this
        agent instance (e.g. ``research_agent = ReActAgent(...)``).

        Returns:
            The variable name (e.g. ``"research_agent"``), or the class name
            as fallback.
        """
        try:
            frame = inspect.currentframe()
            current = frame.f_back  # Skip _infer_name itself
            while current is not None:
                for var_name, value in current.f_locals.items():
                    if (
                        value is self
                        and var_name != "self"
                        and not var_name.startswith("_")
                    ):
                        return var_name
                current = current.f_back
        except Exception:
            pass
        finally:
            del frame

        # Fallback to class name
        return self.__class__.__name__

    # -------------------------------------------------------------------------
    # Serve contract
    # -------------------------------------------------------------------------

    async def run_turn(
        self, message: ChatMessage, state: List[ChatMessage]
    ) -> tuple[ChatMessage, List[ChatMessage]]:
        """Run a single conversational turn, satisfying the serve contract.

        Replays prior conversation history into working memory, executes the
        agent, and returns the response with updated state.

        State is stored as non-system messages only (user + assistant + tool).
        The agent's own system prompt is configured at init time and does not
        need to be persisted in the session state.
        """
        # Replay prior conversation (skip system messages — agent already
        # has its own system prompt from module-level initialization).
        for msg in state:
            if msg.role != "system":
                await self.add_message(msg)
        response_text = await self(message.content)
        response = ChatMessage.assistant_message(content=response_text)
        # Return raw messages (without system prefix) as session state
        new_state = list(self.memory.messages)
        return response, new_state

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "AgentBase[T]":
        """Async context manager entry."""
        if self._tools is not None:
            await self._tools.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._tools is not None:
            await self._tools.__aexit__(exc_type, exc_val, exc_tb)
        # Stop memory promotion task if running
        self._memory.stop_promotion_task()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_last_assistant_message(self) -> Optional[ChatMessage]:
        """
        Get the last assistant message from the conversation.

        Returns:
            The last assistant message or None if none exists
        """
        for msg in reversed(self._memory.messages):
            if msg.role == "assistant":
                return msg
        return None

    async def compact_memory(
        self,
        preserve_ratio: Optional[float] = None,
        preserve_recent: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Manually trigger memory compaction.

        Summarizes older messages to reduce context size while
        preserving important information.

        Returns:
            SummarizationResult if compaction occurred, None otherwise
        """
        return await self._memory.compact(
            preserve_ratio=preserve_ratio, preserve_recent=preserve_recent
        )

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"model={self._model_name!r}, "
            f"tools={len(self._tools) if self._tools else 0}, "
            f"messages={len(self._memory.messages)}"
            f")"
        )

    def _init_tools(self) -> None:
        self._tools = normalize_tools(self._tools) or None
        memory_tools = normalize_tools(self._memory.build_tools())
        if memory_tools and self._tools is not None:
            self._tools.update(memory_tools)
        elif memory_tools:
            self._tools = memory_tools
