"""
Prompt templates for CC-style context compaction.

Contains:
- COMPACTION_USER_PROMPT: Appended as the last user message to trigger summarization
- CONTINUATION_TEMPLATE: Wraps the summary as the first user message in the new context
"""

COMPACTION_USER_PROMPT = """\
[This is a summarization instruction, NOT part of the conversation. \
Do NOT include this message in the user message list or treat it as a user request.]

Please summarize the entire conversation above into a detailed "working state" \
summary. The agent will use ONLY this summary and the system prompt to continue \
its work, so be thorough and reconstruction-grade.

## Output Format

Produce the summary in two parts: **Analysis** followed by **Summary**.

## Analysis

Write a chronological narrative of the conversation. Walk through the session \
step by step, covering:
- What the user asked and how the agent responded at each stage.
- Key exploration, implementation, and debugging phases.
- User feedback and how it redirected or refined the work.
- Important decision points and their outcomes.

This narrative provides temporal context that the structured Summary sections \
below cannot fully capture. Keep it factual and dense -- no filler.

## Summary

### 1. User Intent & Current Task
- What did the user originally ask for?
- What is the most recent request or task the agent was working on?
- Has the user's intent evolved during the conversation?

### 2. Key Technical Decisions & Concepts
- What architectural or design decisions were made?
- What constraints or requirements were established?
- What approaches were considered and why were they accepted/rejected?

### 3. Files & Artifacts
For each file involved, provide:
- **Path**: Full file path.
- **Action**: NEW / MODIFIED / READ / DELETED.
- **Changes**: What was changed and why (one or two sentences).
- **Code snippet**: Include key code snippets (function signatures, class \
definitions, critical logic) when the exact content matters for resumption. \
Omit for files that were only read for reference.
- At the end, list **Key files READ but not modified** with a brief note on \
why each was read (e.g., "read to understand X", "referenced for Y").

### 4. Errors & Fixes
- What errors or issues were encountered?
- How were they resolved (or are they still pending)?

### 5. Progress & Completed Work
- What has been accomplished so far?
- Provide a chronological summary of major steps taken.

### 6. All User Messages (Paraphrased)
- Capture every user message in order, paraphrased to preserve intent.
- This ensures no user instruction is lost during compaction.

### 7. Pending Tasks
- What tasks remain to be done?
- List them in priority order.

### 8. Current Work
- What was the user's last message or request (quote or closely paraphrase it)?
- What was the agent in the middle of doing in response?
- What is the exact current state (e.g., which file was being edited, what step was in progress)?
- Be as specific as possible so the agent can resume without re-reading.

### 9. Next Step
- What should the agent do next to continue the most recent user request?
- Be specific and actionable.

## Guidelines
- Be comprehensive but concise. Include enough detail for reconstruction.
- Preserve all user instructions and preferences mentioned in the conversation.
- Include specific names, paths, values, and identifiers -- do not generalize.
- If tool calls produced important results, include the key outputs.
- Do NOT include meta-commentary about summarization itself.\
"""


CONTINUATION_TEMPLATE = """\
This session is being continued from a previous conversation that ran out of \
context window space. The summary below was auto-generated and covers the \
earlier portion of the conversation.

<context_summary>
{summary}
</context_summary>

Please continue the conversation from where it left off. Do not ask the user \
any further questions about what was already discussed. Continue with the last \
task that was being worked on.\
"""
