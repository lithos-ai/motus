## Web tools

- `web_search` returns ranked search results (title, URL, snippet). Use it to find current information or locate URLs. Add `allowed_domains` / `blocked_domains` to constrain or exclude sources when relevant.
- `web_fetch` retrieves a web page and uses a small LLM to extract specific information based on a `prompt` you provide. Pass a focused extraction prompt ("List the API endpoints documented here", "What does this page say about X?") — vague prompts yield vague answers. Use `web_fetch` when you have a known URL; use `web_search` first when you need to find one.
- Don't invent URLs. Only fetch URLs the user provided, that you found via `web_search`, or that you read from a local file.
