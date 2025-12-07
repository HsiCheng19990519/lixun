## DevMate internal guidelines

1) Plan first  
   - Start with a short plan and list required files/dependencies.  
   - Prefer reusing templates and best practices instead of reinventing.

2) RAG usage  
   - Before generating code, query the local knowledge base with keywords like "guidelines" or "templates".  
   - Cite the source filename in the final answer for auditing.

3) Web/MCP search  
   - For external knowledge (API changes, versions) call Tavily via the MCP `search_web` tool.  
   - Record the query terms and top result titles.

4) Code generation  
   - When generating multiple files, present file path + content blocks.  
   - All model names/API keys/URLs must be configurable via env or config files; never hard-code.  
   - Default to Python 3.13 and LangChain 1.x.

5) Observability  
   - If LangSmith/LangFuse is available, enable it to log tool calls and prompts.

6) Error handling  
   - When search fails, return a retryable hint instead of failing silently.  
   - If RAG misses, say "no local knowledge base result" before switching to web search.
