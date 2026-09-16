- Use `uv`.
- Environment variable `DATASETS` sets the datasets to be used.
- Environment variable `LOGDIR` sets the directory where runs/logs are stored.
- Long jobs (`tune`/`test` runs) MUST be launched through the harness's own
  background-task mechanism so they can be stopped without a raw `kill`.
    - MUST NOT detach with `nohup` or `&`; that leaves a process the harness
      cannot cancel, recoverable only by PID.
    - MUST be one run per task, not several wrapped in a shell loop, so each is
      independently stoppable and a stop reliably reaps the training process.
    - Sequential runs SHOULD be chained by launching the next on completion of
      the previous.
- MUST NOT add Co-Authored-By: Claude <noreply@anthropic.com> (or similar) to commit messages.
- MUST add Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2] ... to the end of commit messages and PR descriptions.
    - AGENT_NAME is the name of the harness (/tool/framework)
    - MODEL_VERSION is the specific model version. SHOULD be as specific as possible including model family, version, and size.
    - [TOOL1] [TOOL2] are optional agent developer tools or MCP. SHOULD NOT include basic developer tools.
    - Example: Assisted-by: claude-code:claude-sonnet-4.6 github-mcp-server
    - Example: Assisted-by: copilot:gemini-3.7-flash
