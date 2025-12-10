from dataclasses import dataclass


@dataclass
class AgentRunFlags:
    used_rag: bool = False
    used_web: bool = False


# Simple global flags for single-process CLI runs.
FLAGS = AgentRunFlags()
