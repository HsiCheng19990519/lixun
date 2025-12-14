from dataclasses import dataclass


@dataclass
class AgentRunFlags:
    used_rag: bool = False
    used_web: bool = False
