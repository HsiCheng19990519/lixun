from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class TodoItem:
    """
    Track a single todo step for the agent loop.
    """

    title: str
    status: Literal["todo", "doing", "done"] = "todo"
    note: str = ""


@dataclass
class AgentRunFlags:
    used_rag: bool = False
    used_web: bool = False
    todos: List[TodoItem] = field(default_factory=list)
