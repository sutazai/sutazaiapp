from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Dict, List, Optional#!/usr/bin/env python3.11"""Memory management module for AutoGPT agent.This module provides classes and utilities for managing the agent's memory,including conversation history and task context."""from typing import dict, list, Optionalfrom dataclasses import dataclassfrom datetime import datetimeimport jsonimport os@dataclassclass Message:    """Represents a message in the conversation history."""    role: str    content: str    timestamp: datetime = None    def __post_init__(self):    """Initialize timestamp if not provided."""        if self.timestamp is None:        self.timestamp = datetime.now()    def to_dict(self) -> Self = max_messages
self.persist_path = persist_path
self.messages: List[Message] = []
self.context: Dict = {}
if persist_path and os.path.exists(persist_path):        self.load()
def add_message(self, role: str, content: str) -> None:    """        Add a new message to the conversation history.        Args:    role: Role of the message sender (        e.g.,        "user",        "assistant")        content: Content of the message        """            message = Message(role=role, content=content)
self.messages.append(message)
        # Maintain maximum message limit
if len(self.messages) > self.max_messages:        self.messages = self.messages[-self.max_messages:]
if self.persist_path:        self.save()
def get_messages(self) -> Self()    def update_context(        self,        key: str,        value: any) -> None:    """        Update a value in \        the context dictionary.        Args:    key: Context key to update        value: New value for the key                        """
self.context[key] = value
if self.persist_path:        self.save()
def get_context(        self,        key: str) -> Self.get(
key)                            def clear_context(        self) -> Self}                                os.makedirs(
os.path.dirname(self.persist_path),
exist_ok=True)
with open(
self.persist_path,
"w") as f:                json.dump(
data,
f,
indent=2)
def load(        self) -> Self,            "r") as f:                data = json.load(                                    f)"""
self.messages = [Message.from_dict(
msg) for msg in data["messages"]]
self.context = data["context"]

"""""""""