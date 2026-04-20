from abc import ABC, abstractmethod

from model.schema import Message


class BaseMemory(ABC):
    """Abstract base class for Memory: manages conversation history"""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Add a message"""
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Get current valid message list"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear history"""
        ...

    @property
    @abstractmethod
    def token_count(self) -> int:
        """Current total token count"""
        ...

