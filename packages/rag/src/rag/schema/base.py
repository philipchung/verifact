"""Custom Base Component and List-like Container that has Pydantic Compatibility and
is compatible with LlamaIndex Node data structures.

Modified from llama_index.core.schema."""

import inspect
import json
from collections.abc import Iterator
from typing import Any, Generic, Self, TypeVar

from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseComponent as _BaseComponent


class BaseComponent(_BaseComponent):
    """Modify Base Component to allow for setting properties with setters in Pydantic V1."""

    def __setattr__(self, name, value) -> None:
        """
        Patch to be able to use properties with setters.
        https://github.com/pydantic/pydantic/issues/1577
        """
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__, predicate=lambda x: isinstance(x, property) and x.fset is not None
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e


# Create a type variable T, which can be any type
T = TypeVar("T")


class BaseListModel(BaseComponent, Generic[T]):
    """List-like Base Class."""

    items: list[T] = Field(default=[], description="List of items.")
    key: str = Field(
        default="items",
        description="Key for dict or JSON representation.",
        exclude=True,
        repr=False,
    )

    def __init__(self, *args, **kwargs) -> None:
        """Initialize with a list of items."""
        if args:
            super().__init__(items=args[0], **kwargs)
        else:
            super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "BaseListModel"

    # Implement methods for list-like behavior
    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __setitem__(self, index: int, value: T) -> None:
        self.items[index] = value

    def __delitem__(self, index: int) -> None:
        del self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def append(self, value: T) -> None:
        self.items.append(value)

    def extend(self, values: list[T]) -> None:
        self.items.extend(values)

    def insert(self, index: int, value: T) -> None:
        self.items.insert(index, value)

    def remove(self, value: T) -> None:
        self.items.remove(value)

    def pop(self, index: int = -1) -> T:
        return self.items.pop(index)

    def clear(self) -> None:
        self.items.clear()

    def index(self, value: T, start: int = 0, end: int = None) -> int:
        return self.items.index(value, start, end)

    def count(self, value: T) -> int:
        return self.items.count(value)

    def sort(self, *, key=None, reverse: bool = False) -> None:
        self.items.sort(key=key, reverse=reverse)

    def reverse(self) -> None:
        self.items.reverse()

    def copy(self) -> list[T]:
        return self.items.copy()

    # Custom Methods for Collection and Pydantic Model
    def to_json(self, key: str | None = None) -> str:
        """Converts the list to a JSON string with a key."""
        key = self.key if key is None else key
        return json.dumps({key: self.to_list_of_dict()})

    def to_list_of_dict(
        self,
        include: list | None = None,
        exclude: list | None = None,
        rename: dict[str, str] | None = None,
    ) -> list[dict]:
        """Convert each item in list to a dict object."""
        return [
            item.to_dict(include=include, exclude=exclude, rename=rename) for item in self.items
        ]


class ListModel(BaseListModel[Any]):
    @classmethod
    def class_name(cls) -> str:
        return "ListModel"


class BaseSetModel(BaseComponent, Generic[T]):
    """Set-like Base Class."""

    items: set[T] = Field(default_factory=set, description="Set of items.")
    key: str = Field(
        default="items",
        description="Key for dict or JSON representation.",
        exclude=True,
        repr=False,
    )

    def __init__(self, *args, **kwargs) -> None:
        """Initialize with a list of items."""
        if args:
            super().__init__(items=args[0], **kwargs)
        else:
            super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "BaseListModel"

    # Implement methods for set-like behavior
    def __repr__(self) -> str:
        return repr(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __contains__(self, other: T) -> bool:
        return other in self.items

    def __eq__(self, others: set[T]) -> bool:
        return self.items == others

    def __ne__(self, others: set[T]) -> bool:
        return self.items != others

    def __ge__(self, others: set[T]) -> bool:
        return self.items >= others

    def __gt__(self, others: set[T]) -> bool:
        return self.items > others

    def __le__(self, others: set[T]) -> bool:
        return self.items <= others

    def __lt__(self, others: set[T]) -> bool:
        return self.items < others

    def __and__(self, others: set[T]) -> set[T]:
        return self.items & others

    def __rand__(self, others: set[T]) -> set[T]:
        return others & self.items

    def __or__(self, others: set[T]) -> set[T]:
        return self.items | others

    def __ror__(self, others: set[T]) -> set[T]:
        return others | self.items

    def __sub__(self, others: set[T]) -> set[T]:
        return self.items - others

    def __xor__(self, others: set[T]) -> set[T]:
        return self.items ^ others

    def __rxor__(self, others: set[T]) -> set[T]:
        return others ^ self.items

    def union(self, *others: T) -> set[T]:
        return self.items.union(*others)

    def intersection(self, *others: T) -> set[T]:
        return self.items.intersection(*others)

    def difference(self, *others: T) -> set[T]:
        return self.items.difference(*others)

    def symmetric_difference(self, other: set[T]) -> set[T]:
        return self.items.symmetric_difference(other)

    def add(self, other: T) -> Self:
        self.items.add(other)

    def remove(self, other: T) -> None:
        self.items.remove(other)

    def discard(self, other: T) -> None:
        self.items.discard(other)

    def pop(self) -> T:
        return self.items.pop()

    def clear(self) -> None:
        self.items.clear()

    def update(self, *others: T) -> None:
        self.items.update(*others)

    def intersection_update(self, *others: T) -> None:
        self.items.intersection_update(*others)

    def difference_update(self, *others: T) -> None:
        self.items.difference_update(*others)

    def symmetric_difference_update(self, *others: T) -> None:
        self.items.symmetric_difference_update(*others)

    def isdisjoint(self, other: set[T]) -> bool:
        return self.items.isdisjoint(other)

    def issubset(self, other: set[T]) -> bool:
        return self.items.issubset(other)

    def issuperset(self, other: set[T]) -> bool:
        return self.items.issuperset(other)

    # Custom Methods for Collection and Pydantic Model
    def to_json(self, key: str | None = None) -> str:
        """Converts the set to a JSON object with key and value of set as a list."""
        key = self.key if key is None else key
        return json.dumps({key: self.to_list_of_dict()})

    def to_list_of_dict(
        self,
        include: list | None = None,
        exclude: list | None = None,
        rename: dict[str, str] | None = None,
    ) -> list[dict]:
        """Convert each item in set a dict object and return as a list of dict."""
        return [
            item.to_dict(include=include, exclude=exclude, rename=rename) for item in self.items
        ]
