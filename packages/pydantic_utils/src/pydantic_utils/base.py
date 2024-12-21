import json
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class CustomBaseModel(BaseModel):
    """Custom base pydantic class."""

    def to_dict(
        self,
        include: set | list | None = None,
        exclude: set | list | None = None,
        rename: dict[str, str] | None = None,
    ) -> dict[Any, list[Any]]:
        """Export model as a dictionary, optionally renaming keys."""
        d = self.model_dump(
            include=set(include) if include else None,
            exclude=set(exclude) if exclude else None,
        )
        if rename:
            output: dict[str, Any] = {}
            for key, value in d.items():
                if key in rename:
                    output[rename[key]] = value
                else:
                    output[key] = value
            return output
        else:
            return d


# Create a type variable T, which can be any type
T = TypeVar("T")


class BaseListModel(CustomBaseModel, Generic[T]):
    """List-like Base Class."""

    items: list[T] = Field(default=[], description="List of items.")
    key: SkipJsonSchema[str] = Field(
        default="items",
        description="Key for dict or JSON representation.",
        exclude=True,
        repr=False,
    )

    def __init__(self, input: list[T] | None = None, **kwargs) -> None:
        """Initialize with a list of items."""
        if input:
            super().__init__(items=input, **kwargs)
        else:
            super().__init__(**kwargs)

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
    pass
