from pydantic import ConfigDict, Field

from pydantic_utils.base import BaseListModel


class SimpleClaimList(BaseListModel[str]):
    """List of string claims."""

    model_config = ConfigDict(populate_by_name=True)

    items: list[str] = Field(default=[], description="List of claims.", alias="claims")
