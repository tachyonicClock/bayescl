from pydantic import BaseModel, ConfigDict


class NumericError(Exception):
    """Raised when a numeric error occurs, such as NaN or Inf values."""


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore

    def kwargs(self) -> dict:
        return self.model_dump(exclude={"type"})  # type: ignore
