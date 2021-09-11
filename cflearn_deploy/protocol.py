from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type

from .toolkit import WithRegister
from .onnx_api import ONNX


models: Dict[str, Type["ModelProtocol"]] = {}


class ModelProtocol(WithRegister, metaclass=ABCMeta):
    d = models

    def __init__(self, onnx_path: str, **kwargs: Any):
        self.onnx = ONNX(onnx_path)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass


__all__ = [
    "ModelProtocol",
]
