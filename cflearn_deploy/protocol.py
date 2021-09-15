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

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ONNXModelProtocol(ModelProtocol, metaclass=ABCMeta):
    def __init__(self, onnx_path: str):
        self.onnx = ONNX(onnx_path)


__all__ = [
    "ModelProtocol",
    "ONNXModelProtocol",
]
