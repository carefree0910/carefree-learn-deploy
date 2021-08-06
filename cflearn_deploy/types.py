import numpy as np

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional


data_type = Optional[Union[np.ndarray, str]]
general_config_type = Optional[Union[str, Dict[str, Any]]]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]


__all__ = [
    "data_type",
    "general_config_type",
    "np_dict_type",
    "states_callback_type",
    "sample_weights_type",
]
