import numpy as np
import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.toolkit import to_uint8
from cflearn_deploy.data.transforms import ToNCHW


class TritonPythonModel:
    def initialize(self, _: Dict[str, Any]) -> None:
        self.to_nchw = ToNCHW()

    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            content = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            style = pb_utils.get_input_tensor_by_name(request, "style").as_numpy()
            content, style = map(self.to_nchw, [content, style])
            m_request = pb_utils.InferenceRequest(
                model_name="adain_core",
                requested_output_names=["predictions"],
                inputs=[
                    pb_utils.Tensor("input", content.astype(np.float32) / 255.0),
                    pb_utils.Tensor("style", style.astype(np.float32) / 255.0),
                ],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            tensor = pb_utils.get_output_tensor_by_name(m_response, "predictions")
            array = to_uint8(tensor.as_numpy().transpose([0, 2, 3, 1]))
            outputs = [pb_utils.Tensor("predictions", array)]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
