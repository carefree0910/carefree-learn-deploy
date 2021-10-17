import numpy as np
import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.toolkit import sigmoid
from cflearn_deploy.data.transforms import ImagenetPreprocess


class TritonPythonModel:
    def initialize(self, _: Dict[str, Any]) -> None:
        self.preprocess = ImagenetPreprocess()

    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            src = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            src = src.astype(np.float32) / 255.0
            array = self.preprocess(src, False)
            m_request = pb_utils.InferenceRequest(
                model_name="sod_core",
                requested_output_names=["predictions"],
                inputs=[pb_utils.Tensor("input", array)],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            tensor = pb_utils.get_output_tensor_by_name(m_response, "predictions")
            logits = tensor.as_numpy()
            logits = np.clip(logits, -50.0, 50.0)
            outputs = [pb_utils.Tensor("predictions", sigmoid(logits)[:, 0])]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
