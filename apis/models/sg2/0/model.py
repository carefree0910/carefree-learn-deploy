import numpy as np
import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.toolkit import to_uint8
from cflearn_deploy.toolkit import min_max_normalize
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
            z = np.random.randn(1, 512).astype(np.float32)
            model = pb_utils.get_input_tensor_by_name(request, "model").as_numpy()[0][0]
            m_request = pb_utils.InferenceRequest(
                model_name=f"sg2_{model.decode()}_core",
                requested_output_names=["predictions"],
                inputs=[pb_utils.Tensor("input", z)],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            tensor = pb_utils.get_output_tensor_by_name(m_response, "predictions")
            array = tensor.as_numpy().transpose([0, 2, 3, 1])
            array = to_uint8(min_max_normalize(array))
            outputs = [pb_utils.Tensor("predictions", array)]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
