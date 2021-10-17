import os

import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.toolkit import IRMixin
from cflearn_deploy.data.transforms import ImagenetPreprocess


class TritonPythonModel(IRMixin):
    appendix_list = ["", "gray"]

    def initialize(self, _: Dict[str, Any]) -> None:
        self.preprocess = ImagenetPreprocess()
        self.init_faiss("cbir", os.path.dirname(__file__))

    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            array = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            top_k = pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()
            n_probe = pb_utils.get_input_tensor_by_name(request, "num_probe").as_numpy()
            gray = pb_utils.get_input_tensor_by_name(request, "gray").as_numpy()
            top_k, n_probe, gray = top_k.item(), n_probe.item(), gray.item()
            array = self.preprocess(array, gray)
            appendix = "_gray" if gray else ""
            m_request = pb_utils.InferenceRequest(
                model_name=f"cbir{appendix}_core",
                requested_output_names=["predictions"],
                inputs=[pb_utils.Tensor("input", array)],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            outputs = self.get_outputs(appendix, m_response, n_probe, top_k)
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
