import os
import dill

import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.toolkit import IRMixin
from cflearn_deploy.toolkit import get_compatible_name
from cflearn_deploy.data.transforms import ImagenetPreprocess


class TritonPythonModel(IRMixin):
    appendix_list = [""]

    def initialize(self, _: Dict[str, Any]) -> None:
        self.preprocess = ImagenetPreprocess()
        current_folder = os.path.dirname(__file__)
        tokenizer_name = get_compatible_name("tokenizer", (3, 8))
        with open(os.path.join(current_folder, f"{tokenizer_name}.pkl"), "rb") as f:
            self.tokenizer = dill.load(f)
        self.init_faiss("tbir", os.path.dirname(__file__))

    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            tokens = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            top_k = pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()
            n_probe = pb_utils.get_input_tensor_by_name(request, "num_probe").as_numpy()
            top_k, n_probe = top_k.item(), n_probe.item()
            decoded = [token[0].decode() for token in tokens]
            text = self.tokenizer.tokenize(decoded)
            m_request = pb_utils.InferenceRequest(
                model_name="tbir_core",
                requested_output_names=["predictions"],
                inputs=[pb_utils.Tensor("input", text)],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            outputs = self.get_outputs("", m_response, n_probe, top_k)
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
