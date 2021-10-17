import os

import numpy as np
import triton_python_backend_utils as pb_utils

from typing import Any
from typing import Dict
from typing import List
from cflearn_deploy.data.transforms import ImagenetPreprocess


class TritonPythonModel:
    output_names = ["predictions", "classes"]

    def initialize(self, _: Dict[str, Any]) -> None:
        self.preprocess = ImagenetPreprocess()
        current_folder = os.path.dirname(__file__)
        self.labels_dict = {}
        for name in self.output_names:
            path = os.path.join(current_folder, f"{name}.txt")
            with open(path, "r", encoding="utf-8") as f:
                self.labels_dict[name] = [line.strip() for line in f]

    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            array = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            top_k = pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()
            top_k = top_k.item()
            array = self.preprocess(array, False)
            m_request = pb_utils.InferenceRequest(
                model_name="product_clf_core",
                requested_output_names=self.output_names,
                inputs=[pb_utils.Tensor("input", array)],
            )
            m_response = m_request.exec()
            if m_response.has_error():
                raise pb_utils.TritonModelException(m_response.error().message())
            outputs = []
            for name in self.output_names:
                labels = self.labels_dict[name]
                tensor = pb_utils.get_output_tensor_by_name(m_response, name)
                probabilities = tensor.as_numpy()
                top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :top_k]
                outputs.append(
                    pb_utils.Tensor(
                        name,
                        np.array(
                            [
                                [
                                    f"{i_probabilities[i]:.8f}:{i}:{labels[i]}"
                                    for i in i_top_indices
                                ]
                                for i_probabilities, i_top_indices in zip(
                                    probabilities, top_indices
                                )
                            ],
                            np.object,
                        ),
                    )
                )
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
