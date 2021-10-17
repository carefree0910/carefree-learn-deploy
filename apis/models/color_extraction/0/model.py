import numpy as np
import triton_python_backend_utils as pb_utils

from typing import List
from sklearn.cluster import KMeans


class TritonPythonModel:
    def execute(
        self,
        requests: List[pb_utils.InferenceRequest],
    ) -> List[pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            array = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()
            num_colors = pb_utils.get_input_tensor_by_name(request, "num_colors")
            num_colors = num_colors.as_numpy().item()
            colors = []
            for elem in array.reshape([array.shape[0], -1, 3]).astype(np.float32):
                k_means = KMeans(num_colors).fit(elem)
                centers = np.clip(k_means.cluster_centers_, 0.0, 255.0)
                colors.append(centers.astype(np.uint8))
            outputs = [pb_utils.Tensor("predictions", np.stack(colors, axis=0))]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
