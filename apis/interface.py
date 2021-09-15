import os
import sys
import json
import time
import yaml
import faiss
import logging
import datetime
import logging.config

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from fastapi import File
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Response
from fastapi import HTTPException
from pydantic import BaseModel

from cflearn_deploy.toolkit import np_to_bytes
from cflearn_deploy.protocol import ONNXModelProtocol


app = FastAPI()
root = os.path.dirname(__file__)

# logging
logging_root = os.path.join(root, "logs")
os.makedirs(logging_root, exist_ok=True)
with open(os.path.join(root, "config.yml")) as f:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_path = os.path.join(logging_root, f"{timestamp}.log")
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["handlers"]["file"]["filename"] = log_path
    logging.config.dictConfig(config)

# models
model_zoo: Dict[str, "LoadedModel"] = {}
model_root = os.path.join(root, "models")

# faiss
faiss_zoo: Dict[str, faiss.Index] = {}
meta_root = os.path.join(root, "src", "meta")


def get_faiss_index(task: str, key: str, data: "ONNXModel") -> faiss.Index:
    name = data.name(key)
    zoo_key = f"{task}_{name}"
    index = faiss_zoo.get(zoo_key)
    if index is None:
        index = faiss.read_index(os.path.join(meta_root, task, f"{name}.index"))
        faiss_zoo[zoo_key] = index
    return index


# unified onnx api


class ONNXModel(BaseModel):
    onnx_name: Optional[str] = None
    onnx_path: Optional[str] = None

    @property
    def run_keys(self) -> List[str]:
        return []

    @property
    def model_keys(self) -> List[str]:
        return []

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.model_keys}

    @property
    def run_kwargs(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.run_keys}

    def name(self, key: str) -> str:
        return self.onnx_name or key

    def api_kwargs(self, key: str) -> Dict[str, Any]:
        name = self.name(key)
        onnx_path = self.onnx_path or os.path.join(model_root, f"{name}.onnx")
        api_kwargs = {"onnx_path": onnx_path}
        model_kwargs = self.model_kwargs
        for k, value in model_kwargs.items():
            if value is not None:
                api_kwargs[k] = value
            else:
                # get rid of "path" appendix
                key_name = "_".join(k.split("_")[:-1])
                api_kwargs[k] = os.path.join(model_root, f"{name}_{key_name}.pkl")
        return api_kwargs


class LoadedModel(dict):
    def __getattr__(self, item: str) -> Any:
        return self[item]

    def check_identical(self, key: str, data: ONNXModel) -> bool:
        api_kwargs = data.api_kwargs(key)
        for key in data.model_keys + ["onnx_path"]:
            if self[key] != api_kwargs[key]:
                return False
        return True


class ImageResponse(BaseModel):
    content: bytes


def _onnx_api(key: str, *args: Any, data: ONNXModel) -> Any:
    logging.debug(f"/cv/{key} endpoint entered")
    t1 = time.time()
    api_bundle = model_zoo.get(key)
    api_kwargs = data.api_kwargs(key)
    onnx_path = api_kwargs["onnx_path"]
    if api_bundle is None or not api_bundle.check_identical(key, data):
        api = ONNXModelProtocol.make(key, api_kwargs)
        api_kwargs["api"] = api
        api_bundle = model_zoo[key] = LoadedModel(**api_kwargs)
    t2 = time.time()
    run_kwargs = data.run_kwargs
    logging.debug(f"-> onnx_path : {onnx_path}")
    logging.debug(f"-> kwargs    : {run_kwargs}")
    result = api_bundle.api.run(*args, **run_kwargs)
    t3 = time.time()
    logging.debug(
        f"/cv/{key} elapsed time : {t3 - t1:8.6f}s | "
        f"load : {t2 - t1:8.6f} | run : {t3 - t2:8.6f}"
    )
    return result


# sod


class SODModel(ONNXModel):
    smooth: int = 0
    tight: float = 0.9

    @property
    def run_keys(self) -> List[str]:
        return ["smooth", "tight"]


@app.post("/cv/sod", response_model=ImageResponse)
def sod(img_bytes0: bytes = File(...), data: SODModel = Depends()) -> Response:
    try:
        rgba = _onnx_api("sod", img_bytes0, data=data)
        return Response(content=np_to_bytes(rgba), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# image retrieval


class IRModel(ONNXModel):
    task: str
    top_k: int = 10
    nprobe: int = 16
    skip_faiss: bool = False


class IRResponse(BaseModel):
    files: List[str]
    distances: List[float]

    @classmethod
    def dummy(cls) -> "IRResponse":
        return cls(files=[""], distances=[0.0])

    @classmethod
    def create_from(cls, key: str, code: np.ndarray, data: IRModel) -> "IRResponse":
        t1 = time.time()
        index = get_faiss_index(data.task, key, data)
        t2 = time.time()
        index.nprobe = data.nprobe
        distances, indices = index.search(code[None, ...], data.top_k)
        t3 = time.time()
        logging.debug(
            f"/cv/{key} -> faiss elapsed time : {t3 - t1:8.6f} | "
            f"load : {t2 - t1:8.6f} | core : {t3 - t2:8.6f}"
        )
        with open(os.path.join(meta_root, data.task, f"{key}_files.json"), "r") as rf:
            files = json.load(rf)
        return IRResponse(
            files=[files[i] for i in indices[0].tolist()],
            distances=distances[0].tolist(),
        )


# cbir


@app.post("/cv/cbir", response_model=IRResponse)
def cbir(img_bytes0: bytes = File(...), data: IRModel = Depends()) -> IRResponse:
    try:
        key = "cbir"
        latent_code = _onnx_api(key, img_bytes0, data=data)
        if data.skip_faiss:
            return IRResponse.dummy()
        return IRResponse.create_from(key, latent_code, data)
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# tbir


class TBIRModel(IRModel):
    tokenizer_path: Optional[str] = None

    @property
    def model_keys(self) -> List[str]:
        return ["tokenizer_path"]


@app.post("/cv/tbir", response_model=IRResponse)
def tbir(text: List[str], data: TBIRModel = Depends()) -> IRResponse:
    try:
        key = "tbir"
        latent_code = _onnx_api(key, text, data=data)
        if data.skip_faiss:
            return IRResponse.dummy()
        return IRResponse.create_from(key, latent_code, data)
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# adain


@app.post("/cv/adain", response_model=ImageResponse)
def adain(
    img_bytes0: bytes = File(...),
    img_bytes1: bytes = File(...),
    data: ONNXModel = Depends(),
) -> Response:
    try:
        stylized = _onnx_api("adain", img_bytes0, img_bytes1, data=data)
        return Response(content=np_to_bytes(stylized), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# clf


class ClfResponse(BaseModel):
    probabilities: List[float]


@app.post("/cv/clf", response_model=ClfResponse)
def clf(img_bytes0: bytes = File(...), data: ONNXModel = Depends()) -> ClfResponse:
    try:
        probabilities = _onnx_api("clf", img_bytes0, data=data)
        return ClfResponse(probabilities=probabilities.tolist())
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
