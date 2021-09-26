import os
import sys
import json
import time
import yaml
import faiss
import logging
import datetime
import logging.config

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

from cflearn_deploy.types import np_dict_type
from cflearn_deploy.toolkit import np_to_bytes
from cflearn_deploy.protocol import ModelProtocol
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


def get_faiss_index(task: str, faiss_name: str) -> faiss.Index:
    zoo_key = f"{task}_{faiss_name}"
    index = faiss_zoo.get(zoo_key)
    if index is None:
        index = faiss.read_index(os.path.join(meta_root, task, f"{faiss_name}.index"))
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


# style gan


@app.post("/cv/style_gan", response_model=ImageResponse)
def style_gan(data: ONNXModel = Depends()) -> Response:
    try:
        rgb = _onnx_api("style_gan", data=data)
        return Response(content=np_to_bytes(rgb), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# cycle gan


@app.post("/cv/cycle_gan", response_model=ImageResponse)
def cycle_gan(img_bytes0: bytes = File(...), data: ONNXModel = Depends()) -> Response:
    try:
        stylized = _onnx_api("cycle_gan", img_bytes0, data=data)
        return Response(content=np_to_bytes(stylized), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


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
    gray: bool = False
    no_transform: bool = False

    @property
    def run_keys(self) -> List[str]:
        return ["gray", "no_transform"]


class IRResponse(BaseModel):
    files: Dict[str, List[str]]
    distances: Dict[str, List[float]]

    @classmethod
    def dummy(cls) -> "IRResponse":
        return cls(files={"main": [""]}, distances={"main": [0.0]})

    @classmethod
    def create_from(cls, key: str, codes: np_dict_type, data: IRModel) -> "IRResponse":
        def _core(sub_key: Optional[str]) -> None:
            t1 = time.time()
            faiss_name = data.name(key)
            if sub_key is None:
                assert len(codes) == 1
                code = next(iter(codes.values()))
            else:
                faiss_name = f"{faiss_name}.{sub_key}"
                code = codes[sub_key]
            index = get_faiss_index(data.task, faiss_name)
            t2 = time.time()
            index.nprobe = data.nprobe
            sub_distances, indices = index.search(code, data.top_k)
            t3 = time.time()
            sub_str = "" if sub_key is None else f"({sub_key}) "
            logging.debug(
                f"/cv/{key} -> faiss elapsed time {sub_str}: "
                f"{t3 - t1:8.6f} | load : {t2 - t1:8.6f} | core : {t3 - t2:8.6f}"
            )
            file_name = f"{faiss_name}_files.json"
            file_path = os.path.join(meta_root, data.task, file_name)
            with open(file_path, "r", encoding="utf-8") as rf:
                sub_files = json.load(rf)
            if sub_key is None:
                sub_key = "main"
            files[sub_key] = [sub_files[i] for i in indices[0].tolist()]
            distances[sub_key] = sub_distances[0].tolist()

        files: Dict[str, List[str]] = {}
        distances: Dict[str, List[float]] = {}
        if len(codes) == 1:
            _core(None)
        else:
            for k in codes:
                _core(k)
        return IRResponse(files=files, distances=distances)


# cbir


@app.post("/cv/cbir", response_model=IRResponse)
def cbir(img_bytes0: bytes = File(...), data: IRModel = Depends()) -> IRResponse:
    try:
        key = "cbir"
        latent_codes = _onnx_api(key, img_bytes0, data=data)
        if data.skip_faiss:
            return IRResponse.dummy()
        return IRResponse.create_from(key, latent_codes, data)
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
        latent_codes = _onnx_api(key, text, data=data)
        if data.skip_faiss:
            return IRResponse.dummy()
        return IRResponse.create_from(key, latent_codes, data)
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


class ClfModel(ONNXModel):
    gray: bool = False
    no_transform: bool = False

    @property
    def run_keys(self) -> List[str]:
        return ["gray", "no_transform"]


class ClfResponse(BaseModel):
    probabilities: Dict[str, List[float]]


@app.post("/cv/clf", response_model=ClfResponse)
def clf(img_bytes0: bytes = File(...), data: ClfModel = Depends()) -> ClfResponse:
    try:
        prob_dict = _onnx_api("clf", img_bytes0, data=data)
        return ClfResponse(probabilities={k: v.tolist() for k, v in prob_dict.items()})
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# unified general cv api


def _cv_api(key: str, *args: Any, data: BaseModel) -> Any:
    logging.debug(f"/cv/{key} endpoint entered")
    t1 = time.time()
    model_kwargs = json.loads(data.json())
    logging.debug(f"-> kwargs    : {model_kwargs}")
    model = ModelProtocol.make(key, model_kwargs)
    t2 = time.time()
    result = model.run(*args)
    t3 = time.time()
    logging.debug(
        f"/cv/{key} elapsed time : {t3 - t1:8.6f}s | "
        f"build : {t2 - t1:8.6f} | run : {t3 - t2:8.6f}"
    )
    return result


# color extraction


class ColorExtractionModel(BaseModel):
    num_colors: int


class ColorExtractionResponse(BaseModel):
    colors: List[List[int]]


@app.post("/cv/color_extraction", response_model=ColorExtractionResponse)
def color_extraction(
    img_bytes0: bytes = File(...),
    data: ColorExtractionModel = Depends(),
) -> ColorExtractionResponse:
    try:
        colors = _cv_api("color_extraction", img_bytes0, data=data)
        return ColorExtractionResponse(colors=colors.tolist())
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
