import os
import sys
import time
import yaml
import logging
import datetime
import cflearn_deploy
import logging.config

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple
from fastapi import File
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Response
from fastapi import HTTPException
from pydantic import BaseModel
from pymilvus_orm import connections
from pymilvus_orm import DataType
from pymilvus_orm import Collection
from pymilvus_orm import FieldSchema
from pymilvus_orm import CollectionSchema
from cflearn_deploy.toolkit import np_to_bytes


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
model_zoo: Dict[str, Any] = {}
model_root = os.path.join(root, "models")


# sod


class SODModel(BaseModel):
    smooth: int = 0
    tight: float = 0.9
    model_name: Optional[str] = None
    model_path: Optional[str] = None


class LoadedSODModel(NamedTuple):
    api: cflearn_deploy.SOD
    path: str


class SODResponse(BaseModel):
    content: bytes


@app.post("/cv/sod", response_model=SODResponse)
def sod(img_bytes0: bytes = File(...), data: SODModel = Depends()) -> Response:
    try:
        logging.debug("/cv/sod endpoint entered")
        t = time.time()
        key = "sod"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.SOD(model_path)
            api_bundle = model_zoo[key] = LoadedSODModel(api, model_path)
        rgba = api_bundle.api.run(img_bytes0, smooth=data.smooth, tight=data.tight)
        logging.debug(f"/cv/sod elapsed time : {time.time() - t:8.6f}s")
        return Response(content=np_to_bytes(rgba), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# milvus


def _milvus_search(
    name: str,
    collection: Collection,
    data: Any,
    latent_code: Any,
    t1: float,
    t2: float,
) -> List[Any]:
    t3 = time.time()
    res = collection.search(
        [latent_code.tolist()],
        data.field_name,
        dict(metric_type=data.metric_type, params={"nprobe": data.nprobe}),
        data.top_k,
        output_fields=["id"],
    )
    hits = res[0]
    t4 = time.time()
    logging.debug(
        f"/cv/{name} elapsed time : {t4 - t1:8.6f}s "
        f"| onnx : {t2 - t1:8.6f} "
        f"| milvus_init : {t3 - t2:8.6f} "
        f"| milvus : {t4 - t3:8.6f} |"
    )
    return hits


# cbir


def get_cbir_collection() -> Collection:
    connections.connect(host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="latent_code", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields=fields, description="cbir collection")
    return Collection(name="cbir", schema=schema)


class CBIRModel(BaseModel):
    top_k: int = 10
    nprobe: int = 16
    metric_type: str = "L2"
    field_name: str = "latent_code"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    skip_milvus: bool = False


class LoadedImageEncoder(NamedTuple):
    api: cflearn_deploy.ImageEncoder
    path: str


class CBIRResponse(BaseModel):
    indices: List[int]
    distances: List[float]


@app.post("/cv/cbir", response_model=CBIRResponse)
def cbir(img_bytes0: bytes = File(...), data: CBIRModel = Depends()) -> CBIRResponse:
    try:
        logging.debug("/cv/cbir endpoint entered")
        t1 = time.time()
        key = "cbir"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.ImageEncoder(model_path)
            api_bundle = model_zoo[key] = LoadedImageEncoder(api, model_path)
        latent_code = api_bundle.api.run(img_bytes0)
        if data.skip_milvus:
            return CBIRResponse(indices=[0], distances=[0])
        t2 = time.time()
        collection = get_cbir_collection()
        hits = _milvus_search("cbir", collection, data, latent_code, t1, t2)
        return CBIRResponse(
            indices=[hit.id for hit in hits],
            distances=[hit.distance for hit in hits],
        )
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# tbir


def get_tbir_collection() -> Collection:
    connections.connect(host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="latent_code", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]
    schema = CollectionSchema(fields=fields, description="tbir collection")
    return Collection(name="tbir", schema=schema)


class TBIRModel(BaseModel):
    top_k: int = 10
    nprobe: int = 16
    metric_type: str = "L2"
    field_name: str = "latent_code"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    skip_milvus: bool = False


class LoadedTextEncoder(NamedTuple):
    api: cflearn_deploy.TextEncoder
    model_path: str
    tokenizer_path: str


class TBIRResponse(BaseModel):
    indices: List[int]
    distances: List[float]


@app.post("/cv/tbir", response_model=TBIRResponse)
def tbir(text: List[str], data: TBIRModel = Depends()) -> TBIRResponse:
    try:
        logging.debug("/cv/tbir endpoint entered")
        t1 = time.time()
        key = "tbir"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if data.tokenizer_path is not None:
            tokenizer_path = data.tokenizer_path
        else:
            model_name = data.model_name or key
            tokenizer_path = os.path.join(model_root, f"{model_name}_tokenizer.pkl")
        if (
            api_bundle is None
            or api_bundle.model_path != model_path
            or api_bundle.tokenizer_path != tokenizer_path
        ):
            api = cflearn_deploy.TextEncoder(model_path, tokenizer_path)
            api_bundle = LoadedTextEncoder(api, model_path, tokenizer_path)
            model_zoo[key] = api_bundle
        latent_code = api_bundle.api.run(text)
        if data.skip_milvus:
            return TBIRResponse(indices=[0], distances=[0])
        t2 = time.time()
        collection = get_tbir_collection()
        hits = _milvus_search("tbir", collection, data, latent_code, t1, t2)
        return TBIRResponse(
            indices=[hit.id for hit in hits],
            distances=[hit.distance for hit in hits],
        )
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# adain


class AdaINModel(BaseModel):
    model_name: Optional[str] = None
    model_path: Optional[str] = None


class LoadedAdaINModel(NamedTuple):
    api: cflearn_deploy.AdaINStylizer
    path: str


class AdaINResponse(BaseModel):
    content: bytes


@app.post("/cv/adain", response_model=AdaINResponse)
def adain(
    img_bytes0: bytes = File(...),
    img_bytes1: bytes = File(...),
    data: AdaINModel = Depends(),
) -> Response:
    try:
        logging.debug("/cv/adain endpoint entered")
        t = time.time()
        key = "adain"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.AdaINStylizer(model_path)
            api_bundle = model_zoo[key] = LoadedAdaINModel(api, model_path)
        stylized = api_bundle.api.run(img_bytes0, img_bytes1)
        logging.debug(f"/cv/adain elapsed time : {time.time() - t:8.6f}s")
        return Response(content=np_to_bytes(stylized), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# clf


class ClfModel(BaseModel):
    model_name: Optional[str] = None
    model_path: Optional[str] = None


class LoadedClfModel(NamedTuple):
    api: cflearn_deploy.Clf
    path: str


class ClfResponse(BaseModel):
    probabilities: List[float]


@app.post("/cv/clf", response_model=ClfResponse)
def clf(img_bytes0: bytes = File(...), data: ClfModel = Depends()) -> ClfResponse:
    try:
        logging.debug("/cv/clf endpoint entered")
        t = time.time()
        key = "clf"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.Clf(model_path)
            api_bundle = model_zoo[key] = LoadedClfModel(api, model_path)
        probabilities = api_bundle.api.run(img_bytes0).tolist()
        logging.debug(f"/cv/clf elapsed time : {time.time() - t:8.6f}s")
        return ClfResponse(probabilities=probabilities)
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
