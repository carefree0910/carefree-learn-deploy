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
from cflearn_deploy.data import sqlite
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

# db
engine = sqlite.get_engine(echo=False)

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
def sod(img_bytes: bytes = File(...), data: SODModel = Depends()) -> Response:
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
        rgba = api_bundle.api.run(img_bytes, smooth=data.smooth, tight=data.tight)
        logging.debug(f"/cv/sod elapsed time : {time.time() - t:8.6f}s")
        return Response(content=np_to_bytes(rgba), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


# cbir


def get_cbir_collection() -> Collection:
    connections.connect(host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="latent_code", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields=fields, description="dino vit collection")
    return Collection(name="dino_vit", schema=schema)


class CBIRModel(BaseModel):
    top_k: int = 10
    nprobe: int = 16
    metric_type: str = "L2"
    field_name: str = "latent_code"
    model_name: Optional[str] = None
    model_path: Optional[str] = None


class LoadedEncoder(NamedTuple):
    api: cflearn_deploy.ImageEncoder
    path: str


class CBIRResponse(BaseModel):
    indices: List[int]
    distances: List[float]


@app.post("/cv/cbir", response_model=CBIRResponse)
def cbir(img_bytes: bytes = File(...), data: CBIRModel = Depends()) -> CBIRResponse:
    try:
        logging.debug("/cv/cbir endpoint entered")
        t1 = time.time()
        key = "dino_vit"
        api_bundle = model_zoo.get(key)
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or key
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.ImageEncoder(model_path)
            api_bundle = model_zoo[key] = LoadedEncoder(api, model_path)
        latent_code = api_bundle.api.run(img_bytes)
        t2 = time.time()
        collection = get_cbir_collection()
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
            f"/cv/cbir elapsed time : {t3 - t1:8.6f}s "
            f"| onnx : {t2 - t1:8.6f} "
            f"| milvus_init : {t3 - t2:8.6f} "
            f"| milvus : {t4 - t3:8.6f} |"
        )
        return CBIRResponse(
            indices=[hit.id for hit in hits],
            distances=[hit.distance for hit in hits],
        )
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
