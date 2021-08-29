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
from typing import Optional
from typing import NamedTuple
from fastapi import File
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Response
from fastapi import HTTPException
from pydantic import BaseModel
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


@app.post("/cv/sod")
def sod(img_bytes: bytes = File(...), data: SODModel = Depends()) -> Response:
    try:
        logging.debug("/cv/sod endpoint entered")
        t = time.time()
        api_bundle = model_zoo.get("sod")
        if data.model_path is not None:
            model_path = data.model_path
        else:
            model_name = data.model_name or "sod"
            model_path = os.path.join(model_root, f"{model_name}.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.SOD(model_path)
            api_bundle = model_zoo["sod"] = LoadedSODModel(api, model_path)
        rgba = api_bundle.api.run(img_bytes, smooth=data.smooth, tight=data.tight)
        logging.debug(f"/cv/sod elapsed time : {time.time() - t:8.6f}s")
        return Response(content=np_to_bytes(rgba), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
