import os
import sys
import logging
import cflearn_deploy

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
from cflearn_deploy.toolkit import np_to_bytes


app = FastAPI()
model_zoo: Dict[str, Any] = {}
model_root = os.path.join(os.path.dirname(__file__), "models")


class SODModel(BaseModel):
    smooth: int = 0
    tight: float = 0.9
    model_path: Optional[str] = None


class LoadedSODModel(NamedTuple):
    api: cflearn_deploy.SOD
    path: str


@app.post("/ai/sod")
def sod(img_bytes: bytes = File(...), data: SODModel = Depends()) -> Response:
    try:
        api_bundle = model_zoo.get("sod")
        model_path = data.model_path or os.path.join(model_root, "sod.onnx")
        if api_bundle is None or api_bundle.path != model_path:
            api = cflearn_deploy.SOD(model_path)
            api_bundle = model_zoo["sod"] = LoadedSODModel(api, model_path)
        rgba = api_bundle.api.run(img_bytes, smooth=data.smooth, tight=data.tight)
        return Response(content=np_to_bytes(rgba), media_type="image/png")
    except Exception as err:
        logging.exception(err)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
