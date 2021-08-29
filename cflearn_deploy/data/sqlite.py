import os

import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Any
from typing import Optional
from sqlmodel import create_engine
from sqlmodel import Field
from sqlmodel import Session
from sqlmodel import SQLModel
from sqlalchemy.engine import Engine

from ..toolkit import np_to_bytes
from ..constants import SQLITE_FILE


class ImageItem(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    bytes: bytes


def get_engine(*, echo: bool, **kwargs: Any) -> Engine:
    sqlite_url = f"sqlite:///{SQLITE_FILE}"
    kwargs.setdefault("connect_args", {"check_same_thread": False})
    return create_engine(sqlite_url, echo=echo, **kwargs)


def insert_images_from(folder: str) -> Engine:
    engine = get_engine(echo=True)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for stuff in tqdm(sorted(os.listdir(folder))):
            path = os.path.join(folder, stuff)
            try:
                Image.open(path).verify()
                img = Image.open(path)
                img_item = ImageItem(bytes=np_to_bytes(np.array(img)))
                session.add(img_item)
            except Exception as e:
                print(f"> error occurred with '{path}': {e}")
        session.commit()


def select_image(image_id: int) -> ImageItem:
    engine = get_engine(echo=False)
    with Session(engine) as session:
        return session.get(ImageItem, image_id)


__all__ = [
    "get_engine",
    "insert_images_from",
    "select_image",
    "ImageItem",
]
