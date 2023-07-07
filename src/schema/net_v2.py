from pydantic import BaseModel
from typing import List, Optional, Union, Any


class OutputStep(BaseModel):
    loss: List[float] = []
    preds: List[float] = []
    labels: List[float] = []
    imgs: Optional[List[Any]] = []

    def add(self, loss, preds, labels, imgs=None):
        self.loss.append(loss)
        self.preds.append(preds)
        self.labels.append(labels)
        if imgs is not None:
            self.imgs.append(imgs)

    def get(self):
        return self.loss, self.preds, self.labels, self.imgs

    def clear(self):
        self.loss.clear()
        self.preds.clear()
        self.labels.clear()
        self.imgs.clear()
