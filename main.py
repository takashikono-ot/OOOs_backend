from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np

app = FastAPI(title="OOOs 潜在ランク推定API")

# ↓↓↓ CORS 設定をここから追加 ↓↓↓
origins = [
    "https://ooos-frontend.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ↑↑↑ ここまで CORS 設定 ↑↑↑

# IRPパラメータ読み込み
# irp_output.csv は main.py と同じディレクトリに配置してください
irp = pd.read_csv("irp_output.csv").values  # shape: (n_binary, 7)

def to_binary_all(responses: List[int]) -> np.ndarray:
    """
    responses: 42項目のスコア (1～4)
    Returns: 各項目をしきい値1,2,3で累積二値化したベクトル
    """
    arr = []
    for x in responses:
        # threshold = 1,2,3 の3個ずつ
        for th in (1, 2, 3):
            arr.append(1 if x > th else 0)
    return np.array(arr, dtype=int)

def predict_rank_probs(responses: List[int]) -> np.ndarray:
    """
    responses: List[int] 長さ42
    Return: numpy array 長さ7 の各ランク所属確率
    """
    b = to_binary_all(responses)      # shape: (n_binary,)
    P = irp                            # shape: (n_binary, 7)
    # log-likelihood for each rank j
    logL = (b[:, None] * np.log(P) +
            (1 - b)[:, None] * np.log(1 - P)).sum(axis=0)
    # overflow対策
    logL -= logL.max()
    L = np.exp(logL)
    probs = L / L.sum()
    return probs

class OOOsRequest(BaseModel):
    responses: List[int]  # 長さ42, 各要素は1～4

class OOOsResponse(BaseModel):
    rank_probs: List[float]  # 長さ7
    estimated_rank: int      # 最も確率の高いランク (1～7)

@app.post("/predict", response_model=OOOsResponse)
def predict_endpoint(req: OOOsRequest):
    probs = predict_rank_probs(req.responses)
    est = int(probs.argmax()) + 1
    return {"rank_probs": probs.tolist(), "estimated_rank": est}
