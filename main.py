from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np

app = FastAPI(title="OOOs 潜在ランク推定API")

# CORS 設定
origins = [
    "https://ooos-frontend.netlify.app",
    "http://localhost:3000",  # ローカル開発用に追加
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IRPパラメータ読み込み（スペース・タブ区切りの場合）
# ————————————————————————————————
# 1) delim_whitespace=True で「空白/タブ」を区切り文字と認識
# 2) header=0 で１行目を列見出しとしてスキップ
# 3) dtype=float で全列を強制的に float に変換
# ————————————————————————————————
irp_df = pd.read_csv(
    "irp_output.csv",
    delim_whitespace=True,
    header=0,
    dtype=float
)
irp = irp_df.values  # ⇒ shape (n_binary, 7), 中身はすべて float
print("IRP shape:", irp.shape, " dtype:", irp.dtype)



def to_binary_all(responses: List[int]) -> np.ndarray:
    """
    responses: 42項目のスコア (1～4)
    Returns: 各項目をしきい値1,2,3で累積二値化したベクトル
    """
    arr = []
    for x in responses:
        for th in (1, 2, 3):
            # スコアがしきい値以上なら1
            arr.append(1 if x >= th else 0)
    return np.array(arr, dtype=int)


def predict_rank_probs(responses: List[int]) -> np.ndarray:
    """
    responses: List[int] 長さ42
    Return: numpy array 長さ7 の各ランク所属確率
    """
    b = to_binary_all(responses)    # shape: (n_binary,)
    P = irp                          # shape: (n_binary, 7)
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
    # 入力検証
    if len(req.responses) != 42:
        raise HTTPException(status_code=400, detail="`responses` must be length 42")
    if any(x < 1 or x > 4 for x in req.responses):
        raise HTTPException(status_code=400, detail="each response must be in 1..4")

    # 確率計算
    try:
        probs = predict_rank_probs(req.responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"計算エラー: {e}")

    estimated = int(np.argmax(probs)) + 1
    return OOOsResponse(rank_probs=probs.tolist(), estimated_rank=estimated)
