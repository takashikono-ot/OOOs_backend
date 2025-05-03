from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
import numpy as np

app = FastAPI(title="OOOs 潜在ランク推定API")

# ─────────────────────────
#  CORS
# ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ooos-frontend.netlify.app",
        "http://localhost:3000",   # 開発用
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────
#  ルート（動作確認用）
# ─────────────────────────
@app.get("/")
def read_root():
    return {"message": "OOOs API is up and running"}

# ─────────────────────────
#  IRP 読み込み（ヘッダー行＝空白区切り、データ行＝カンマ）
# ─────────────────────────
BASEDIR = os.path.dirname(__file__)
irp_path = os.path.join(BASEDIR, "irp_output.csv")

# 1行目をスキップし、残りをカンマ区切りで読む
irp_df = pd.read_csv(
    irp_path,
    sep=",",
    header=None,
    skiprows=1,       # ヘッダー行 "IRP1 IRP2 …" を無視
    engine="python",
    dtype=float,      # すべて float で読み取る
)

irp = irp_df.values        # shape (126, 7)
irp = np.clip(irp, 1e-12, 1-1e-12)  # log(0) 回避
print("IRP shape:", irp.shape, " dtype:", irp.dtype)

# ─────────────────────────
#  ユーティリティ
# ─────────────────────────
def to_binary_all(res: List[int]) -> np.ndarray:
    """42項目(1–4)を累積二値化 => shape (126,)"""
    return np.array(
        [1 if x >= th else 0 for x in res for th in (1, 2, 3)],
        dtype=int
    )

def predict_rank_probs(res: List[int]) -> np.ndarray:
    b = to_binary_all(res)           # (126,)
    logL = (b[:, None] * np.log(irp) +
            (1 - b)[:, None] * np.log(1 - irp)).sum(axis=0)
    logL -= logL.max()               # オーバーフロー対策
    probs = np.exp(logL)
    return probs / probs.sum()       # 正規化 (7,)

# ─────────────────────────
#  スキーマ
# ─────────────────────────
class OOOsRequest(BaseModel):
    responses: List[int]  # 42 要素, 1–4

class OOOsResponse(BaseModel):
    rank_probs: List[float]  # 7 要素
    estimated_rank: int      # 1–7

# ─────────────────────────
#  エンドポイント
# ─────────────────────────
@app.post("/predict", response_model=OOOsResponse)
def predict(req: OOOsRequest):
    if len(req.responses) != 42:
        raise HTTPException(400, "`responses` must have 42 elements")
    if any((x < 1 or x > 4) for x in req.responses):
        raise HTTPException(400, "each response must be 1–4")

    try:
        probs = predict_rank_probs(req.responses)
    except Exception as e:
        raise HTTPException(500, f"計算エラー: {e}")

    return OOOsResponse(
        rank_probs=probs.tolist(),
        estimated_rank=int(np.argmax(probs) + 1)
    )
