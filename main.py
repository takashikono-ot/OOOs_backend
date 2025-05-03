from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
import numpy as np

# ───────────────────────────────
# FastAPI アプリケーション
# ───────────────────────────────
app = FastAPI(title="OOOs 潜在ランク推定API")

# CORS（本番とローカルの両方許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ooos-frontend.netlify.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 動作確認用
@app.get("/")
def read_root():
    return {"message": "OOOs API is up and running"}

# ───────────────────────────────
# IRP 行列の読み込み
# ───────────────────────────────
BASEDIR = os.path.dirname(__file__)
irp_path = os.path.join(BASEDIR, "irp_output.csv")

irp_df = pd.read_csv(irp_path, dtype=float)     # ヘッダあり
irp = np.clip(irp_df.values, 1e-12, 1 - 1e-12)  # log(0) 回避

# 形状チェック：126×7 でない場合は起動時に落とす
if irp.shape != (126, 7):
    raise RuntimeError(f"IRP matrix must be (126,7) — got {irp.shape}")

print(f"IRP loaded: shape={irp.shape}, dtype={irp.dtype}")

# ───────────────────────────────
#  ユーティリティ
# ───────────────────────────────
def to_binary_all(responses: List[int]) -> np.ndarray:
    """
    42項目の回答(1–4)を累積二値化して 126ビット配列に変換
    ビット i は『回答 > しきい値 th』のとき 1
    """
    return np.array(
        [1 if x > th else 0          # ← 重要：>= ではなく >
         for x in responses
         for th in (1, 2, 3)],
        dtype=int,
    )


def predict_rank_probs(responses: List[int]) -> np.ndarray:
    """
    b: shape (126,)
    irp: shape (126,7)
    戻り値: shape (7,) — 各ランクの事後確率
    """
    b = to_binary_all(responses)            # (126,)
    logL = (
        b[:, None] * np.log(irp) +
        (1 - b)[:, None] * np.log(1 - irp)
    ).sum(axis=0)                           # (7,)
    logL -= logL.max()                      # 安定化
    probs = np.exp(logL)
    return probs / probs.sum()


# ───────────────────────────────
#  Pydantic スキーマ
# ───────────────────────────────
class OOOsRequest(BaseModel):
    responses: List[int]  # 42 要素, 1–4

class OOOsResponse(BaseModel):
    rank_probs: List[float]  # 7 要素
    estimated_rank: int      # 1–7


# ───────────────────────────────
#  推定エンドポイント
# ───────────────────────────────
@app.post("/predict", response_model=OOOsResponse)
def predict(req: OOOsRequest):
    # 入力検証
    if len(req.responses) != 42:
        raise HTTPException(status_code=400,
                            detail="`responses` must contain 42 elements")
    if any(x < 1 or x > 4 for x in req.responses):
        raise HTTPException(status_code=400,
                            detail="each response must be an integer 1–4")

    try:
        probs = predict_rank_probs(req.responses)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"計算エラー: {e}")

    return OOOsResponse(
        rank_probs=[round(p, 6) for p in probs],       # 小数点丸め
        estimated_rank=int(np.argmax(probs) + 1)       # 1-indexed
    )
