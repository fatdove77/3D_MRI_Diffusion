# save as build_description.py
import argparse
import numpy as np
import pandas as pd

TEMPLATE = "T1-weighted brain scan of {age}-year-old {gender} showing normal anatomy, high-resolution imaging"

def main(inp, outp):
    df = pd.read_csv(inp)

    # 基本校验
    need = {"dir", "sex", "age"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV缺少列: {missing}")

    # 删除 dataset / id（若不存在则忽略）
    df = df.drop(columns=[c for c in ["dataset", "id"] if c in df.columns])

    # sex: m/f -> male/female（大小写/空白均兼容）
    sex_norm = (
        df["sex"].astype(str).str.strip().str.lower().map({"m": "male", "f": "female"})
    )
    # 遇到非 m/f 的值则原样保留（尽量不丢信息）
    sex_final = sex_norm.fillna(df["sex"].astype(str).str.strip().str.lower())

    # age 向下取整（非数字用 "unknown" 兜底）
    age_num = pd.to_numeric(df["age"], errors="coerce")
    age_floor = np.floor(age_num)
    age_str = (
        pd.Series(age_floor, index=df.index)
        .astype("Int64")             # 可空整数
        .astype(str)
        .replace({"<NA>": "unknown"})
    )

    # 组装 describtion
    df["describtion"] = [
        TEMPLATE.format(age=a, gender=g) for a, g in zip(age_str, sex_final)
    ]

    # 仅保留需要的列
    out_df = df[["dir", "describtion"]]
    out_df.to_csv(outp, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    args = ap.parse_args()
    main(args.input_csv, args.output_csv)
