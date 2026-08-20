from pathlib import Path
import re

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor


# =========================
# 配置
# =========================
INPUT_CSV = "results/explanation/question.csv"
OUTPUT_DIR = "results/explanation/molecular_png"

IMAGE_SIZE = (800, 500)


# =========================
# 工具函数
# =========================
def safe_filename(name: str) -> str:
    """将名称转换为适合文件名的形式。"""
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


# =========================
# 主程序
# =========================
def main():
    input_path = Path(INPUT_CSV)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_columns = ["fifth_name", "fifth_smiles"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少必要列: {col}")

    success = 0
    failed = 0
    skipped = 0

    for idx, row in df.iterrows():
        fifth_name = row["fifth_name"]
        fifth_smiles = row["fifth_smiles"]

        # 跳过空值
        if pd.isna(fifth_smiles) or str(fifth_smiles).strip() == "":
            skipped += 1
            continue

        fifth_smiles = str(fifth_smiles).strip()

        if pd.isna(fifth_name) or str(fifth_name).strip() == "":
            fifth_name = f"unknown_{idx}"
        else:
            fifth_name = str(fifth_name).strip()

        # SMILES -> RDKit Mol
        mol = Chem.MolFromSmiles(fifth_smiles)

        if mol is None:
            print(
                f"[FAILED] row={idx}, "
                f"name={fifth_name}, "
                f"SMILES={fifth_smiles}"
            )
            failed += 1
            continue

        # 重新计算较清晰的 2D 坐标
        rdDepictor.Compute2DCoords(mol)

        # 文件名加入原始行号，防止重复名称覆盖
        file_name = f"{idx:03d}_{safe_filename(fifth_name)}.png"
        output_path = output_dir / file_name

        # 绘制分子，并使用 fifth_name 作为图片标题
        Draw.MolToFile(
            mol,
            str(output_path),
            size=IMAGE_SIZE,
            legend=fifth_name,
        )

        print(f"[OK] {fifth_name} -> {output_path}")
        success += 1

    print("\n=========================")
    print("处理完成")
    print(f"成功生成 : {success}")
    print(f"SMILES失败: {failed}")
    print(f"跳过空行 : {skipped}")
    print(f"输出目录 : {output_dir.resolve()}")
    print("=========================")


if __name__ == "__main__":
    main()