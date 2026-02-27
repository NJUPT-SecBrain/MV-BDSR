#!/usr/bin/env python3
"""
验证 Big-Vul 数据集是否包含必需的列和数据
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def verify_bigvul(csv_path: str) -> bool:
    """
    验证 Big-Vul 数据集
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("请先安装 pandas: pip install pandas")
        return False
    
    logger.info(f"读取数据集: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return False
    
    all_columns = list(df.columns)
    logger.info(f"数据集包含 {len(all_columns)} 列")
    
    # 检查必需列（支持多种命名）
    code_before_cols = ["func_before", "buggy_code", "before"]
    code_after_cols = ["func_after", "fixed_code", "after"]
    patch_cols = ["diff", "patch"]
    
    found_before = None
    found_after = None
    found_patch = None
    
    for col in code_before_cols:
        if col in all_columns:
            found_before = col
            break
    
    for col in code_after_cols:
        if col in all_columns:
            found_after = col
            break
    
    for col in patch_cols:
        if col in all_columns:
            found_patch = col
            break
    
    # 输出验证结果
    is_valid = True
    
    print("\n" + "=" * 60)
    print("数据集验证结果")
    print("=" * 60 + "\n")
    
    if found_before:
        print(f"✅ 有漏洞代码列: '{found_before}'")
    else:
        print(f"❌ 缺少有漏洞代码列（需要: {', '.join(code_before_cols)}）")
        is_valid = False
    
    if found_after:
        print(f"✅ 修复后代码列: '{found_after}'")
    else:
        print(f"❌ 缺少修复后代码列（需要: {', '.join(code_after_cols)}）")
        is_valid = False
    
    if found_patch:
        print(f"✅ 补丁列: '{found_patch}'")
    else:
        print(f"⚠️  补丁列缺失（可选: {', '.join(patch_cols)}）")
    
    # 读取完整数据集统计
    if is_valid:
        print("\n" + "-" * 60)
        print("统计信息")
        print("-" * 60 + "\n")
        
        logger.info("读取完整数据集进行统计...")
        df_full = pd.read_csv(csv_path)
        
        total_count = len(df_full)
        print(f"📊 总样本数: {total_count:,}")
        
        # 检查非空值
        valid_before = df_full[found_before].notna().sum()
        valid_after = df_full[found_after].notna().sum()
        
        print(f"📊 有效的有漏洞代码: {valid_before:,} ({valid_before/total_count*100:.1f}%)")
        print(f"📊 有效的修复代码: {valid_after:,} ({valid_after/total_count*100:.1f}%)")
        
        # 代码长度统计
        df_full['before_len'] = df_full[found_before].fillna('').str.len()
        df_full['after_len'] = df_full[found_after].fillna('').str.len()
        
        avg_before = df_full['before_len'].mean()
        avg_after = df_full['after_len'].mean()
        
        print(f"\n📏 平均代码长度（字符数）:")
        print(f"   - 有漏洞代码: {avg_before:.0f}")
        print(f"   - 修复后代码: {avg_after:.0f}")
        
        # CWE 分布
        if 'cwe' in all_columns or 'cwe_id' in all_columns:
            cwe_col = 'cwe' if 'cwe' in all_columns else 'cwe_id'
            cwe_dist = df_full[cwe_col].value_counts().head(5)
            print(f"\n🏷️  Top 5 CWE 类型:")
            for cwe, count in cwe_dist.items():
                print(f"   - {cwe}: {count:,} ({count/total_count*100:.1f}%)")
        
        # 语言分布
        if 'lang' in all_columns or 'language' in all_columns:
            lang_col = 'lang' if 'lang' in all_columns else 'language'
            lang_dist = df_full[lang_col].value_counts()
            print(f"\n🌐 编程语言分布:")
            for lang, count in lang_dist.items():
                print(f"   - {lang}: {count:,} ({count/total_count*100:.1f}%)")
    
    print("\n" + "=" * 60)
    
    if is_valid:
        print("✅ 数据集验证通过！适合用于 MV-BDSR 框架")
        print("\n下一步:")
        print(f"  python scripts/prepare_bigvul.py --input {csv_path} --sample 10")
    else:
        print("❌ 数据集验证失败！")
        print("\n可能的原因:")
        print("  1. 这是 Big-Vul 的元数据版本（只有 commit 信息，没有代码）")
        print("  2. 列名不匹配（请查看 docs/BIGVUL_CORRECT_DATASET.md）")
        print("\n解决方案:")
        print("  bash scripts/download_bigvul_full.sh")
        print("  或查看: docs/BIGVUL_CORRECT_DATASET.md")
    
    print("=" * 60 + "\n")
    
    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="验证 Big-Vul 数据集是否适合 MV-BDSR 框架"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/bigvul.csv",
        help="Big-Vul CSV 文件路径",
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        logger.error(f"文件不存在: {args.input}")
        sys.exit(1)
    
    is_valid = verify_bigvul(args.input)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
