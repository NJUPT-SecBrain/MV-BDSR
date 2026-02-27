#!/usr/bin/env python3
"""验证数据集格式"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger


def verify_dataset(csv_path: str) -> bool:
    """
    验证数据集格式
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        验证是否通过
    """
    try:
        logger.info(f"读取数据集: {csv_path}")
        df = pd.read_csv(csv_path)
        
        required_cols = ['buggy_code', 'fixed_code']
        optional_cols = ['patch', 'id', 'cwe_id', 'cve_id', 'project', 'language']
        
        logger.info("=" * 70)
        logger.info("✅ 数据集验证报告")
        logger.info("=" * 70)
        logger.info(f"文件路径: {csv_path}")
        logger.info(f"总记录数: {len(df)}")
        logger.info(f"总列数: {len(df.columns)}")
        logger.info(f"\n列名: {df.columns.tolist()}")
        
        # 检查必需列
        logger.info(f"\n必需列检查:")
        all_required_present = True
        for col in required_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                null_count = df[col].isna().sum()
                logger.info(f"  ✓ {col}: {non_null} 非空, {null_count} 缺失")
                
                if null_count > 0:
                    logger.warning(f"    警告: {col} 列有 {null_count} 条缺失记录")
            else:
                logger.error(f"  ✗ {col}: 缺失!")
                all_required_present = False
        
        if not all_required_present:
            logger.error("\n✗ 验证失败: 缺少必需列")
            return False
        
        # 检查可选列
        logger.info(f"\n可选列检查:")
        for col in optional_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                logger.info(f"  ✓ {col}: {non_null} 非空")
            else:
                logger.info(f"  - {col}: 未找到")
        
        # 统计信息
        logger.info(f"\n代码长度统计:")
        buggy_lengths = df['buggy_code'].str.len()
        fixed_lengths = df['fixed_code'].str.len()
        
        logger.info(f"  Buggy Code:")
        logger.info(f"    平均长度: {buggy_lengths.mean():.0f} 字符")
        logger.info(f"    最小长度: {buggy_lengths.min():.0f} 字符")
        logger.info(f"    最大长度: {buggy_lengths.max():.0f} 字符")
        logger.info(f"    中位数: {buggy_lengths.median():.0f} 字符")
        
        logger.info(f"  Fixed Code:")
        logger.info(f"    平均长度: {fixed_lengths.mean():.0f} 字符")
        logger.info(f"    最小长度: {fixed_lengths.min():.0f} 字符")
        logger.info(f"    最大长度: {fixed_lengths.max():.0f} 字符")
        logger.info(f"    中位数: {fixed_lengths.median():.0f} 字符")
        
        # 漏洞类型分布
        if 'cwe_id' in df.columns:
            logger.info(f"\n漏洞类型分布 (Top 10):")
            cwe_counts = df['cwe_id'].value_counts().head(10)
            for cwe, count in cwe_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  {cwe}: {count} ({percentage:.1f}%)")
        
        # 项目分布
        if 'project' in df.columns:
            logger.info(f"\n项目分布 (Top 5):")
            project_counts = df['project'].value_counts().head(5)
            for project, count in project_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  {project}: {count} ({percentage:.1f}%)")
        
        # 数据质量检查
        logger.info(f"\n数据质量检查:")
        
        # 检查空代码
        empty_buggy = (df['buggy_code'].str.len() < 10).sum()
        empty_fixed = (df['fixed_code'].str.len() < 10).sum()
        
        if empty_buggy > 0:
            logger.warning(f"  ⚠ {empty_buggy} 条记录的 buggy_code 长度 < 10 字符")
        else:
            logger.info(f"  ✓ 所有 buggy_code 长度 >= 10 字符")
        
        if empty_fixed > 0:
            logger.warning(f"  ⚠ {empty_fixed} 条记录的 fixed_code 长度 < 10 字符")
        else:
            logger.info(f"  ✓ 所有 fixed_code 长度 >= 10 字符")
        
        # 检查相同代码
        same_code = (df['buggy_code'] == df['fixed_code']).sum()
        if same_code > 0:
            logger.warning(f"  ⚠ {same_code} 条记录的 buggy_code 与 fixed_code 完全相同")
        else:
            logger.info(f"  ✓ 所有记录的 buggy_code 与 fixed_code 都不相同")
        
        # 显示样例
        logger.info(f"\n样例数据 (第 1 条):")
        sample = df.iloc[0]
        logger.info(f"  ID: {sample.get('id', 'N/A')}")
        logger.info(f"  CWE: {sample.get('cwe_id', 'N/A')}")
        logger.info(f"  Project: {sample.get('project', 'N/A')}")
        logger.info(f"  Buggy Code (前 100 字符):")
        logger.info(f"    {str(sample['buggy_code'])[:100]}...")
        logger.info(f"  Fixed Code (前 100 字符):")
        logger.info(f"    {str(sample['fixed_code'])[:100]}...")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ 数据集格式验证通过!")
        logger.info("=" * 70)
        return True
        
    except FileNotFoundError:
        logger.error(f"✗ 文件未找到: {csv_path}")
        return False
    except Exception as e:
        logger.error(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证数据集格式")
    parser.add_argument(
        "csv_path",
        type=str,
        help="CSV 文件路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )
    
    success = verify_dataset(args.csv_path)
    sys.exit(0 if success else 1)
