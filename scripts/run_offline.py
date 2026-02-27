#!/usr/bin/env python3
"""离线索引构建主脚本

API 调用流程（完整链路）：
========================
1. 用户运行: python scripts/run_offline.py --data-path ... --llm-provider openai
2. 脚本加载 .env 文件，读取 OPENAI_API_KEY 环境变量
3. 初始化 LLMInterface(provider="openai", api_key=从环境变量获取)
4. 对每个样本、每个视角执行：
   a) MultiViewGenerator.generate_single_view()  
      → self.llm.generate(prompt, ...)
      → LLMInterface.generate()
      → _generate_openai()
      → self.client.chat.completions.create(...)
      → OpenAI SDK 发送 POST https://api.openai.com/v1/chat/completions
      → Headers: {"Authorization": "Bearer sk-你的API密钥"}
   
   b) ViewDistillation._distill_without_patch()
      → 同样的调用链
   
   c) ViewDistillation.assess_quality()
      → 同样的调用链
   
   d) （可选）ViewDistillation._refine_with_patch()
      → 同样的调用链

每个样本、每个视角会调用 LLM 5-9 次（取决于质量评估结果）
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量（从 .env 文件）
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import argparse
from loguru import logger

from config.paths import ProjectPaths
from utils.helpers import load_yaml
from utils.logger import setup_logger
from data_loader import BigVulLoader
from models import LLMInterface, EmbeddingModel
from models.llm_interface import LLMAuthError, LLMError
from offline_indexing import (
    MultiViewGenerator,
    ViewDistillation,
    VectorStore,
    IndexBuilder,
)


def main():
    """离线索引主函数"""
    parser = argparse.ArgumentParser(description="MV-BDSR 离线索引构建")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Big-Vul 数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="索引输出目录（默认从配置读取）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="--output-dir 的别名",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mock"],
        help="LLM 提供商（使用 'mock' 进行离线测试）",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="只构建指定视角（逗号分隔），例如: data_flow 或 data_flow,control_flow",
    )

    args = parser.parse_args()

    # 加载配置
    config = load_yaml(args.config)
    
    # 设置日志
    setup_logger(
        log_file=config["logging"]["log_file"],
        level=config["logging"]["level"],
    )

    logger.info("启动离线索引构建流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据集: {args.data_path}")

    # 初始化路径
    paths = ProjectPaths()
    output_dir = args.output_dir or args.output or paths.indices

    # 加载数据集
    logger.info("加载数据集")
    data_loader = BigVulLoader(args.data_path)
    dataset = data_loader.load()
    logger.info(f"已加载 {len(dataset)} 个样本")

    # 转换为字典列表
    samples = []
    for idx, row in dataset.iterrows():
        samples.append({
            "id": idx,
            "buggy_code": row.get("buggy_code", ""),
            "patch": row.get("patch", ""),
        })

    # ★★★ 初始化 LLM 接口（这里决定了后续所有 API 调用的配置） ★★★
    logger.info("初始化模型")
    
    import os
    provider = (args.llm_provider or config["models"]["llm"]["provider"]).lower()
    api_key = config["models"]["llm"].get("api_key") or os.getenv("OPENAI_API_KEY")
    base_url = config["models"]["llm"].get("base_url")  # 可选的自定义端点
    
    if not api_key and provider in ["openai", "anthropic"]:
        logger.warning("⚠️  未找到 API Key（配置文件或环境变量）")
        logger.warning("   请设置 OPENAI_API_KEY 环境变量或在 config.yaml 中配置")
        logger.warning("   或使用 --llm-provider mock 进行离线自测")
    
    # 创建 LLM 接口（所有后续的 LLM 调用都通过这个实例）
    llm = LLMInterface(
        provider=provider,
        model_name=config["models"]["llm"]["model_name"],
        api_key=api_key,
        base_url=base_url,
        temperature=config["models"]["llm"]["temperature"],
        max_tokens=config["models"]["llm"]["max_tokens"],
    )

    # 创建 embedding 模型（用于向量化蒸馏后的视角）
    embedding_model = EmbeddingModel(
        model_name=config["models"]["embeddings"]["model_name"],
        device=config["experiment"]["device"],
    )

    # 初始化离线索引组件
    logger.info("初始化离线索引组件")
    
    # 多视角生成器（会调用 LLM 生成 3 个盲视）
    # 视角选择优先级：命令行 --views > config.yaml 的 offline.multiview.views > 默认全部
    selected_views = None
    if args.views:
        selected_views = [x.strip() for x in args.views.split(",") if x.strip()]
    else:
        selected_views = config.get("offline", {}).get("multiview", {}).get("views")

    if selected_views:
        logger.info(f"仅构建视角: {selected_views}")
    else:
        logger.info("构建全部默认视角")

    multiview_gen = MultiViewGenerator(llm, view_types=selected_views)
    
    # 视角蒸馏器（会调用 LLM 进行蒸馏和质量评估）
    distillation = ViewDistillation(
        llm,
        use_patch_refinement=config["offline"]["distillation"]["use_patch_refinement"],
    )

    # 向量存储（不调用 LLM，只负责 FAISS 索引）
    vector_store = VectorStore(
        dimension=config["models"]["embeddings"]["dimension"],
        index_type=config["offline"]["vector_store"]["index_type"],
        nlist=config["offline"]["vector_store"].get("nlist", 100),
    )

    # 索引构建器（协调整个流程）
    index_builder = IndexBuilder(
        multiview_gen,
        distillation,
        vector_store,
        embedding_model,
    )

    # ★★★ 构建索引（这里会发起大量 LLM API 请求） ★★★
    logger.info("开始构建索引（可能需要一段时间）...")
    try:
        index_builder.build_from_dataset(
            samples,
            output_dir,
            batch_size=args.batch_size,
        )
    except LLMAuthError as e:
        # API Key 无效/过期
        logger.error(f"LLM 认证失败（API Key 无效/过期）: {e}")
        logger.error("修复方法：设置有效的 OPENAI_API_KEY（或切换到 --llm-provider mock）")
        raise SystemExit(2) from e
    except LLMError as e:
        # 其他 LLM 错误
        logger.error(f"LLM 生成失败: {e}")
        logger.error("提示：可以稍后重试，或使用 --llm-provider mock 验证管线")
        raise SystemExit(2) from e

    # 打印统计信息
    logger.info("索引构建完成！")
    stats = index_builder.get_index_statistics()
    for view_type, view_stats in stats.items():
        logger.info(f"{view_type}: {view_stats}")

    logger.info(f"索引已保存到 {output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("下一步：")
    logger.info("1. 查看生成的索引: ls -lh " + str(output_dir))
    logger.info("2. 运行在线修复: python scripts/run_online.py --buggy-code '...'")
    logger.info("="*60)


if __name__ == "__main__":
    main()
