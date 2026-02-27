# MV-BDSR: Multi-View Blind Distillation for Software Repair

多视图盲蒸馏的软件漏洞自动修复框架

## 项目架构

本项目分为**离线索引**和**在线推理**两大部分：

### 离线阶段
- 多视图盲生成（数据流、控制流、API语义）
- 盲蒸馏对齐与精炼
- 向量索引构建

### 在线阶段
1. **Phase 1**: 代理诊断（工具增强探测）
2. **Phase 2**: 检索与重排序（多视图检索 + 结构感知重排）
3. **Phase 3**: 迭代修复（测试驱动循环）

## 目录结构

```
MV-BDSR/
├── data_loader/          # Big-Vul 数据集处理
├── static_analysis/      # 静态分析工具封装（Joern/Tree-sitter）
├── offline_indexing/     # 离线索引构建
├── online_inference/     # 在线推理三阶段
│   ├── phase1_diagnosis/ # 诊断 Agent
│   ├── phase2_retrieval/ # 检索与重排
│   └── phase3_repair/    # 迭代修复
├── models/               # GraphCodeBERT 和 LLM 接口
├── utils/                # 工具函数
├── config/               # 配置文件
└── scripts/              # 运行脚本
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 离线索引构建
```bash
python scripts/run_offline.py --config config/config.yaml
```

### 在线修复
```bash
python scripts/run_online.py --buggy-code path/to/buggy.c
```

## 文档

详细的框架流程说明见 [docs/FRAMEWORK_FLOW.md](docs/FRAMEWORK_FLOW.md)
