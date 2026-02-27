# 离线索引 Prompt 管理

本目录包含离线索引构建流程中使用的所有 Prompt 模板。

---

## 目录结构

```
prompts/offline/
├── README.md                  # 本文件
├── quality_check.txt          # 质量判断 prompt
├── blind/                     # 盲视生成 prompts
│   ├── data_flow.txt         # 数据流视角
│   ├── control_flow.txt      # 控制流视角
│   └── api_semantic.txt      # API/语义视角
└── distill/                   # 蒸馏 prompts
    ├── without_patch.txt     # 盲视蒸馏
    └── with_patch.txt        # 定向蒸馏（补丁引导）
```

---

## Prompt 用途说明

### 1. 盲视生成（Blind View Generation）

**位置**：`blind/*.txt`

**用途**：基于 Buggy Code 生成三视角的根因分析（RCA），**不使用**真实补丁。

**可用变量**：
- `{buggy_code}` - 有缺陷的代码
- `{view_type}` - 视角类型（data_flow/control_flow/api_semantic）

**三个视角**：

| 文件 | 视角 | 关注点 |
|------|------|--------|
| `data_flow.txt` | 数据流 | 变量依赖、污点流、定义-使用链 |
| `control_flow.txt` | 控制流 | 分支条件、循环、异常路径 |
| `api_semantic.txt` | API/语义 | API 误用、资源管理、契约违背 |

---

### 2. 盲视蒸馏（Blind Distillation）

**位置**：`distill/without_patch.txt`

**用途**：将盲视 RCA 蒸馏为更短、更标准化、便于检索的原型表示。

**可用变量**：
- `{blind_view}` - 盲视生成的 RCA 文本
- `{view_type}` - 视角类型

**输出格式**：建议 JSON 结构，包含：
- `root_cause` - 根因总结
- `signals` - 关键信号
- `entities` - 实体（变量、函数、API）
- `constraints` - 约束条件
- `fix_hints` - 修复提示

---

### 3. 质量判断（Quality Assessment）

**位置**：`quality_check.txt`

**用途**：评估盲视蒸馏结果的质量，决定是否需要定向蒸馏。

**可用变量**：
- `{blind_distilled_view}` - 盲视蒸馏后的结果
- `{buggy_code}` - 有缺陷的代码
- `{view_type}` - 视角类型

**输出格式**：JSON，包含：
```json
{
  "accurate": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "判断理由",
  "missing_aspects": ["缺失的关键点"]
}
```

**判断标准**：
1. 根因定位准确性
2. 证据充分性
3. 修复提示可行性
4. 完整性

---

### 4. 定向蒸馏（Patch-Guided Distillation）

**位置**：`distill/with_patch.txt`

**用途**：当盲视蒸馏质量不足时，使用真实补丁进行精炼。

**可用变量**：
- `{blind_view}` - 原始盲视 RCA
- `{blind_distilled_view}` - 盲视蒸馏结果（可选）
- `{buggy_code}` - 有缺陷的代码
- `{patch}` - 真实补丁
- `{view_type}` - 视角类型

**输出格式**：与盲视蒸馏相同的 JSON 结构，但增加：
- `patch_mechanism` - 补丁修复的机制/约束

**注意**：只在"根因与修复机制"层面利用补丁，不要逐行复述 diff。

---

## 完整流程示例

### 输入数据
```python
sample = {
    "buggy_code": "char buf[10]; strcpy(buf, input);",
    "patch": "char buf[256]; // ... size check ..."
}
```

### Step 1: 盲视生成
```
Prompt: blind/data_flow.txt
Input: buggy_code
Output: "数据流根因：buf 大小固定为 10，但 input 长度未检查，
         可能导致缓冲区溢出。source=input, sink=buf ..."
```

### Step 2: 盲视蒸馏
```
Prompt: distill/without_patch.txt
Input: blind_view from step 1
Output: {
  "root_cause": "Fixed-size buffer overflow",
  "signals": ["strcpy", "buf[10]", "unchecked input"],
  "fix_hints": ["Add size check", "Use strncpy"]
}
```

### Step 3: 质量判断
```
Prompt: quality_check.txt
Input: distilled from step 2 + buggy_code
Output: {
  "accurate": true,
  "confidence": 0.85,
  "reasoning": "正确识别缓冲区溢出，提到关键函数和边界检查"
}
```

### Step 4a: 如果 accurate=true
```
直接使用 step 2 的结果作为最终原型 → 向量化 → 写入索引
```

### Step 4b: 如果 accurate=false
```
Prompt: distill/with_patch.txt
Input: blind_view + buggy_code + patch
Output: {
  "root_cause": "Fixed-size buffer overflow",
  "patch_mechanism": "Increased buffer size to 256 and added input length validation",
  "signals": [...],
  "fix_hints": ["Increase buffer size", "Add bounds checking"]
}
→ 向量化 → 写入索引
```

---

## Prompt 编写指南

### 基本原则

1. **清晰的角色定义**："你是代码漏洞根因分析专家"
2. **具体的任务说明**："请生成数据流视角的盲视分析"
3. **结构化输出**：指定 JSON 或固定格式
4. **示例优先**：在 prompt 中包含 1-2 个示例
5. **约束明确**：说明长度限制、禁止事项

### 模板变量规范

使用 Python `str.format()` 风格的变量：
- `{buggy_code}` - 缺陷代码
- `{patch}` - 真实补丁
- `{view_type}` - 视角类型
- `{blind_view}` - 盲视 RCA
- `{blind_distilled_view}` - 盲视蒸馏结果

### 长度建议

| Prompt 类型 | 建议长度 | 理由 |
|------------|---------|------|
| 盲视生成 | 200-300 tokens | 需要详细指导分析方向 |
| 盲视蒸馏 | 150-200 tokens | 聚焦结构化输出 |
| 质量判断 | 250-350 tokens | 需明确判断标准 |
| 定向蒸馏 | 200-300 tokens | 需说明如何利用补丁 |

---

## 自定义 Prompt

### 方法 1: 直接修改文件

```bash
vim prompts/offline/blind/data_flow.txt
# 修改后保存，重新运行离线索引构建即可
```

### 方法 2: 通过配置指定

在 `config/config.yaml` 中：

```yaml
offline:
  multiview:
    prompts:
      data_flow: "custom_prompts/my_data_flow.txt"
      # ...
```

### 方法 3: 代码中动态设置

```python
from offline_indexing import MultiViewGenerator

generator = MultiViewGenerator(llm)

# 自定义 data_flow prompt
custom_prompt = """
你的自定义 prompt...
{buggy_code}
"""

generator.customize_prompt("data_flow", custom_prompt)
```

---

## Prompt 版本控制

建议为不同实验保存 prompt 版本：

```
prompts/offline/versions/
├── v1_baseline/
│   └── blind/data_flow.txt
├── v2_structured/
│   └── blind/data_flow.txt
└── v3_few_shot/
    └── blind/data_flow.txt
```

在配置中切换：

```yaml
offline:
  multiview:
    prompt_version: "v3_few_shot"  # 会加载 prompts/offline/versions/v3_few_shot/
```

---

## 常见问题

### Q1: Prompt 加载失败怎么办？

A: 系统会自动回退到代码内置的默认 prompt。检查日志中的警告信息：
```
WARNING: Failed to read prompt file prompts/offline/blind/data_flow.txt
WARNING: Some blind-view prompts missing; falling back to built-in templates.
```

### Q2: 如何测试 Prompt 效果？

A: 使用单样本测试：

```python
from offline_indexing import MultiViewGenerator
from models import LLMInterface

llm = LLMInterface(provider="openai", model_name="gpt-4")
gen = MultiViewGenerator(llm)

# 测试单个样本
buggy_code = "char buf[10]; strcpy(buf, input);"
result = gen.generate_single_view(buggy_code, "data_flow", None)
print(result)
```

### Q3: 支持其他语言的 Prompt 吗？

A: 支持。Prompt 文件是纯文本，可以用任何语言编写。例如创建英文版本：

```
prompts/offline/en/
├── blind/
│   ├── data_flow.txt
│   └── ...
```

然后在配置中指定：

```yaml
offline:
  multiview:
    prompts:
      data_flow: "prompts/offline/en/blind/data_flow.txt"
```

---

## 相关文档

- 质量引导蒸馏机制：`docs/QUALITY_GUIDED_DISTILLATION.md`
- 离线索引流程：`docs/FRAMEWORK_FLOW.md`
- 配置说明：`config/config.yaml`
- 使用指南：`USAGE.md`
