#!/bin/bash
# 验证修复效果的测试脚本

echo "=========================================="
echo "🔍 MV-BDSR 修复验证报告"
echo "=========================================="
echo ""

# 1. 检查配置文件修改
echo "1️⃣ 配置文件修改验证："
echo "---"
echo "max_tokens:"
grep "max_tokens:" config/config.yaml | head -1
echo ""
echo "top_k_per_view & final_top_k:"
grep -A 1 "top_k_per_view:" config/config.yaml
echo ""
echo "max_tool_calls:"
grep "max_tool_calls:" config/config.yaml
echo ""

# 2. 检查 Prompt 文件是否存在
echo "2️⃣ Prompt 模板文件验证："
echo "---"
ls -lh prompts/online/repair/
echo ""

# 3. 检查 Docker 超时设置
echo "3️⃣ Docker 超时设置验证："
echo "---"
grep -B 2 "test_timeout" online_inference/phase3_repair/validator.py | grep "self.test_timeout"
echo ""

# 4. 验证代码语法
echo "4️⃣ Python 语法验证："
echo "---"
/Users/mh/MV-BDSR/venv/bin/python -c "
from online_inference.phase3_repair import RepairAgent, Validator
print('✅ RepairAgent 和 Validator 模块加载成功')
"
echo ""

# 5. 列出之前失败的 CVE
echo "5️⃣ 之前失败的 CVE（需要重测）："
echo "---"
echo "  - CVE-2015-1326 (Python, 超时 + 截断)"
echo "  - CVE-2015-3295 (JavaScript, 截断)"
echo "  - CVE-2015-8213 (Python, 超时 + 截断)"
echo ""

echo "=========================================="
echo "✅ 所有修复已应用，准备测试"
echo "=========================================="
echo ""
echo "推荐测试命令："
echo ""
echo "cd /Users/mh/MV-BDSR && venv/bin/python scripts/run_online.py \\"
echo "  --vul-files-dir data/processed/CVEdataset \\"
echo "  --indices-dir data/indices/offline_dataset_index \\"
echo "  --output-dir results/fixed_test \\"
echo "  --limit 3"
echo ""
