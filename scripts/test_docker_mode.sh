#!/bin/bash
# 测试 Docker 验证模式是否正常工作

set -e

echo "========================================="
echo "测试 Docker 验证模式"
echo "========================================="

# 检查 Docker
echo ""
echo "1. 检查 Docker 是否可用..."
if command -v docker &> /dev/null; then
    docker --version
    echo "✓ Docker 可用"
else
    echo "✗ Docker 未安装"
    exit 1
fi

# 检查配置文件
echo ""
echo "2. 检查配置文件..."
if [ -f "config/config.yaml" ]; then
    execution_mode=$(grep -A 10 "phase3_repair:" config/config.yaml | grep "execution_mode:" | awk '{print $2}' | tr -d '"')
    echo "当前执行模式: $execution_mode"
    
    if [ "$execution_mode" = "docker" ]; then
        echo "✓ Docker 模式已启用"
    else
        echo "! 当前为本地模式，如需测试 Docker 请修改 config/config.yaml"
        echo "  将 execution_mode: \"local\" 改为 execution_mode: \"docker\""
    fi
else
    echo "✗ 配置文件不存在"
    exit 1
fi

# 检查测试数据
echo ""
echo "3. 检查测试数据..."

cve_count=$(find data/processed/vul_files -name "test_cmds.json" | wc -l | tr -d ' ')
echo "找到 $cve_count 个 CVE 包含 test_cmds.json"

if [ $cve_count -gt 0 ]; then
    echo "✓ 测试数据已就绪"
    
    # 显示一个示例
    echo ""
    echo "示例 CVE（第一个）:"
    first_cve=$(find data/processed/vul_files -name "test_cmds.json" | head -1)
    cve_name=$(basename "$(dirname "$first_cve")")
    echo "  CVE: $cve_name"
    
    # 尝试提取 Docker 镜像
    if grep -q "# From" "$first_cve"; then
        # macOS(BSD grep) 不支持 -P，这里用 python 从 JSON 中提取镜像名
        image=$(
            python3 - "$first_cve" <<'PY'
import json, re, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
cmd = d.get("unit_test_cmd") or d.get("poc_test_cmd") or ""
m = re.search(r"#\s*From\s+([^\s]+)", cmd)
print(m.group(1) if m else "")
PY
        )
        echo "  Docker 镜像: $image"
        
        # 检查镜像是否存在
        if [ -z "$image" ]; then
            echo "  ! 未能从 test_cmds.json 提取到镜像名"
            echo "    请确保 unit_test_cmd/poc_test_cmd 中包含 '# From <image-name>' 注释"
        else
            if [ -n "$(docker images -q "$image" 2>/dev/null | head -1)" ]; then
                echo "  ✓ 镜像已在本地"
            else
                echo "  ! 镜像不在本地，首次运行将自动拉取"
            fi
        fi
    else
        echo "  ! test_cmds.json 中未找到 Docker 镜像注释"
        echo "    请确保包含格式为 '# From <image-name>' 的注释"
    fi
else
    echo "✗ 未找到包含 test_cmds.json 的 CVE"
    echo "  请先运行: python scripts/attach_test_cmds_to_vul_files.py"
fi

# 检查 input.json
echo ""
echo "4. 检查 input.json..."
if [ -f "data/raw/input.json" ]; then
    echo "✓ input.json 存在"
else
    echo "✗ input.json 不存在"
    echo "  Docker 模式需要此文件来获取 repo/commit 信息"
fi

# 运行测试建议
echo ""
echo "========================================="
echo "测试建议"
echo "========================================="

if [ "$execution_mode" = "docker" ] && [ $cve_count -gt 0 ]; then
    echo ""
    echo "✓ 环境已就绪，可以运行 Docker 模式测试："
    echo ""
    echo "  # 测试 1 个 CVE（Mock 模式，无需 API Key）"
    echo "  python scripts/run_online_batch.py \\"
    echo "    --vul-files-dir data/processed/vul_files \\"
    echo "    --indices-dir data/indices/test_index \\"
    echo "    --output-dir results/online_docker_test \\"
    echo "    --llm-provider mock \\"
    echo "    --limit 1"
    echo ""
elif [ "$execution_mode" != "docker" ]; then
    echo ""
    echo "如需启用 Docker 模式："
    echo "  1. 编辑 config/config.yaml"
    echo "  2. 将 execution_mode: \"local\" 改为 execution_mode: \"docker\""
    echo "  3. 重新运行此测试脚本"
    echo ""
else
    echo ""
    echo "! 请先准备测试数据："
    echo "  python scripts/attach_test_cmds_to_vul_files.py"
    echo ""
fi

echo "========================================="
echo "测试完成"
echo "========================================="
