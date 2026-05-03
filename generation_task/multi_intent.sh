#!/bin/bash
# ==========================================
# 🚀 LLM 多Prompt评估自动引导脚本
# 用法:
#   bash run_eval.sh [model_key] [persona] [prompt_type]
# 示例:
#   bash run_eval.sh qwen implicit cot
# ==========================================

cd "$(dirname "$0")/.."

# 传参
MODEL_KEY="gpt-4o" 
JUDGE_MODEL_KEY="gpt-41"            # 模型名
PERSONA="explicit"          # explicit / implicit
PROMPT="zs"                 # zs / cot / rsa / rule / mle / ...
THREADS=50
LIMIT=150

# 路径配置
EXPLICIT_INPUT="./benchmark_dataset/explicit_preference/multi_testset.json"
IMPLICIT_INPUT="./benchmark_dataset/implicit_preference/multi_testset.json"

if [ "$PERSONA" = "explicit" ]; then
    INPUT_PATH=$EXPLICIT_INPUT
else
    INPUT_PATH=$IMPLICIT_INPUT
fi

OUTPUT_DIR="./eval8/eval_mmcq/${PERSONA}/${PROMPT}"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "🧠 Running evaluation"
echo "Model:   $MODEL_KEY"
echo "Prompt:  $PROMPT"
echo "Persona: $PERSONA"
echo "Input:   $INPUT_PATH"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="

# 运行 Python 主脚本
python3 generation_task/eval_multi.py \
    --model_key "$MODEL_KEY" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --judge_model_key "$JUDGE_MODEL_KEY" \
    --threads "$THREADS" \
    --limit "$LIMIT"

echo "✅ Done. Results saved to: $OUTPUT_DIR"
