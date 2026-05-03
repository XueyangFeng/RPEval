import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# 顶部 imports 旁边加上
import logging
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import OpenAIClient
from prompts_715 import LLM_judge_multiple_template
from prompts_715 import MLE_estimation_prompt_mmcq_template_v0, CPE_estimation_prompt_mmcq_template_v0, RPA_multi_generation_template 
from prompts import personalized_responser, personalized_responser_reminder, personalized_responser_cot, personalized_cot_policy_responser, Personalized_cot_mgen_responser 
from utils import (
    load_config,
    load_json,
    safe_parse_json
)
import random
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import re

METRIC_KEYS = ["OPB", "UPB", "RII", "FM", "VG", "Judge"]

def _nm_pair(match_str):
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(match_str))
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))

def _to_int_0_5(x):
    try:
        v = int(round(float(x)))
    except Exception:
        return None
    return max(0, min(5, v))



LABELS = ("A", "B", "C")

def aggregate_policy_ranking_single(rankings_dict: dict) -> dict:
    """
    单选聚合：每个模块一条排名（如 'BAC'）。
    分数：第1名+1，第2名+2，第3名+3；分数越低越优。
    首先按总分升序，再按第一名次数降序；仍平则随机。
    """
    score = defaultdict(int)
    first_place_counter = Counter()

    for module_name, ranking in rankings_dict.items():
        r = _normalize_rank(ranking)
        for i, label in enumerate(r):
            score[label] += i + 1
            if i == 0:
                first_place_counter[label] += 1

    sorted_labels = sorted(score.items(),
                           key=lambda x: (x[1], -first_place_counter[x[0]]))
    final_ranking = "".join([x[0] for x in sorted_labels])

    min_score = sorted_labels[0][1]
    tied = [k for k, v in score.items() if v == min_score]
    top_policy = tied[0]
    tie_break_applied = False
    tie_break_method = None

    if len(tied) > 1:
        max_first = max(first_place_counter.get(k, 0) for k in tied)
        final_candidates = [k for k in tied if first_place_counter.get(k, 0) == max_first]
        if len(final_candidates) == 1:
            top_policy = final_candidates[0]
            tie_break_applied = True
            tie_break_method = "first_place_count"
        else:
            top_policy = random.choice(final_candidates)
            tie_break_applied = True
            tie_break_method = "random"

    return {
        "individual_rankings": {k: _normalize_rank(v) for k, v in rankings_dict.items()},
        "aggregated_score": dict(score),
        "first_place_count": dict(first_place_counter),
        "final_ranking": final_ranking,
        "top_policy": top_policy,
        "tie_break_applied": tie_break_applied,
        "tie_break_method": tie_break_method
    }

def _normalize_rank(rank: str) -> str:
    """规范化为合法 A/B/C 全排列；非法则回退 'ABC'。"""
    if not isinstance(rank, str):
        return "ABC"
    r = rank.strip().upper().replace(" ", "")
    return r if len(r) == 3 and set(r) == set(LABELS) else "ABC"

def parse_pipe_rankings(pipe_str: str) -> list[str]:
    """
    把 'ABC|BAC|CBA' 解析为 ['ABC','BAC','CBA']，逐段兜底。
    空或非法则返回 ['ABC']。
    """
    if not isinstance(pipe_str, str):
        return ["ABC"]
    parts = [p.strip() for p in pipe_str.split("|") if p.strip()]
    parts = [_normalize_rank(p) for p in parts]
    return parts or ["ABC"]

def init_openai_client(config_path="utils/api_config.json", model_key="openai_41"):
    config = load_config(config_path)[model_key]
    return OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

def _normalize_rank(rank: str) -> str:
    """规范化为合法 A/B/C 全排列；非法则回退 'ABC'。"""
    if not isinstance(rank, str):
        return "ABC"
    r = rank.strip().upper().replace(" ", "")
    return r if len(r) == 3 and set(r) == set(LABELS) else "ABC"

def parse_pipe_rankings(pipe_str: str) -> list[str]:
    """
    把 'ABC|BAC|CBA' 解析为 ['ABC','BAC','CBA']，逐段兜底。
    空或非法则返回 ['ABC']。
    """
    if not isinstance(pipe_str, str):
        return ["ABC"]
    parts = [p.strip() for p in pipe_str.split("|") if p.strip()]
    parts = [_normalize_rank(p) for p in parts]
    return parts or ["ABC"]

def aggregate_policy_ranking_multi(rankings_dict: dict) -> dict:
    """
    多选聚合：rankings_dict 的 value 可以是
      - 'ABC|BAC|CBA' 这样的字符串，或
      - ['ABC','BAC','CBA'] 这样的列表。

    返回：
    {
      "per_dimension": [...],
      "top_policy": "..."   # 每维合并后的第一名拼接
    }
    """
    # 统一成列表
    lists_by_module = {
        m: (v if isinstance(v, list) else parse_pipe_rankings(v))
        for m, v in rankings_dict.items()
    }

    max_len = max((len(v) for v in lists_by_module.values()), default=0)
    if max_len == 0:
        # 兜底：退化为单选聚合（几乎不会触发）
        single = aggregate_policy_ranking_single({m: "ABC" for m in lists_by_module})
        return {"per_dimension": [single], "top_policy": single["top_policy"]}

    per_dimension = []
    top_letters = []

    for i in range(max_len):
        # 取第 i 段：缺段的模块用 'ABC' 兜底
        ith_dict = {m: (lst[i] if i < len(lst) else "ABC")
                    for m, lst in lists_by_module.items()}
        merged = aggregate_policy_ranking_single(ith_dict)
        merged_dim = dict(merged)
        merged_dim["dimension_index"] = i
        per_dimension.append(merged_dim)
        top_letters.append(merged["top_policy"])

    return {
        "per_dimension": per_dimension,
        "top_policy": "".join(top_letters)
    }

def _policy_from_rank_list(rank_list: list[str]) -> str:
    """
    从某模块的多段 ranking 列表生成该模块的 policy：
    规则：每段 ranking 取首字母并拼接；如 ["ABC","BAC","CBA"] -> "ABC"
    """
    if not isinstance(rank_list, list):
        return ""
    return "".join((blk[0] if isinstance(blk, str) and blk else "A") for blk in rank_list)

def rpa_multi(item, client):
    """
    多选版 RPA：期望 LLM 返回形如 "ABC|BAC|..." 的 ranking 字符串。
    输入:
        item: {
            "persona": <str 或 list[str]，原样注入 prompt>,
            "question": <str>
        }
        client: 具备 get_single_chat_completion(prompt: str) -> str 的对象
    返回:
        {
            "ranking_inputs": {module: "<pipe-string>"},
            "ranking_lists":  {module: ["ABC","BAC", ...]},
            "llm_outputs":    {module: parsed_json_plus},  # 追加 ranking_list 字段
            "mle_policy": <str>,                           # 例如 "AAAAAA"
            "cpe_policy": <str>,                           # 例如 "AABACA"
            "aggregated_result": <aggregate_policy_ranking_multi 的原样结果>,
            "policy": <aggregated_result["top_policy"]>    # 多维冠军拼接
        }
    """
    persona = item["persona"]
    question = item["question"]

    prompts = {
        "mle": MLE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
        "cpe": CPE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
    }

    rankings_pipe = {}   # module -> 原始 pipe 字符串
    rankings_list = {}   # module -> 解析后的列表形式
    outputs = {}         # module -> 解析后的 JSON（附加 ranking_list）

    for module, prompt in prompts.items():
        response = client.get_single_chat_completion(prompt)
        try:
            parsed = safe_parse_json(response)
            pipe_ranking = (parsed.get("ranking", "") or "").strip()
            rank_list = parse_pipe_rankings(pipe_ranking)   # 需你已有该工具；非法会兜底 ["ABC"]
            parsed["ranking_list"] = rank_list              # 便于调试
        except Exception:
            # JSON 解析失败的兜底
            pipe_ranking = "ABC"
            rank_list = ["ABC"]
            parsed = {"ranking": pipe_ranking, "ranking_list": rank_list, "raw": response}

        rankings_pipe[module] = pipe_ranking
        rankings_list[module] = rank_list
        outputs[module] = parsed

    # 逐维度聚合（需要你已有 aggregate_policy_ranking_multi）
    aggregation_result = aggregate_policy_ranking_multi(rankings_list)

    # 单模块 policy（逐段取第一名）
    mle_policy = _policy_from_rank_list(rankings_list.get("mle", []))
    cpe_policy = _policy_from_rank_list(rankings_list.get("cpe", []))

    return {
        "ranking_inputs": rankings_pipe,        # 原始管道字符串
        "ranking_lists": rankings_list,         # 列表化后的 ranking
        "llm_outputs": outputs,                 # LLM 原始 JSON（含 ranking_list）
        "aggregated_result": aggregation_result,
        "mle_policy": mle_policy,
        "cpe_policy": cpe_policy,
        "policy": aggregation_result["top_policy"],  # 多维冠军拼接，如 "AAAAAA"
    }


_single_call_pool = ThreadPoolExecutor(max_workers=64)  # 供所有调用复用

def call_with_timeout(fn, *args, timeout_s=630, **kwargs):
    """在线程里跑 fn，并在 timeout_s 内拿结果；超时抛 FuturesTimeout。"""
    future = _single_call_pool.submit(fn, *args, **kwargs)
    return future.result(timeout=timeout_s)

def retry_once_with_timeout(fn, *args, timeout_s=630, **kwargs):
    """
    先尝试一次（30s），超时或异常则重试一次；两次都不行就抛最后的异常。
    你也可以改成返回 None/特殊结构来“丢掉”该条数据。
    """
    last_err = None
    for attempt in (1, 2):
        try:
            return call_with_timeout(fn, *args, timeout_s=timeout_s, **kwargs)
        except Exception as e:
            print(f"retry_once_with_timeout failed: {e!r}")
            traceback.print_exc()   # <<< 打印完整堆栈
            last_err = e
            # 仅允许重试一次
    raise last_err

def init_openai_client(config_path="utils/api_config.json", model_key="openai_41"):
    config = load_config(config_path)[model_key]
    print("config", config)
    return OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

def evaluate_generated_response(item, prompt_type, generator_client, judge_client):
    question = item.get("question") or item.get("query")
    persona = item["persona"]
    intent = item["intent_type"]
    #intent = item.get("intent") or item.get("reason")
    print(f"[EVALUATE] question: {question} | Persona: {persona} | Intent: {intent}...")

    try:
        if prompt_type == "rpa":
            prompt_item = {**item, "persona": persona, "question": question}
            response_json = rpa_multi(prompt_item, generator_client)
            # 预测策略（聚合后的）
            pred_policy = (response_json.get("policy") or "").strip()
            gen_prompt = RPA_multi_generation_template.format(
                personas=persona,
                question=question,
                intent=pred_policy
            )
            generated_response = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=630
            )

        # === Phase 1: Generation（30s 超时 + 一次重试）===
        elif prompt_type == "reminder":
            gen_prompt = personalized_responser_reminder.format(persona=persona, question=question)
            generated_response = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=630
            )
        elif prompt_type == "zs":
            gen_prompt = personalized_responser.format(persona=persona, question=question)
            generated_response = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=630
            )
        elif prompt_type == "cot":
            gen_prompt = personalized_responser_cot.format(persona=persona, question=question)
            generated_response = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=630
            )
        elif prompt_type == "policy_cot":
            gen_prompt = Personalized_cot_mgen_responser.format(personas=persona, question=question)
            tmp = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=630
            )
            generated_response = safe_parse_json(tmp)["response"]
        else:
            generated_response = ""

        print("judging response...")

        # === Phase 2: Judge（同样 30s 超时 + 一次重试）===
        judge_prompt_input = LLM_judge_multiple_template.format(
            persona=persona,
            question=question,
            reply=generated_response,
            intent=intent,
            intent_type=intent,
        )
        judge_response = retry_once_with_timeout(
            judge_client.get_single_chat_completion, judge_prompt_input, timeout_s=630
        )

        try:
            judge_json = safe_parse_json(judge_response)
        except Exception:
            judge_json = {"match": False, "reason": "", "raw": judge_response}

        print(f"[JUDGE] Response: {judge_json}...")
        return {
            "persona": persona,
            "question": question,
            "ground_truth_intent": intent,
            #"ground_truth_type": intent_type,
            "generated_reply": generated_response,
            "judge_result": judge_json,
            "status": "ok",
        }

    except Exception as e:
        # 两次尝试都失败：丢掉该条，但返回一个可记录的“失败条目”，不中断主流程
        print(f"[DROP] {question} -> drop_after_timeout: {e}")
        #print(f"[DROP] {question} -> drop_after_timeout: {e!r}")
        traceback.print_exc()   # <<< 打印完整堆栈
        return {
            "persona": persona,
            "question": question,
            "ground_truth_intent": intent,
            #"ground_truth_type": intent_type,
            "generated_reply": None,
            "judge_result": {"match": False, "reason": f"drop_after_timeout: {e.__class__.__name__}: {e}"},
            "status": "dropped",
        }


def run_generation_evaluation(input_json_path, output_dir, generator_client, judge_client, model_name="default-model", prompt_type="reminder", max_workers=10, max_items=None):
    data = load_json(input_json_path)
    if max_items is not None:
        data = data[:max_items]

    results = []
    total = 0

    full_match_cnt = 0
    nm_sum_n = 0
    nm_sum_m = 0

    # 每个维度累计分数
    score_sums = {k: 0 for k in METRIC_KEYS}
    score_cnts = {k: 0 for k in METRIC_KEYS}

    print(f"🎯 Running generation + LLM-as-judge evaluation with {max_workers} threads on {len(data)} samples...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_generated_response, item, prompt_type, generator_client, judge_client) for item in data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result_entry = future.result()
            results.append(result_entry)

            total += 1
            jr = result_entry.get("judge_result", {})

            # 统计 full_match
            if isinstance(jr.get("full_match"), bool) and jr["full_match"]:
                full_match_cnt += 1

            # 统计 n/m
            n_i, m_i = _nm_pair(jr.get("match", "0/0"))
            nm_sum_n += n_i
            nm_sum_m += m_i

            # 累计每个维度的分数
            for k in METRIC_KEYS:
                v = _to_int_0_5(jr.get(k, None))
                if v is not None:
                    score_sums[k] += v
                    score_cnts[k] += 1

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}/{prompt_type}/gen_eval_results_{model_name}_{ts}_{prompt_type}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    # 计算两个原指标
    full_match_acc = (full_match_cnt / total) if total > 0 else 0.0
    nm_ratio = (nm_sum_n / nm_sum_m) if nm_sum_m > 0 else 0.0

    # 计算每个维度的平均奖励
    avg_scores = {}
    for k in METRIC_KEYS:
        c = score_cnts[k]
        avg_scores[k] = (score_sums[k] / c) if c > 0 else 0.0

    print("\n=== Generation-based Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Total samples: {total}")
    print(f"Full-match accuracy: {full_match_acc:.2%} ({full_match_cnt}/{total})")
    print(f"Aggregated n/m: {nm_sum_n}/{nm_sum_m} (ratio={nm_ratio:.2%})")
    print("Average reward per dimension:")
    for k, v in avg_scores.items():
        print(f"  - {k}: {v:.3f}")

    summary = {
        "timestamp": ts,
        "model": model_name,
        "total": total,
        "prompt_type": prompt_type,
        "full_match_cnt": full_match_cnt,
        "full_match_acc": full_match_acc,
        "nm_sum_n": nm_sum_n,
        "nm_sum_m": nm_sum_m,
        "nm_ratio": nm_ratio,
        "avg_scores": avg_scores,
        "output_file": output_path
    }

    summary_path = f"{output_dir}/gen_summary_log.jsonl"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as logf:
        logf.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"📄 Summary appended to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="qwen")
    parser.add_argument("--judge_model_key", type=str, default="openai_41")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", "--prompt_type", dest="prompt_type", type=str, default="cot")
    parser.add_argument("--persona", type=str, default="explicit")
    parser.add_argument("--threads", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = init_openai_client(model_key=args.model_key)
    run_generation_evaluation(
        args.input_path,
        args.output_dir,
        client,
        init_openai_client(model_key=args.judge_model_key),
        model_name=args.model_key,
        prompt_type=args.prompt_type,
        max_workers=args.threads,
        max_items=args.limit
    )
