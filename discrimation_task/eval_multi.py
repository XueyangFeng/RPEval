import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict, Counter
import re
import time
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import OpenAIClient
from utils import load_config, load_json, safe_parse_json
from prompts import Personalized_intent_mmcq, Personalized_intent_mmcq_reminder, Personalized_cot_mmcq_prompt, Personalized_intent_mmcq_reminder_implicit, Personalized_cot_mmcq_implicit_prompt, Personalized_intent_mmcq_implicit
from prompts_715 import MLE_estimation_prompt_mmcq_template_v0, CPE_estimation_prompt_mmcq_template_v0, MLE_estimation_prompt_mmcq_implicit_template_v0, CPE_estimation_prompt_mmcq_implicit_template_v0
import random


LABELS = ("A", "B", "C")
PER_TASK_TIMEOUT = 1880  # 每个样本最多等待 60s

def _is_ia_str(s: str) -> bool:
    """IA = 非空且全为 'A'"""
    s = str(s or "").strip().upper()
    return bool(s) and set(s) == {"A"}

def _slot_match(gt: str, pred: str):
    """逐位比较（按较短长度对齐），返回 (matched, total)"""
    gt_s = str(gt or "")
    pred_s = str(pred or "")
    L = min(len(gt_s), len(pred_s))
    if L == 0:
        return 0, 0
    matched = sum(1 for a, b in zip(gt_s[:L], pred_s[:L]) if a == b)
    return matched, L

def _slot_match_recall(gt: str, pred: str):
    """
    召回视角：分母用 len(gt)
    返回 (matched_on_gt, gt_total)
    """
    gt_s = str(gt or "")
    pred_s = str(pred or "")
    gt_total = len(gt_s)
    if gt_total == 0:
        return 0, 0
    # 只比较前 len(gt) 位；pred 更长的部分不影响召回
    matched = sum(1 for a, b in zip(gt_s, pred_s) if a == b)
    return matched, gt_total

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
        # 改成：按第一个模块的顺序挑最靠前的
        #print(rankings_dict);
        #raise
        # for label in rankings_dict["mle"]:
        #     if label in final_candidates:
        #         top_policy = label
        #         tie_break_applied = True
        #         tie_break_method = "first_module_priority"
        #         break
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

# def rpa_multi(item, model, client):
#     """
#     多选版 RPA：期望 LLM 返回形如 "ABC|BAC|..." 的 ranking 字符串。
#     输入:
#         item: {
#             "persona": <str 或 list[str]，原样注入 prompt>,
#             "question": <str>
#         }
#         client: 具备 get_single_chat_completion(prompt: str) -> str 的对象
#     返回:
#         {
#             "ranking_inputs": {module: "<pipe-string>"},
#             "ranking_lists":  {module: ["ABC","BAC", ...]},
#             "llm_outputs":    {module: parsed_json_plus},  # 追加 ranking_list 字段
#             "mle_policy": <str>,                           # 例如 "AAAAAA"
#             "cpe_policy": <str>,                           # 例如 "AABACA"
#             "aggregated_result": <aggregate_policy_ranking_multi 的原样结果>,
#             "policy": <aggregated_result["top_policy"]>    # 多维冠军拼接
#         }
#     """
#     persona = item["persona"]
#     question = item["question"]

#     prompts = {
#         "mle": MLE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
#         "cpe": CPE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
#     }

#     rankings_pipe = {}   # module -> 原始 pipe 字符串
#     rankings_list = {}   # module -> 解析后的列表形式
#     outputs = {}         # module -> 解析后的 JSON（附加 ranking_list）

#     for module, prompt in prompts.items():
#         if model == "qwen":
#             format = load_json(f"./config/vllm_formats.json")["rpa_mmcq"]
#             response = client.get_single_chat_completion(prompt, response_format=format)
#         else:    
#             response = client.get_single_chat_completion(prompt)
#         try:
#             parsed = safe_parse_json(response)
#             pipe_ranking = (parsed.get("ranking", "") or "").strip()
#             rank_list = parse_pipe_rankings(pipe_ranking)   # 需你已有该工具；非法会兜底 ["ABC"]
#             parsed["ranking_list"] = rank_list              # 便于调试
#         except Exception:
#             # JSON 解析失败的兜底
#             # JSON 解析失败，标记 invalid
#             parsed = {"ranking": "", "ranking_list": [], "raw": response, "_invalid": True}
#             pipe_ranking = "ABC"           # 兜底
#             rank_list = ["ABC"]            # 兜底
#             # pipe_ranking = ""
#             # rank_list = []
#             # return parsed
#             # pipe_ranking = "ABC"
#             # rank_list = ["ABC"]
#             # parsed = {"ranking": pipe_ranking, "ranking_list": rank_list, "raw": response}

#         rankings_pipe[module] = pipe_ranking
#         rankings_list[module] = rank_list
#         outputs[module] = parsed

#     # 逐维度聚合（需要你已有 aggregate_policy_ranking_multi）
#     aggregation_result = aggregate_policy_ranking_multi(rankings_list)

#     # 单模块 policy（逐段取第一名）
#     mle_policy = _policy_from_rank_list(rankings_list.get("mle", []))
#     cpe_policy = _policy_from_rank_list(rankings_list.get("cpe", []))

#     return {
#         "ranking_inputs": rankings_pipe,        # 原始管道字符串
#         "ranking_lists": rankings_list,         # 列表化后的 ranking
#         "llm_outputs": outputs,                 # LLM 原始 JSON（含 ranking_list）
#         "aggregated_result": aggregation_result,
#         "mle_policy": mle_policy,
#         "cpe_policy": cpe_policy,
#         "policy": aggregation_result["top_policy"],  # 多维冠军拼接，如 "AAAAAA"
#     }


def rpa_multi(persona_type, item, model, client):
    """
    多选版 RPA：优先使用双模块聚合；若仅一模块有效则用该模块；若两者皆无效则丢弃本条。
    返回结构与原版兼容，并额外包含 used_modules / reason 字段。
    """
    persona = item["persona"]
    question = item["question"]
    if persona_type == "explicit_persona":
        prompts = {
            "mle": MLE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
            "cpe": CPE_estimation_prompt_mmcq_template_v0.format(personas=persona, question=question),
        }
    else:
        prompts = {
            "mle": MLE_estimation_prompt_mmcq_implicit_template_v0.format(personas=persona, question=question), 
            "cpe": CPE_estimation_prompt_mmcq_implicit_template_v0.format(personas=persona, question=question),
        }

    rankings_pipe = {}   # module -> 原始 pipe 字符串
    rankings_list = {}   # module -> 解析后的列表形式
    outputs = {}         # module -> 解析后的 JSON（附加 ranking_list）
    valid = {}           # module -> bool 是否有效

    for module, prompt in prompts.items():
        if model == "qwen":
            if persona_type == "explicit_persona":
                resp_format = load_json("./config/vllm_formats.json")["rpa_mmcq"]
            else:
                resp_format = load_json("./config/vllm_formats.json")["rpa_mmcq_implicit"]
            response = client.get_single_chat_completion(prompt, response_format=resp_format)
        else:
            response = client.get_single_chat_completion(prompt)

        try:
            parsed = safe_parse_json(response)
            pipe_ranking = (parsed.get("ranking", "") or "").strip()
            rank_list = parse_pipe_rankings(pipe_ranking) if pipe_ranking else []
            parsed["ranking_list"] = rank_list
            # 有效条件：成功解析 + 至少有一个分段
            is_ok = (not parsed.get("_invalid")) and len(rank_list) > 0
        except Exception:
            # 解析失败：标记无效，不给兜底 ABC，防止污染聚合
            parsed = {"ranking": "", "ranking_list": [], "raw": response, "_invalid": True}
            pipe_ranking, rank_list, is_ok = "", [], False

        rankings_pipe[module] = pipe_ranking
        rankings_list[module] = rank_list
        outputs[module] = parsed
        valid[module] = is_ok

    num_valid = sum(1 for v in valid.values() if v)

    # --- 分支 1：两个模块都有效 -> 双模块聚合 ---
    if num_valid == 2:
        aggregation_result = aggregate_policy_ranking_multi(rankings_list)
        mle_policy = _policy_from_rank_list(rankings_list["mle"])
        cpe_policy = _policy_from_rank_list(rankings_list["cpe"])
        policy = aggregation_result["top_policy"]
        used = ["mle", "cpe"]
        reason = "both modules valid; multi-module aggregation"
    # --- 分支 2：仅一个模块有效 -> 单模块聚合 ---
    elif num_valid == 1:
        only = "mle" if valid.get("mle") else "cpe"
        single_dict = {only: rankings_list[only]}
        aggregation_result = aggregate_policy_ranking_multi(single_dict)  # 兼容：只传一个模块
        mle_policy = _policy_from_rank_list(rankings_list.get("mle", [])) if only == "mle" else ""
        cpe_policy = _policy_from_rank_list(rankings_list.get("cpe", [])) if only == "cpe" else ""
        policy = aggregation_result["top_policy"]
        used = [only]
        reason = f"only {only} valid; fallback to single-module aggregation"
    # --- 分支 3：两个模块都无效 -> 丢弃本条 ---
    else:
        return {
            "_invalid": True,
            "reason": "both modules invalid; drop this sample",
            "ranking_inputs": rankings_pipe,
            "ranking_lists": rankings_list,
            "llm_outputs": outputs,
        }

    return {
        "ranking_inputs": rankings_pipe,         # 原始管道字符串
        "ranking_lists": rankings_list,          # 列表化后的 ranking
        "llm_outputs": outputs,                  # LLM 原始 JSON（含 ranking_list / _invalid）
        "aggregated_result": aggregation_result, # per_dimension + top_policy 等
        "mle_policy": mle_policy,                # MLE 单模块策略（若未用则为空串）
        "cpe_policy": cpe_policy,                # CPE 单模块策略（若未用则为空串）
        "policy": policy,                        # 最终用于评估的策略（聚合后的 top_policy）
        "used_modules": used,                    # 实际参与聚合的模块
        "reason": reason,                        # 分支说明
    }


def evaluate_mm_policy(gt: str, pred: str, mode: str = "recall"):
    gt = str(gt); pred = str(pred)

    if mode == "strict":
        if len(gt) != len(pred):
            return {"total": len(gt), "correct": 0, "accuracy": 0.0, "match_list": [False]*len(gt)}
        L = len(gt)
    elif mode == "recall":
        L = len(gt)
    else:  # precision-like (原逻辑)
        L = min(len(gt), len(pred))

    if L == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "match_list": []}

    match_list = [g == p for g, p in zip(gt[:L], pred[:L])]
    correct = sum(match_list)

    return {"total": L, "correct": correct, "accuracy": correct / L, "match_list": match_list}


def evaluate_single_item(item, prompt_type, persona_type, model, client):
    question = item["question"]
    if persona_type == "explicit_persona":
        personas = item.get("persona", "")
    elif persona_type == "implicit_persona":
        personas = item.get("implicit_persona", "")
    else:
        personas = ""

    #gt_policy = "A"* len(personas) # only for testing, replace with actual ground truth
    gt_policy = item.get("intent_type", "")  # e.g., "AABCC"

    if prompt_type == "rpa":
        prompt_item = {**item, "persona": personas, "question": question}
        response_json = rpa_multi(persona_type, prompt_item, model, client)
        # 如果无效，直接返回原始错误对象（或给出占位）
        if response_json.get("_invalid"):
            return response_json
        pred_policy = (response_json.get("policy") or "").strip()

        # 模块级策略（可选）
        mle_policy = (response_json.get("mle_policy") or "").strip()
        cpe_policy = (response_json.get("cpe_policy") or "").strip()

        # 评估（主结果）
        match_result = evaluate_mm_policy(gt_policy, pred_policy)

        # 评估（分模块，可为空）
        mle_match = evaluate_mm_policy(gt_policy, mle_policy) if mle_policy else None
        cpe_match = evaluate_mm_policy(gt_policy, cpe_policy) if cpe_policy else None

        # 安全打印
        try:
            acc = match_result.get("accuracy", 0.0)
            print(f"[COMPARE] GT: {gt_policy} | PRED: {pred_policy} | ACC: {acc:.2%}")
        except Exception:
            print(f"[COMPARE] GT: {gt_policy} | PRED: {pred_policy} | ACC: <N/A>")

        return {
            "persona": item.get("persona"),
            "question": item.get("question"),
            "ground_truth": gt_policy,
            "prediction": pred_policy,
            "match": match_result,          # 聚合后的匹配结果
            "mle_policy": mle_policy,       # 可选：MLE 的多维拼接策略
            "mle_match": mle_match,         # 可选：MLE 与 GT 的匹配
            "cpe_policy": cpe_policy,       # 可选：CPE 的多维拼接策略
            "cpe_match": cpe_match,         # 可选：CPE 与 GT 的匹配
            "llm_output": response_json     # rpa_multi 的原始输出（含 per-dimension 等）
        }
    else:
        if prompt_type == "zs":
            if persona_type == "explicit_persona":
                mmcq_template = Personalized_intent_mmcq
            else:  
                mmcq_template = Personalized_intent_mmcq_implicit
        elif prompt_type == "reminder":
            if persona_type == "explicit_persona":
                mmcq_template = Personalized_intent_mmcq_reminder
            else:  
                mmcq_template = Personalized_intent_mmcq_reminder_implicit
        elif prompt_type == "cot":
            if persona_type == "explicit_persona":
                mmcq_template = Personalized_cot_mmcq_prompt
            else:  
                mmcq_template = Personalized_cot_mmcq_implicit_prompt
        elif prompt_type == "mle_v0":
            mmcq_template = MLE_estimation_prompt_mmcq_template_v0
        elif prompt_type == "cpe_v0":
            mmcq_template = CPE_estimation_prompt_mmcq_template_v0
        prompt = mmcq_template.format(personas=personas, question=question)
        if model == "qwen":
            if persona_type == "explicit_persona":
                format = load_json(f"./config/vllm_formats.json")["mmcq"]
            else:
                format = load_json(f"./config/vllm_formats.json")["mmcq_implicit"]
            response = client.get_single_chat_completion(prompt, response_format=format)
        else:
            response = client.get_single_chat_completion(prompt)
        #response = client.get_single_chat_completion(prompt)

        try:
            response_json = safe_parse_json(response)
            pred_policy = response_json.get("policy", "").strip()
        except Exception:
            # print(f"[ERROR] JSON parse failed: {response}")
            # pred_policy = ""
            # response_json = {"policy": "", "raw": response}
            print(f"[ERROR] JSON parse failed: {response}")
            pred_policy = ""
            response_json = {"policy": "", "raw": response, "_invalid": True}
            return response_json
        match_result = evaluate_mm_policy(gt_policy, pred_policy)

        print(f"[COMPARE] GT: {gt_policy} | PRED: {pred_policy} | ACC: {match_result['accuracy']:.2%}")

        return {
            "personas": personas,
            "question": question,
            "ground_truth": gt_policy,
            "prediction": pred_policy,
            "match": match_result,
            "llm_output": response_json
        }


def run_evaluation(input_json_path, output_dir, client, model_name="default-model", prompt_type="reminder", persona_type="explicit_persona", max_workers=10, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    data = load_json(input_json_path)
    if limit is not None:
        data = data[:limit]

    results = []
    all_gt = []
    all_pred = []

    print(f"🔁 Running MMCQ evaluation with {max_workers} threads... (limit={limit})")



    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_map = {executor.submit(evaluate_single_item, item, prompt_type, persona_type, model_name, client): idx
                    for idx, item in enumerate(data)}
        start_time = {f: time.time() for f in future_map}
        pending = set(future_map.keys())

        pbar = tqdm(total=len(pending), desc="Evaluating")

        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
            )

            # 收割已完成
            for f in done:
                try:
                    result = f.result()
                    results.append(result)
                    all_gt.extend(result.get("ground_truth",""))
                    all_pred.extend(result.get("prediction",""))
                except Exception as e:
                    print(f"[WARN] worker failed: {e!r}")
                    placeholder = {"ground_truth": "", "prediction": "", "_invalid": True}
                    results.append(placeholder)
                    all_gt.extend("")
                    all_pred.extend("")
                pbar.update(1)

            # for f in done:
            #     try:
            #         result = f.result()
            #     except Exception as e:
            #         #result = {"ground_truth": "", "prediction": "", "_invalid": True}
            #         print(f"[WARN] worker failed: {e!r}")
            #         results.append({"ground_truth": "", "prediction": "", "_invalid": True})
            #     results.append(result)
            #     all_gt.extend(result.get("ground_truth",""))
            #     all_pred.extend(result.get("prediction",""))
            #     pbar.update(1)

            # 超时任务：标记、占位并停止等待
            now = time.time()
            for f in list(pending):
                if now - start_time[f] > PER_TASK_TIMEOUT:
                    # 1) 从等待集合中移除
                    pending.remove(f)
                    # 2) 记录并占位
                    print(f"[TIMEOUT] cancelled job >{PER_TASK_TIMEOUT}s (idx={future_map[f]})")
                    #results.append({"ground_truth": "", "prediction": ""})
                    placeholder = {"ground_truth": "", "prediction": "", "_invalid": True, "_reason": "timeout"}
                    results.append(placeholder)
                    all_gt.extend("")
                    all_pred.extend("")
                    pbar.update(1)
                    # 3) 尝试取消（若未开始可取消；已开始取消不了也没关系，因为我们不再等待它）
                    f.cancel()

        pbar.close()

    finally:
        # 关键：不要阻塞等线程自然结束；直接告诉线程池取消未开始任务并不等待
        executor.shutdown(wait=False, cancel_futures=True)

    groups = {
        "IA":  {"samples": 0, "all_correct": 0, "pos_matched": 0, "pos_total": 0},
        "LKN": {"samples": 0, "all_correct": 0, "pos_matched": 0, "pos_total": 0},
    }
    for r in results:
        if r.get("_invalid"):   # 跳过无效样本
            continue
        gt = str(r.get("ground_truth", ""))
        pred = str(r.get("prediction", ""))
        grp = "IA" if _is_ia_str(gt) else "LKN"
        g = groups[grp]
        g["samples"] += 1
        #if gt == pred:
        # matched, total_slots = _slot_match(gt, pred)  # 仍是 min 对齐
        # if total_slots > 0 and matched == total_slots:
        #     g["all_correct"] += 1
        matched, total_slots = _slot_match_recall(gt, pred)  # 按较短长度对齐
        g["pos_matched"] += matched
        g["pos_total"] += total_slots
        if total_slots > 0 and matched == total_slots:
            g["all_correct"] += 1

    # 打印两组
    print("\n--- Grouped Accuracy (Multi-Preference: IA vs LKN) ---")
    for grp in ["IA", "LKN"]:
        g = groups[grp]
        em = (g["all_correct"] / g["samples"]) if g["samples"] > 0 else 0.0
        slot_acc_grp = (g["pos_matched"] / g["pos_total"]) if g["pos_total"] > 0 else 0.0
        print(f"{grp}: 全对 {g['all_correct']}/{g['samples']} = {em:.2%} | "
              f"逐位 {g['pos_matched']}/{g['pos_total']} = {slot_acc_grp:.2%}")

    # 矫正整体（与两组逐位加权一致）
    tot_matched = groups["IA"]["pos_matched"] + groups["LKN"]["pos_matched"]
    tot_slots   = groups["IA"]["pos_total"]   + groups["LKN"]["pos_total"]
    acc = (tot_matched / tot_slots) if tot_slots > 0 else 0.0
    print(f"Combined slot acc (IA+LKN, micro): {tot_matched}/{tot_slots} = {acc:.2%}")

    # 统一设置 total/correct，避免 NameError

    total = tot_slots
    correct = tot_matched

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}/intent_eval_results_{model_name}_{ts}_{prompt_type}_{persona_type}.json"
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print("\n=== MMCQ Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Total slots: {total}")
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")

    summary = {
        "timestamp": ts,
        "model": model_name,
        "prompt": prompt_type,
        "persona_type": persona_type,
        "total_slots": total,
        "accuracy": acc,  # 矫正后的整体逐位准确率
        "output_file": output_path,
    }
    invalid_count = sum(1 for r in results if r.get("_invalid"))
    summary["invalid_samples"] = invalid_count

    # 如果是多偏好，附带 IA/LKN 小结（可选）

    ia_em   = (groups["IA"]["all_correct"]  / groups["IA"]["samples"])  if groups["IA"]["samples"]  else 0.0
    ia_slot = (groups["IA"]["pos_matched"]  / groups["IA"]["pos_total"]) if groups["IA"]["pos_total"] else 0.0
    lkn_em  = (groups["LKN"]["all_correct"] / groups["LKN"]["samples"]) if groups["LKN"]["samples"] else 0.0
    lkn_slot= (groups["LKN"]["pos_matched"] / groups["LKN"]["pos_total"]) if groups["LKN"]["pos_total"] else 0.0

    summary.update({
        "IA": {
            "samples": groups["IA"]["samples"],
            "all_correct": groups["IA"]["all_correct"],
            "all_correct_rate": ia_em,
            "pos_matched": groups["IA"]["pos_matched"],
            "pos_total": groups["IA"]["pos_total"],
            "slot_accuracy": ia_slot,
        },
        "LKN": {
            "samples": groups["LKN"]["samples"],
            "all_correct": groups["LKN"]["all_correct"],
            "all_correct_rate": lkn_em,
            "pos_matched": groups["LKN"]["pos_matched"],
            "pos_total": groups["LKN"]["pos_total"],
            "slot_accuracy": lkn_slot,
        },
        "combined": {
            "total_samples": groups["IA"]["samples"] + groups["LKN"]["samples"],
            "correct_samples": groups["IA"]["all_correct"] + groups["LKN"]["all_correct"],
            "total_slots": groups["IA"]["pos_total"] + groups["LKN"]["pos_total"],
            "correct_slots": groups["IA"]["pos_matched"] + groups["LKN"]["pos_matched"],
            "slot_accuracy": (groups["IA"]["pos_matched"] + groups["LKN"]["pos_matched"])/(groups["IA"]["pos_total"] + groups["LKN"]["pos_total"]) if (groups["IA"]["pos_total"] + groups["LKN"]["pos_total"]) > 0 else 0.0,
            "all_correct": groups["IA"]["all_correct"] + groups["LKN"]["all_correct"],
            "all_correct_rate": (groups["IA"]["all_correct"] + groups["LKN"]["all_correct"]) / (groups["IA"]["samples"] + groups["LKN"]["samples"]) if (groups["IA"]["samples"] + groups["LKN"]["samples"]) > 0 else 0.0,
        }   
    })

    summary_path = f"{output_dir}/mmcq_summary_log.jsonl"
    with open(summary_path, "a", encoding="utf-8") as logf:
        logf.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="qwen")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="cot")
    parser.add_argument("--persona", type=str, default="implicit")
    parser.add_argument("--threads", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = init_openai_client(model_key=args.model_key)
    persona_type = f"{args.persona}_persona"

    run_evaluation(
        args.input_path,
        args.output_dir,
        client,
        model_name=args.model_key,
        prompt_type=args.prompt_type,
        persona_type=persona_type,
        max_workers=args.threads,
        limit=args.limit
    )
