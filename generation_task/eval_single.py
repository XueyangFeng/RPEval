import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import OpenAIClient
from prompts_715 import LLM_judge_template, RPA_generation_template
from prompts_715 import MLE_estimation_template_v1, CPE_recall_estimation_prompt, CPE_intent_estimation_prompt, MLE_estimation_template_v2
from prompts import personalized_responser, personalized_responser_reminder, personalized_responser_cot, personalized_cot_policy_responser
from utils import (
    load_config,
    load_json,
    safe_parse_json
)
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from collections import defaultdict, Counter


_single_call_pool = ThreadPoolExecutor(max_workers=64)  # 供所有调用复用


def aggregate_policy_ranking(rankings_dict):
    """
    :param rankings_dict: dict, keys = module names, values = ranking strings, e.g.
           {
             "mle": "BAC",
             "cpe_recall": "BCA",
             "cpe_intent": "CAB"
           }
    :return: dict with final_ranking, top_policy, scores, etc.
    """

    score = defaultdict(int)
    first_place_counter = Counter()

    # Step 1: 计算每个策略的得分和第一位出现次数
    for module_name, ranking in rankings_dict.items():
        for i, label in enumerate(ranking):
            score[label] += i + 1  # 位次从1开始累加
            if i == 0:
                first_place_counter[label] += 1

    # Step 2: 得到基础排序（得分越低越靠前），如 BAC
    sorted_labels = sorted(score.items(), key=lambda x: (x[1], -first_place_counter[x[0]]))
    final_ranking = ''.join([x[0] for x in sorted_labels])

    # Step 3: 判断是否需要 tie-break
    min_score = sorted_labels[0][1]
    tied = [k for k, v in score.items() if v == min_score]
    top_policy = tied[0]
    tie_break_applied = False
    tie_break_method = None

    if len(tied) > 1:
        # Check if first-place count can resolve
        max_first = max(first_place_counter.get(k, 0) for k in tied)
        final_candidates = [k for k in tied if first_place_counter.get(k, 0) == max_first]
        if len(final_candidates) == 1:
            top_policy = final_candidates[0]
            tie_break_applied = True
            tie_break_method = "first_place_count"
        else:
            # Still tie, use random
            top_policy = random.choice(final_candidates)
            tie_break_applied = True
            tie_break_method = "random"

    return {
        "individual_rankings": rankings_dict,
        "aggregated_score": dict(score),
        "first_place_count": dict(first_place_counter),
        "final_ranking": final_ranking,
        "top_policy": top_policy,
        "tie_break_applied": tie_break_applied,
        "tie_break_method": tie_break_method
    }

def rpa_new(item, client):
    persona = item["persona"]
    question = item["question"]

    prompts = {
        "mle": MLE_estimation_template_v2.format(persona=persona, question=question),
        "cpe_recall": CPE_recall_estimation_prompt.format(persona=persona, question=question),
        "cpe_intent": CPE_intent_estimation_prompt.format(persona=persona, question=question)
    }

    rankings = {}
    outputs = {}

    for module, prompt in prompts.items():
        response = client.get_single_chat_completion(prompt)
        try:
            parsed = safe_parse_json(response)
            ranking = parsed.get("ranking", "").strip()
            if not ranking or len(ranking) != 3:
                print(f"[{module}] ranking 格式异常: {ranking}")
                ranking = "ABC"  # fallback
        except Exception:
            print(f"[{module}] JSON 解析失败: {response}")
            ranking = "ABC"
            parsed = {"ranking": ranking, "raw": response}

        rankings[module] = ranking
        outputs[module] = parsed

    aggregation_result = aggregate_policy_ranking(rankings)

    return {
        "ranking_inputs": rankings,
        "llm_outputs": outputs,
        "aggregated_result": aggregation_result,
        "policy": aggregation_result["top_policy"]  # for统一结构
    }


def call_with_timeout(fn, *args, timeout_s=130, **kwargs):
    """在线程里跑 fn，并在 timeout_s 内拿结果；超时抛 FuturesTimeout。"""
    future = _single_call_pool.submit(fn, *args, **kwargs)
    return future.result(timeout=timeout_s)

def retry_once_with_timeout(fn, *args, timeout_s=130, **kwargs):
    """
    先尝试一次（30s），超时或异常则重试一次；两次都不行就抛最后的异常。
    你也可以改成返回 None/特殊结构来“丢掉”该条数据。
    """
    last_err = None
    for attempt in (1, 2):
        try:
            return call_with_timeout(fn, *args, timeout_s=timeout_s, **kwargs)
        except Exception as e:
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

def evaluate_generated_response(item, prompt_type, model, generator_client, judge_client):
    question = item.get("question") or item.get("query")
    persona = item["persona"]
    intent_type = item["intent_type"]
    intent_type = {
        "ignore": "忽略偏好",
        "support": "支持性偏好",
        "supportive": "支持性偏好",
        "调和偏好": "支持性偏好",
        "dominate": "以偏好为主线",
        "dominant": "以偏好为主线",
    }.get(str(intent_type).strip(), intent_type)
    intent = item.get("intent") or item.get("reason")
    print(f"[EVALUATE] question: {question} | Persona: {persona} | Intent Type: {intent_type} | Intent: {intent}...")

    # try:
    if prompt_type == "rpa":
            # 字母 → 中文
        option_map = {
            "A": "忽略偏好",
            "B": "支持性偏好",
            "C": "以偏好为主线"
        }
        response_json = rpa_new(item, generator_client)
        print(response_json)    
        llm_label = response_json.get("policy", "").strip()
        predict_intent = option_map.get(llm_label, "无效")
        response_prompt = RPA_generation_template.format(persona=persona, question=question, intent=predict_intent)
        print(response_prompt)
        print("=======")
        #raise

        # gen_prompt = generator_client.get_single_chat_completion(response_prompt)
        # print(f"[GENERATED] Response: {generated_response}...")
        # raise
        generated_response = retry_once_with_timeout(
            generator_client.get_single_chat_completion, response_prompt, timeout_s=130
        )
        print(f"[GENERATED] Response: {generated_response}...")
        #raise


    # === Phase 1: Generation（30s 超时 + 一次重试）===
    elif prompt_type == "reminder":
        gen_prompt = personalized_responser_reminder.format(persona=persona, question=question)
        generated_response = retry_once_with_timeout(
            generator_client.get_single_chat_completion, gen_prompt, timeout_s=130
        )
    elif prompt_type == "zs":
        gen_prompt = personalized_responser.format(persona=persona, question=question)
        generated_response = retry_once_with_timeout(
            generator_client.get_single_chat_completion, gen_prompt, timeout_s=130
        )
    elif prompt_type == "cot":
        gen_prompt = personalized_responser_cot.format(persona=persona, question=question)
        generated_response = retry_once_with_timeout(
            generator_client.get_single_chat_completion, gen_prompt, timeout_s=130
        )
    elif prompt_type == "policy_cot":
        gen_prompt = personalized_cot_policy_responser.format(persona=persona, question=question)
        if model == "qwen":
            resp_format = load_json("./config/vllm_formats.json")["policy_cot_gen"]
            #response = client.get_single_chat_completion(prompt, response_format=resp_format)
            tmp = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, response_format=resp_format, timeout_s=130
            )
        else:
            tmp = retry_once_with_timeout(
                generator_client.get_single_chat_completion, gen_prompt, timeout_s=130
            )          
        generated_response = safe_parse_json(tmp)["response"]
    else:
        generated_response = ""

    print("judging response...")

    # === Phase 2: Judge（同样 30s 超时 + 一次重试）===
    judge_prompt_input = LLM_judge_template.format(
        persona=persona,
        question=question,
        reply=generated_response,
        intent_type=intent_type,
        intent=intent
    )
    judge_response = retry_once_with_timeout(
        judge_client.get_single_chat_completion, judge_prompt_input, timeout_s=130
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
        "ground_truth_type": intent_type,
        "generated_reply": generated_response,
        "judge_result": judge_json,
        "status": "ok",
    }

    # except Exception as e:
    #     # 两次尝试都失败：丢掉该条，但返回一个可记录的“失败条目”，不中断主流程
    #     print(f"[DROP] {question} -> drop_after_timeout: {e}")
    #     return {
    #         "persona": persona,
    #         "question": question,
    #         "ground_truth_intent": intent,
    #         "ground_truth_type": intent_type,
    #         "generated_reply": None,
    #         "judge_result": {"match": False, "reason": f"drop_after_timeout: {e.__class__.__name__}: {e}"},
    #         "status": "dropped",
    #     }


def run_generation_evaluation(input_json_path, output_dir, generator_client, judge_client,
                              model_name="default-model", prompt_type="reminder",
                              max_workers=10, max_items=None):
    data = load_json(input_json_path)
    if max_items is not None:
        data = data[:max_items]

    results = []
    total = 0
    match = 0

    # 需要统计平均分的维度（与你的 judge JSON 对齐）
    METRICS = ["OPB", "UPB", "RII", "FM", "VG", "Judge"]

    # 分数累加器
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    # 按 GT 分类的分数
    categorized_scores = defaultdict(lambda: defaultdict(list))  # 分类存储：A/B/C -> 各项分数

    print(f"🎯 Running generation + LLM-as-judge evaluation with {max_workers} threads on {len(data)} samples...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_generated_response, item, prompt_type, model_name, generator_client, judge_client)
                   for item in data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result_entry = future.result()
            except Exception as e:
                result_entry = {
                    "persona": None,
                    "question": None,
                    "generated_reply": None,
                    "judge_result": {"match": False, "reason": f"main_loop_exception: {type(e).__name__}: {e}"},
                    "status": "dropped",
                }
                # 也可以顺手打印一下，方便看控制台
                print(f"[ERROR][main_loop] {type(e).__name__}: {e}")
            results.append(result_entry)

            total += 1
            jr = result_entry.get("judge_result", {}) or {}
            if jr.get("match"):
                match += 1

            # 累加各维度分数（0-5），自动忽略缺失或非数值
            for m in METRICS:
                val = jr.get(m)
                if isinstance(val, (int, float)):
                    metric_sums[m] += val
                    metric_counts[m] += 1

            # 按 GT 分类累计分数
            # gt = result_entry.get("ground_truth_intent")
            # if gt in ["忽略偏好", "支持性偏好", "以偏好为主线"]:
            #     for m in METRICS:
            #         val = jr.get(m)
            #         if isinstance(val, (int, float)):
            #             categorized_scores[gt][m].append(val)
            print(result_entry)
            gt = result_entry.get("ground_truth_type")
            intent_type = gt

            print(f"Evaluating: ground_truth_intent = {gt}, intent_type = {intent_type}")

            if intent_type in ["忽略偏好", "支持性偏好", "以偏好为主线"]:
                print(f"Matching intent_type: {intent_type}")
                if gt in ["忽略偏好", "支持性偏好", "以偏好为主线"]:
                    print(f"Matching ground_truth_intent: {gt}")
                    for m in METRICS:
                        val = jr.get(m)
                        if isinstance(val, (int, float)):
                            categorized_scores[gt][m].append(val)
            else:
                print(f"Unexpected intent_type: {intent_type}")


    # 计算各维度平均分
    avg_scores = {
        m: (metric_sums[m] / metric_counts[m]) if metric_counts[m] > 0 else None
        for m in METRICS
    }

    # 计算每个 GT 类别的平均分
    categorized_avg_scores = {}
    for gt, scores in categorized_scores.items():
        categorized_avg_scores[gt] = {
            m: (sum(values) / len(values)) if values else None
            for m, values in scores.items()
        }

    # 总体结果路径
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}/{prompt_type}/gen_eval_results_{model_name}_{ts}_{prompt_type}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    acc = match / total if total > 0 else 0.0
    print("\n=== Generation-based Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Total samples: {total}")
    print(f"Intent match accuracy: {match}/{total} = {acc:.2%}")
    print("Average scores by metric (0-5):")
    for m in METRICS:
        v = avg_scores[m]
        print(f"  {m}: {v:.3f}" if isinstance(v, (int, float)) else f"  {m}: N/A")

    # 输出按 GT 分类的平均分
    print("\n=== Average scores by Ground Truth (A/B/C) ===")
    for gt, avg_scores in categorized_avg_scores.items():
        print(f"\nGT = {gt}:")
        for metric, avg in avg_scores.items():
            print(f"  {metric}: {avg:.3f}" if avg is not None else f"  {metric}: N/A")

    summary = {
        "timestamp": ts,
        "model": model_name,
        "total": total,
        "prompt_type": prompt_type,
        "acc": acc,
        "avg_scores": avg_scores,  # 总体各维度的平均分
        "categorized_avg_scores": categorized_avg_scores,  # 分类后每个 GT 的平均分
        "output_file": output_path
    }

    summary_path = f"{output_dir}/gen_summary_log.jsonl"
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

    generator_client = init_openai_client(model_key=args.model_key)
    judge_client = init_openai_client(model_key=args.judge_model_key)

    run_generation_evaluation(
        args.input_path,
        args.output_dir,
        generator_client,
        judge_client,
        model_name=args.model_key,
        prompt_type=args.prompt_type,
        max_workers=args.threads,
        max_items=args.limit,
    )
