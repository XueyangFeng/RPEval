import json
import math
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import OpenAIClient
from prompts import Personalized_intent_mcq, Personalized_intent_mcq_reminder, cot_reasoner_prompt, Personalized_rule_cot_mcq_prompt
from prompts_528 import relevance_reasoner_prompt, rational_speech_acts_prompt
from prompts_715 import MLE_estimation_template_v1, CPE_recall_estimation_prompt, CPE_intent_estimation_prompt, MLE_estimation_template_v2
from utils import (
    load_config,
    load_json,
    safe_parse_json
)
from rpa_func import rpa
import random
from collections import defaultdict, Counter

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

def init_openai_client(config_path="utils/api_config.json", model_key="openai_41"):
    config = load_config(config_path)[model_key]
    return OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"],
        request_timeout=20, max_retries=2, verbose=True  # 打开详细日志
    )

def evaluate_single_item(item, prompt_type, persona_type, model, client):
    question = item["question"]
    if persona_type == "explicit_persona":
        persona = item["persona"]
    elif persona_type == "implicit_persona":
        persona = item["implicit_persona"]
    else:
        persona = ""

    ground_truth_text = item["intent_type"]

    # 字母 → 中文
    option_map = {
        "A": "忽略偏好",
        "B": "支持性偏好",
        "C": "以偏好为主线"
    }

    # 中文 → 字母
    label_to_option = {
        **{v: k for k, v in option_map.items()},
        "ignore": "A",
        "support": "B",
        "supportive": "B",
        "支持性偏好": "B",
        "dominate": "C",
        "dominant": "C",
    }
    ground_truth_letter = label_to_option.get(str(ground_truth_text).strip(), "")

    if prompt_type == "rpa":
        response_json = rpa(item)
        llm_label = response_json.get("policy", "").strip()
        cpe_label = response_json.get("cpe_policy", "").strip()
        mle_label = response_json.get("mle_policy", "").strip()

        match = (llm_label == ground_truth_letter)
        match_cpe = (cpe_label == ground_truth_letter)
        match_mle = (mle_label == ground_truth_letter)

        print(f"[COMPARE] AGG: {llm_label} | CPE: {cpe_label} | MLE: {mle_label} | GT: {ground_truth_letter} ({ground_truth_text})")

        return {
            "persona": persona,
            "question": question,
            "ground_truth": {
                "label": ground_truth_text,
                "option": ground_truth_letter
            },
            "llm_prediction": {
                "aggregated": {
                    "option": llm_label,
                    "label": option_map.get(llm_label, "")
                },
                "cpe_only": {
                    "option": cpe_label,
                    "label": option_map.get(cpe_label, "")
                },
                "mle_only": {
                    "option": mle_label,
                    "label": option_map.get(mle_label, "")
                },
            },
            "match": {
                "aggregated": match,
                "cpe_only": match_cpe,
                "mle_only": match_mle
            },
            "llm_output": response_json
        }
    elif prompt_type == "rpa_new":
        response_json = rpa_new(item, client)
        llm_label = response_json.get("policy", "").strip()
        match = (llm_label == ground_truth_letter)

        print(f"[COMPARE] AGG: {llm_label} | GT: {ground_truth_letter} ({ground_truth_text})")

        return {
            "persona": persona,
            "question": question,
            "ground_truth": {
                "label": ground_truth_text,
                "option": ground_truth_letter
            },
            "llm_prediction": {
                "aggregated": {
                    "option": llm_label,
                    "label": option_map.get(llm_label, "")
                },
            },
            "match": {
                "aggregated": match
            },
            "llm_output": response_json
        }
    else:
        # 单轮 prompt 模式
        if prompt_type == "zs":
            prompt = Personalized_intent_mcq.format(persona=persona, question=question)
        elif prompt_type == "relevance":
            prompt = relevance_reasoner_prompt.format(persona=persona, question=question)
        elif prompt_type == "cot":
            prompt = cot_reasoner_prompt.format(persona=persona, question=question)
        elif prompt_type == "rsa":
            prompt = rational_speech_acts_prompt.format(persona=persona, question=question)
        elif prompt_type == "reminder":
            prompt = Personalized_intent_mcq_reminder.format(persona=persona, question=question)
        elif prompt_type == "rule":
            prompt = Personalized_rule_cot_mcq_prompt.format(persona=persona, question=question)
        elif prompt_type == "mle":
            prompt = MLE_estimation_template_v1.format(question=question, persona=persona)
        elif prompt_type == "mle_v2":
            prompt = MLE_estimation_template_v2.format(question=question, persona=persona)
        elif prompt_type == "cpe_recall":
            prompt = CPE_recall_estimation_prompt.format(question=question, persona=persona)
        elif prompt_type == "cpe_intent":
            prompt = CPE_intent_estimation_prompt.format(question=question, persona=persona)
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        
        try:
            if model == "qwen":
                format = load_json(f"./config/vllm_formats.json")["mcq"]
                response = client.get_single_chat_completion(prompt, response_format=format)
            else:
                response = client.get_single_chat_completion(prompt)
            response_json = safe_parse_json(response)
            llm_label = response_json.get("policy", "").strip()
        except Exception as e:
            print(f"[COMPARE] LLM 调用或解析错误: {e}")
            llm_label = ""
            response_json = {"policy": "", "reason": "", "raw": locals().get("response", "")}

        match = (llm_label == ground_truth_letter)

        print(f"[COMPARE] LLM: {llm_label} ({option_map.get(llm_label, '无效')}) | GT: {ground_truth_letter} ({ground_truth_text})")

        return {
            "persona": persona,
            "question": question,
            "ground_truth": {
                "label": ground_truth_text,
                "option": ground_truth_letter
            },
            "llm_prediction": {
                "option": llm_label,
                "label": option_map.get(llm_label, "")
            },
            "match": match,
            "llm_output": response_json
        }


def run_evaluation(input_json_path, output_dir, client, model_name="default-model", prompt_type="reminder", persona_type="implicit_persona", max_workers=10, limit=None):
    import os
    os.makedirs(output_dir, exist_ok=True)

    data = load_json(input_json_path)
    if limit is not None:
        data = data[:limit]

    results = []
    total = 0
    match = 0

    print(f"🔁 Running classification evaluation with {max_workers} threads... (limit={limit})")

    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_item, item, prompt_type, persona_type, model_name, client) 
                for item in data]

        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for fut in as_completed(futures):
                try:
                    # ✅ 单任务拿结果再给一个小的超时保护
                    result_entry = fut.result(timeout=5)
                except TimeoutError:
                    # ✅ 取消这个卡住的任务，继续其它
                    fut.cancel()
                    print("[Warn] a task timed out and was cancelled.")
                    pbar.update(1)
                    continue
                except Exception as e:
                    print(f"[Error] task failed: {e}")
                    pbar.update(1)
                    continue

                results.append(result_entry)
                total += 1
                if prompt_type in ("rpa", "rpa_new"):
                    match += int(result_entry["match"]["aggregated"])
                else:
                    match += int(bool(result_entry["match"]))
                pbar.update(1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}/intent_eval_results_{model_name}_{ts}_{prompt_type}_{persona_type}.json"
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    acc = match / total if total > 0 else 0.0
    print("\n=== Intent Classification Summary ===")
    print(f"Model: {model_name}")
    print(f"Total samples: {total}")
    print(f"accuracy: {match}/{total} = {acc:.2%}")

    summary = {
        "timestamp": ts,
        "model": model_name,
        "prompt": prompt_type,
        "persona_type": persona_type,
        "total": total,
        "acc": acc,
        "output_file": output_path
    }

    summary_path = f"{output_dir}/intent_summary_log.jsonl"
    with open(summary_path, "a", encoding="utf-8") as logf:
        logf.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"📄 Summary appended to: {summary_path}")


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
        prompt_type=args.prompt,
        persona_type=persona_type,
        max_workers=args.threads,
        limit=args.limit
    )
