import json
from datetime import datetime
from model.model import OpenAIClient
from prompts_715 import (
    Seed_generation_template,
    Persona_init_harmonize_template,
    Persona_init_ignored_template,
    Persona_init_dominant_template,
    Persona_update_harmonize_template,
    Persona_update_ignored_template,
    Persona_update_dominant_template,
    Actr_checker_ignore_template,
    Actr_checker_supportive_template,
    Actr_checker_dominant_template,
    MMCQ_check_template
)
from prompts import gen_question_prompt_429
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from constants import ALL_DOMAINS
from collections import defaultdict
from utils import (
    load_config, 
    load_json, 
    parse_advice_blocks,
    safe_parse_json
)


def init_openai_client(model_name="openai_41", config_path="./api_config.json"):
    """加载配置并初始化 OpenAI 客户端"""
    config = load_config(config_path)[model_name]
    return OpenAIClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_path=config["model_path"]
    )

def generate_seed(output_path="data7/question_seed.json"):
    """从常量文件生成查询种子"""
    client = init_openai_client()
    res = []
        # 提前生成每个 prompt
    prompts = [(topic, Seed_generation_template.format(topic=topic)) for topic in ALL_DOMAINS]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(client.get_single_chat_completion, prompt): topic for topic, prompt in prompts}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating seeds"):
            topic = futures[future]
            try:
                response = future.result()
                parsed = safe_parse_json(response)
                for item in parsed:
                    item["topic"] = topic
                res.extend(parsed)
            except json.JSONDecodeError:
                raise ValueError(f"模型返回的响应不是合法的 JSON（Topic: {topic}）：\n{response}")
            except Exception as e:
                raise ValueError(f"生成 Topic {topic} 时出错：{e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Seed 生成成功：{output_path}")
    return output_path

def generate_question(seed_path, seed_index=5, output_path="data7/question_v0.json"):
    """
    从种子数据生成查询并保存（并行版）
    """
    seed_data = load_json(seed_path)

    if seed_index > 0:
        seed_data = seed_data[:seed_index]

    res = []
    client = init_openai_client()

    # 提前生成所有prompt
    prompts = []
    for seed in seed_data:
        prompt = gen_question_prompt_429.format(
            What=seed["What"],
            Why=seed["Why"]
        )
        prompts.append((seed, prompt))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(client.get_single_chat_completion, prompt): seed for seed, prompt in prompts}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
            seed = futures[future]
            try:
                response = future.result()
                parsed = safe_parse_json(response)

                # 给每个结果加上来源Seed信息（可选）
                for item in parsed:
                    item["Seed_What"] = seed["What"]
                    item["Seed_Why"] = seed["Why"]

                res.extend(parsed)

            except json.JSONDecodeError:
                raise ValueError(f"模型返回的响应不是合法的 JSON（Seed: {seed}）：\n{response}")
            except Exception as e:
                raise ValueError(f"生成 Seed {seed} 时出错：{e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"✅ question 生成成功：{output_path}")
    return output_path

def init_persona(question_path, output_dir="data7/", intent_type="忽略偏好", limit=None):
    """
    根据查询生成Persona建议并保
    存（并行版）
    """
    queries = load_json(question_path)

    if not queries:
        raise ValueError("❌ question 文件为空，无法生成建议。")

    os.makedirs(output_dir, exist_ok=True)

    if limit is not None:
        queries = queries[:limit]

    client = init_openai_client()

    all_results = []

    if intent_type == "忽略偏好":
        init_template = Persona_init_ignored_template
    elif intent_type == "支持性偏好":
        init_template = Persona_init_harmonize_template
    elif intent_type == "以偏好为主线":
        init_template = Persona_init_dominant_template


    # 提前生成所有prompt
    prompts = []
    for question in queries:
        prompt = init_template.format(
            question=question["Query"]
        )
        prompts.append((question, prompt))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(client.get_single_chat_completion, prompt): question
            for question, prompt in prompts
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating persona"):
            question = futures[future]
            try:
                response = future.result()
                print(f"🧠 响应: {response}")

                parsed = safe_parse_json(response)

                # 可选：在结果里加溯源question
                for item in parsed:
                    item["question"] = question["Query"]

                all_results.extend(parsed)

            except json.JSONDecodeError:
                raise ValueError(f"模型返回的响应不是合法的JSON（question: {question['question']}）：\n{response}")
            except Exception as e:
                print(f"⚠️ 生成 question 时出错：{e}")
                continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if intent_type == "忽略偏好":
        output_path = os.path.join(output_dir, f"ignore_persona_iter0_v0.json")
    elif intent_type == "支持性偏好":
        output_path = os.path.join(output_dir, f"supportive_persona_iter0_v0.json")
    elif intent_type == "以偏好为主线":
        output_path = os.path.join(output_dir, f"dominant_persona_iter0_v0.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Persona 生成成功：{output_path}")
    return output_path


def update_persona(input_path="data7/ignore_persona_iter0_v0.json", output_path="data729/ignore_persona_iter1_v0.json", intent_type="忽略偏好", limit=None):
    """批量生成意图，保存为一个 JSON 文件"""

    data_list = load_json(input_path)
    if limit is not None:
        data_list = data_list[:limit]

    client = init_openai_client()
    all_results = []

    if intent_type == "忽略偏好":
        update_template = Persona_update_ignored_template
    elif intent_type == "支持性偏好":
        update_template = Persona_update_harmonize_template
    elif intent_type == "以偏好为主线":
        update_template = Persona_update_dominant_template


    # 提前生成所有prompt
    prompts = []
    for data in data_list:
        persona = data["persona"]
        question = data["question"]
        prompt = update_template.format(
            persona_old=persona,
            question=question
        )
        prompts.append((question, prompt))


    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(client.get_single_chat_completion, prompt): question
            for question, prompt in prompts
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Updating persona"):
            question = futures[future]
            try:
                response = future.result()
                print(f"🧠 响应: {response}")

                parsed = safe_parse_json(response)
                #parsed["question"] = question # 添加问题信息
                all_results.append(parsed)

            except json.JSONDecodeError:
                raise ValueError(f"模型返回的响应不是合法的JSON（question: {question['question']}）：\n{response}")
            except Exception as e:
                print(f"⚠️ 更新 persona 时出错：{e}")
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ 全部生成成功：共 {len(all_results)} 条，保存到 {output_path}")
    return True




def actr_checker(data_path="data6/merged.json", output_path="data6/rsa_results.json", intent="ignore", model="openai_41", limit=None):
    """检查人类标注与 gold 标签的一致性"""
    data = load_json(data_path)[:limit] if limit is not None else load_json(data_path)

    client = init_openai_client(model_name=model)
    results = []
    def process_item(item):
        persona = item.get("persona") or item["preference"]
        question = item.get("question") or item["query"]

        if data_path.startswith("./prefeval"):
            intent_type = "以偏好为主线"
            gold_intent = "以偏好为主线"
        else:
            if intent == "ignore":
                intent_type = "忽略偏好"
                gold_intent =  "忽略偏好"
            elif intent == "supportive":
                intent_type = "支持性偏好"
                gold_intent = "支持性偏好"
            elif intent == "dominant":
                intent_type = "以偏好为主线"
                gold_intent =  "以偏好为主线"
            
            
        if intent == "ignore":
            # 忽略意图类型，直接使用默认的检查模板
            actr_checker_template = Actr_checker_ignore_template
        elif intent == "supportive":
            # 使用支持性意图的检查模板
            actr_checker_template = Actr_checker_supportive_template
        elif intent == "dominant":
            actr_checker_template = Actr_checker_dominant_template
        else:
            raise ValueError(f"未知意图类型: {intent}. 请选择 'ignore', 'supportive' 或 'dominant'.")

        prompt = actr_checker_template.format(
            persona=persona,
            question=question,
            intent_type=intent_type,    
            intent=gold_intent,
        )
        #print(prompt)
        response = client.get_single_chat_completion(prompt)
        parsed = safe_parse_json(response)
        print(f"question: {question}")
        print(f"persona: {persona}")
        print(f"intent: {intent_type}")
        print(f"🧠 响应: \n{response}")
        return {**item, **parsed}

    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ACTR Checking"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"❌ 出错: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def MMCQ_check(data_path="data805/mmcq.json", output_path="data805/mmcq_check_results.json", model="openai_41", limit=None):
    """检查多重偏好中是否存在冲突"""
    data = load_json(data_path)[:limit] if limit is not None else load_json(data_path)

    client = init_openai_client(model_name=model)
    results = []
    def process_item(item):
        persona = item.get("persona")
        prompt = MMCQ_check_template.format(
            persona=persona
        )
        response = client.get_single_chat_completion(prompt)
        parsed = safe_parse_json(response)
        print(f"persona: {persona}")
        print(f"🧠 响应: \n{response}")
        return {**item, **parsed}

    results = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="MMCQ Checking"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"❌ 出错: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 全部生成成功：共 {len(results)} 条，保存到 {output_path}")
    return True

if __name__ == "__main__":

    seed_file = generate_seed(output_path="data7/question_seed.json")
    #第一步：生成查询
    question_file = generate_question(seed_path="./question_seed.json", seed_index=100, output_path="data7/question_v0.json")

    #第二步：根据查询生成偏好 (把intent_type改成"支持性偏好"或"以偏好为主线", 或"忽略偏好")
    init_persona(
        question_path="./data7/query_v0.json",
        output_dir="./data805/",
        intent_type="以偏好为主线",
        #model_name="openai_41",
        #limit=100
    )

    #第三步：优化一次偏好 (把intent_type改成"支持性偏好"或"以偏好为主线", 或"忽略偏好")
    update_persona(
        input_path="./data805/dominant_persona_iter0_v0.json",
        output_path="data805/dominant_persona_iter1_v0.json",
        intent_type="以偏好为主线",
        limit=2000
    )

    #第四步：为所有偏好的质量进行评分。高分的数据质量会相对较高。intent为"ignore", "supportive", "dominant"
    actr_checker(
        data_path="./data805/dominant_persona_iter1_v0.json",
        output_path="./data805/actr_dominant_score_iter1.json",
        intent="dominant",
        limit=None
    )


    # 多偏好的一致性检查, 单偏好不需要
    # MMCQ_check(
    #     data_path="./data805/mmcq.json",
    #     output_path="./data805/mmcq_check_results.json",
    #     model="openai_41",
    #     limit=None
    # )
