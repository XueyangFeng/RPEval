import re
import json

def parse_advice_blocks(markdown_text, query_text):
    blocks = re.split(r"\n(?=\*\*)", markdown_text.strip())  # 按建议类型分块

    data = []

    for block in blocks:
        # 匹配标题（建议类型）
        advice_type_match = re.search(r"\*\*(.+?)\*\*", block)
        reason_match = re.search(r"- 原因：(.*)", block)
        preference_matches = re.findall(r"- 偏好\d+[:：](.*)", block)

        if not advice_type_match or not reason_match:
            continue  # 跳过格式不完整的块

        advice_type = advice_type_match.group(1).strip()
        reason = reason_match.group(1).strip()

        for pref in preference_matches:
            if pref.strip():
                data.append({
                    "query": query_text.strip(),
                    "advice_type": advice_type,
                    "reason": reason,
                    "preference": pref.strip()
                })

    return data


def safe_parse_json(text):
    """
    容错 JSON 解析：
    1. 支持 markdown 代码块中的 JSON（```json ... ```）
    2. 支持 <think>...</think> 后跟 JSON
    3. 自动修复常见 JSON 错误：
       - 未转义的换行符
       - 未转义的双引号
       - 中文/弯引号替换为半角
       - 补全未加引号的键
    返回解析后的 Python 对象（字典或列表）。
    """
    # Step 1: 尝试从 markdown 代码块提取
    match = re.search(r"```(?:json)?\s*([$begin:math:display${].*?[$end:math:display$}])\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Step 2: 去除 <think>...</think> 内容
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        json_str = cleaned_text.strip()

    # Step 3: 尝试标准解析
    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Step 4: 容错修复
    fixed = json_str

    # 4.1 替换常见中文引号/符号为半角
    trans = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '—': '-', '…': '...',
        '，': ',', '：': ':'
    }
    for k, v in trans.items():
        fixed = fixed.replace(k, v)

    # 4.2 去掉 BOM 和多余空白
    fixed = fixed.replace('\r\n', '\n').replace('\r', '\n').strip()

    # 4.3 给未加引号的键补引号（行首或逗号后）
    fixed = re.sub(
        r'(?m)(^|\{|\s|,)([A-Za-z0-9_\u4e00-\u9fff]+)\s*:',
        lambda m: f'{m.group(1)}"{m.group(2)}":',
        fixed
    )

    # 4.4 转义多行字符串（自动把内部换行 -> \n，引号转义）
    def _escape_multiline_strings(s):
        pattern = re.compile(
            r'("(?P<key>[^"]+)"\s*:\s*)"(?P<val>(?:.|\n)*?)"(?=(\s*,\s*"[^"]+"\s*:)|\s*}\s*$)',
            re.MULTILINE | re.DOTALL
        )
        def repl(m):
            head = m.group(1)
            val = m.group('val')
            val = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            return f'{head}"{val}"'
        return pattern.sub(repl, s)

    fixed = _escape_multiline_strings(fixed)

    # Step 5: 再尝试解析
    return json.loads(fixed)


# def safe_parse_json(text):
#     """
#     统一处理两种格式的 JSON 解析：
#     1. ```json ... ```
#     2. <think>...</think> 后面是 JSON
    
#     返回解析后的 Python 对象（字典或列表）。
#     """
#     # 先尝试匹配 markdown 代码块中的 JSON
#     match = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#     else:
#         # 去除 <think>...</think> 内容
#         cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#         json_str = cleaned_text.strip()
    
#     return json.loads(json_str)

def extract_and_parse_json(text):
    """
    解析 LLM 输出中用 ```json ... ``` 包裹的 JSON 内容。
    如果找不到 ```，尝试直接解析。
    支持多个 JSON 块。
    """
    # 找出 ```json 包裹的部分
    #blocks = re.findall(r"```json\\s*(\{.*?\})\\s*```", text, flags=re.DOTALL)
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)

    if not blocks:
        # 尝试直接解析整个文本（如果没有 markdown 包裹）
        try:
            return [json.loads(text)]
        except:
            raise ValueError("未检测到 JSON 块或无法解析")

    results = []
    for block in blocks:
        try:
            results.append(json.loads(block))
        except json.JSONDecodeError as e:
            print(f"解析失败: {e}")
    return results

# 从文件中加载 JSON 数据
def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 从配置文件中加载参数
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config