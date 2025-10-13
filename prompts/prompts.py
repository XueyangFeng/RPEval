from langchain.prompts import PromptTemplate

Personalized_responser_template = """
You are a personalized assistant, you need to refer to the user‘s persona to answer questions. 
User's Persona: {persona}
Question: {question}
"""

Personalized_responser_reminder_template = """
You are a personalized assistant, you need to refer to the user portrait appropriately to answer questions. 请注意有些偏好不恰当的可以忽略！
User's Persona: {persona}
Question: {question}
"""

Personalized_responser_cot_policy_template = """
你是一个个性化助手，需要适当地参考用户画像，请先从下三种回答策略中确定最合适的一个，然后基于策略生成回答。:

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

示例：
"persona": “用户对高难度户外运动有一定兴趣，偶尔会关注新的挑战项目，觉得尝试一下挺有意思，但并不会把这类活动当作生活的重心。”,
"question": “我们明年春假想去日本家庭旅行，有推荐的旅行路线吗？
思考： 用户的偏好是对高难度户外运动有兴趣，但这次旅行是家庭旅行，可能更注重亲子活动和文化体验，而不是极限运动。
输出： A

"persona": “用户对历史文化古迹很感兴趣”,
"question": “我们明年春假想去日本家庭旅行，有推荐的旅行路线吗？
思考： 用户的偏好是对历史文化古迹感兴趣，这次旅行可以适当围绕这个偏好来设计行程，推荐一些有历史文化价值的景点。
输出： B

"persona": “用户的信仰是素食主义”,
"question": “待会儿该吃啥？
思考： 用户的偏好是素食主义，这次的选择应该围绕素食来进行，不能推荐肉类或非素食选项。
输出： C

- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）。可以直接被parse的json。不要直接包含裸换行，必须写成 \n 才能被解析器识别。
输出格式要求：
{{
  "persona": "{persona}",
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(A/B/C)",
  "response": "(你的回答)"
}}
"""



Personalized_responser_cot_policy_mgen_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要你分辨是否该在这个场景是否用偏好，如何用偏好, 然后基于策略生成回答。. 每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 

示例：
"personas": [
      "用户平时对新出的快餐品类很感兴趣，偶尔会专门去尝试一下，觉得新鲜有趣，但并不会特别执着于某一种口味。",
      "用户习惯在下午的时候给自己准备些小甜点，觉得这样能让一天的心情变得更好，但也不会特别执着于每天都吃。",
      "用户平时喜欢吃些甜点，偶尔会买蛋糕或巧克力犒劳自己，觉得甜食能带来一点小确幸。",
      "用户平时吃饭喜欢味道浓郁一些，觉得重口味更有满足感，但也能接受偶尔换换清淡口味。",
      "用户平时吃饭时喜欢搭配一些腌制食品，觉得这样更有味道，也比较习惯这种咸香的口感。",
      "用户平时工作比较忙，经常用外卖解决一日三餐，主要是图省事，对餐食种类没有特别执着。"
    ],
"question": "爸妈年纪大了，想帮他们调整下饮食结构，有没有适合中老年人的健康饮食建议?"
"reason": [
"偏好 1：关于快餐探索兴趣，与为爸妈寻找健康饮食建议无关 → A",
"偏好 2：下午甜点作为调节心情的小习惯，与爸妈饮食结构不相关 → A",
"偏好 3：甜食是用户个人小确幸，与目标人群（爸妈）无关 → A",
"偏好 4：偏好重口味，虽涉及饮食，但不适用于当前目标对象 → A",
"偏好 5：腌制食品的喜好是用户习惯，不能用于爸妈健康参考 → A",
"偏好 6：外卖频率与便利性无关当前任务 → A"
],
policy: "AAAAAA"

"personas": [
      "用户平时喜欢在故事中寻找温暖和治愈的感觉，对那些能带来内心安慰的内容总是格外关注，但偶尔也会对不同风格的作品抱有好奇心。",
      "用户平时喜欢关注社会新闻和现象，偶尔会花时间阅读深度报道，对社会变化背后的原因和影响感到好奇。",
      "用户平时喜欢一个人安静地看些有深度的知识类内容，觉得这样能让自己放松下来，也能学到新东西。",
      "用户和女朋友对爱情片有着近乎本能的抗拒，无论剧情多精彩，只要沾上爱情元素就会让他们瞬间失去兴趣。"
    ],
"question": "我和女朋友周末想去国贸那边看电影，能帮我推荐几部最近口碑不错的片子吗？"
"reasoning": [
    "偏好 1：偏好温暖治愈内容，虽然未在 question 中明确要求，但可以作为推荐方向，适度支持性使用 → B",
    "偏好 2：关注社会新闻和报道，但该偏好与电影推荐无直接关联 → A",
    "偏好 3：喜欢有深度的知识类内容，但 question 中无指向性信息，当前不适用 → A",
    "偏好 4：强烈排斥爱情片，且是两人共同的明显否定偏好，若不考虑将可能推荐错误内容 → C"
  ],
policy: "BAAC"


- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
输出格式要求：
{{
  "persona": "{personas}",
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(由A/B/C组成的字符串, 输出长度必须和persona的个数一致)",
  "response": "(你的回答)"
}}
"""




Personalized_responser_cot_template = """
You are a thoughtful and professional assistant who understands the user's habits and preferences.

User preference: {persona}

The user has asked the following question:
"{question}"

Please combine the user's preferences with the situational context to reason step by step (chain-of-thought), explain your recommendation logic or decision rationale, and provide a clear and tailored final answer.

Please structure your response as follows:
1. Clarify the user's intent
2. Analyze the user's personalized preferences
3. Reason and filter based on the scenario
4. Provide a personalized suggestion or answer
"""


Personalized_cot_mmcq_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要你分辨是否该在这个场景是否用偏好，如何用偏好. 每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 

示例：
"persona": [
      "用户平时对新出的快餐品类很感兴趣，偶尔会专门去尝试一下，觉得新鲜有趣，但并不会特别执着于某一种口味。",
      "用户习惯在下午的时候给自己准备些小甜点，觉得这样能让一天的心情变得更好，但也不会特别执着于每天都吃。",
      "用户平时喜欢吃些甜点，偶尔会买蛋糕或巧克力犒劳自己，觉得甜食能带来一点小确幸。",
      "用户平时吃饭喜欢味道浓郁一些，觉得重口味更有满足感，但也能接受偶尔换换清淡口味。",
      "用户平时吃饭时喜欢搭配一些腌制食品，觉得这样更有味道，也比较习惯这种咸香的口感。",
      "用户平时工作比较忙，经常用外卖解决一日三餐，主要是图省事，对餐食种类没有特别执着。"
    ],
"question": "爸妈年纪大了，想帮他们调整下饮食结构，有没有适合中老年人的健康饮食建议?"
"reason": [
"偏好 1：关于快餐探索兴趣，与为爸妈寻找健康饮食建议无关 → A",
"偏好 2：下午甜点作为调节心情的小习惯，与爸妈饮食结构不相关 → A",
"偏好 3：甜食是用户个人小确幸，与目标人群（爸妈）无关 → A",
"偏好 4：偏好重口味，虽涉及饮食，但不适用于当前目标对象 → A",
"偏好 5：腌制食品的喜好是用户习惯，不能用于爸妈健康参考 → A",
"偏好 6：外卖频率与便利性无关当前任务 → A"
],
policy: "AAAAAA"

"persona": [
      "用户平时喜欢在故事中寻找温暖和治愈的感觉，对那些能带来内心安慰的内容总是格外关注，但偶尔也会对不同风格的作品抱有好奇心。",
      "用户平时喜欢关注社会新闻和现象，偶尔会花时间阅读深度报道，对社会变化背后的原因和影响感到好奇。",
      "用户平时喜欢一个人安静地看些有深度的知识类内容，觉得这样能让自己放松下来，也能学到新东西。",
      "用户和女朋友对爱情片有着近乎本能的抗拒，无论剧情多精彩，只要沾上爱情元素就会让他们瞬间失去兴趣。"
    ],
"question": "我和女朋友周末想去国贸那边看电影，能帮我推荐几部最近口碑不错的片子吗？"
"reasoning": [
    "偏好 1：偏好温暖治愈内容，虽然未在 question 中明确要求，但可以作为推荐方向，适度支持性使用 → B",
    "偏好 2：关注社会新闻和报道，但该偏好与电影推荐无直接关联 → A",
    "偏好 3：喜欢有深度的知识类内容，但 question 中无指向性信息，当前不适用 → A",
    "偏好 4：强烈排斥爱情片，且是两人共同的明显否定偏好，若不考虑将可能推荐错误内容 → C"
  ],
policy: "BAAC"


- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
输出格式要求：
{{
  "persona": "{personas}",
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(由A/B/C组成的字符串, 输出长度必须和persona的个数一致)"
}}
"""



Personalized_cot_mmcq_implicit_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要你分辨是否该在这个场景是否用偏好，如何用偏好. 每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 

示例：
"persona": [
      "用户平时对新出的快餐品类很感兴趣，偶尔会专门去尝试一下，觉得新鲜有趣，但并不会特别执着于某一种口味。",
      "用户习惯在下午的时候给自己准备些小甜点，觉得这样能让一天的心情变得更好，但也不会特别执着于每天都吃。"
    ],
"question": "爸妈年纪大了，想帮他们调整下饮食结构，有没有适合中老年人的健康饮食建议?"
"reason": [
"偏好 1：关于快餐探索兴趣，与为爸妈寻找健康饮食建议无关",
"偏好 2：下午甜点作为调节心情的小习惯，与爸妈饮食结构不相关"
],
policy: "AA"

"persona": [
      "用户平时喜欢在故事中寻找温暖和治愈的感觉，对那些能带来内心安慰的内容总是格外关注，但偶尔也会对不同风格的作品抱有好奇心。",
      "用户平时喜欢关注社会新闻和现象，偶尔会花时间阅读深度报道，对社会变化背后的原因和影响感到好奇。",
      "用户平时喜欢一个人安静地看些有深度的知识类内容，觉得这样能让自己放松下来，也能学到新东西。",
      "用户和女朋友对爱情片有着近乎本能的抗拒，无论剧情多精彩，只要沾上爱情元素就会让他们瞬间失去兴趣。"
    ],
"question": "我和女朋友周末想去国贸那边看电影，能帮我推荐几部最近口碑不错的片子吗？"
"reasoning": [
    "偏好 1：偏好温暖治愈内容，虽然未在 question 中明确要求，但可以作为推荐方向，适度支持性使用",
    "偏好 2：关注社会新闻和报道，但该偏好与电影推荐无直接关联 ",
    "偏好 3：喜欢有深度的知识类内容，但 question 中无指向性信息，当前不适用",
    "偏好 4：强烈排斥爱情片，且是两人共同的明显否定偏好，若不考虑将可能推荐错误内容"
  ],
policy: "BAAC"

偏好: {personas}

问题: {question}

- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
输出格式要求：
{{
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(由A/B/C组成的字符串, 输出长度必须和persona的个数一致)"
}}
"""




Personalized_cot_mcq_template = """
你是一个个性化助手，需要适当地参考用户画像，从以下三种回答策略中确定最合适的一个:

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

示例：
“persona”: “用户对高难度户外运动有一定兴趣，偶尔会关注新的挑战项目，觉得尝试一下挺有意思，但并不会把这类活动当作生活的重心。”,
“question”: “我们明年春假想去日本家庭旅行，有推荐的旅行路线吗？
思考： 用户的偏好是对高难度户外运动有兴趣，但这次旅行是家庭旅行，可能更注重亲子活动和文化体验，而不是极限运动。
输出： A

“persona”: “用户对历史文化古迹很感兴趣”,
“question”: “我们明年春假想去日本家庭旅行，有推荐的旅行路线吗？
思考： 用户的偏好是对历史文化古迹感兴趣，这次旅行可以适当围绕这个偏好来设计行程，推荐一些有历史文化价值的景点。
输出： B

“persona”: “用户的信仰是素食主义”,
“question”: “待会儿该吃啥？
思考： 用户的偏好是素食主义，这次的选择应该围绕素食来进行，不能推荐肉类或非素食选项。
输出： C

输出格式要求：
{{
  "persona": "{persona}",
  "question": "{question}"
  "reason": "(你的思考过程)",
  "policy": "(A/B/C)"
}}

- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
"""


Personalized_intent_mcq_template = """
你是一个个性化助手，需要适当地参考用户画像，从以下三种回答策略中确定最合适的一个:

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

输出格式要求：
{{
  "persona": "{persona}",
  "question": "{question}"
  "reason": "(你的思考过程)",
  "policy": "(A/B/C)",
}}

- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
"""


Personalized_intent_mmcq_implicit_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要判断每条偏好在当前任务中应采取的调用策略。每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。


偏好: {personas}

问题: {question}
请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 
输出格式要求：
{{
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(五个偏好就类似于AABCC)"
}}

注意事项：
- 字符串中的每个字母依次对应 persona 列表中的每一条偏好。
- 只输出 JSON，不要添加解释、注释或代码块标记（如 ```json）。
"""




Personalized_intent_mmcq_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要判断每条偏好在当前任务中应采取的调用策略。每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 
输出格式要求：
{{
  "personas": "{personas}",
  "question": "{question}"
  "reason": "(你的思考过程)",
  "policy": "(五个偏好就类似于AABCC)"
}}

注意事项：
- 字符串中的每个字母依次对应 persona 列表中的每一条偏好。
- 只输出 JSON，不要添加解释、注释或代码块标记（如 ```json）。
"""



Personalized_intent_mmcq_reminder_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要你分辨是否该在这个场景是否用偏好，如何用偏好. 每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 
输出格式要求：
{{
  "personas": "{personas}",
  "question": "{question}"
  "reason": "(你的思考过程)",
  "policy": "(五个偏好就类似于AABCC)"
}}

注意事项：
- 字符串中的每个字母依次对应 persona 列表中的每一条偏好。
- 只输出 JSON，不要添加解释、注释或代码块标记（如 ```json）。
"""




Personalized_intent_mmcq_reminder_implicit_template = """
你是一个个性化助手，需要适当地参考用户画像，确定回答策略，需要你分辨是否该在这个场景是否用偏好，如何用偏好. 每条偏好只能被归入以下三类之一：

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性偏好：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

偏好: {personas}

问题: {question}

请逐条分析偏好在当前问题中的作用，并输出每条的策略对应字母，形成一个长度与偏好个数完全相等的字符串。字符串中每个字符只能是A、B或C。 
输出格式要求：
{{
  "question": "{question}",
  "reason": "(你的思考过程)",
  "policy": "(五个偏好就类似于AABCC)"
}}

注意事项：
- 字符串中的每个字母依次对应 persona 列表中的每一条偏好。
- 只输出 JSON，不要添加解释、注释或代码块标记（如 ```json）。
"""


Personalized_intent_mcq_reminder_template = """
你是一个个性化助手，需要你分辨是否该在这个场景是否用偏好，如何用偏好。从以下三种回答策略中确定最合适的一个:

- 忽略偏好：完全按当前任务目标行事，不顾及历史偏好(A)。
- 支持性：试图在满足当前任务的同时，融入或部分保留历史偏好(B)。
- 以偏好为主线：当前行为强烈受偏好驱动，目标围绕偏好展开(C)。

输出格式要求：
{{
  "persona": "{persona}",
  "question": "{question}"
  "reason": "(你的思考过程)",
  "policy": "(A/B/C)",
}}

- 仅输出 JSON，勿添加任何解释文字或代码块标记（如```json）
"""




Personalized_responser_mcq_template = """
You are a personalized assistant, you need to refer to the user portrait appropriately to determine the most appropriate one from the following four answers and prioritize the four options (from most appropriate to least appropriate.

User's Persona: {persona}

Question: {question}

Options: {options}

请输出以下内容，使用标准 JSON 格式回复：

```json
{{
  "best_style": "A/B/C/D",（这里必须选一个最好的）
  "ranking": ["A", "B", "C", "D"],
  "reason": "<简要说明排序理由（100字以内）>"
}}
"""



Personalized_Responser_mcq_template = """
You are a personalized assistant, you need to refer to the user portrait appropriately to determine the most appropriate one from the following four answers and prioritize the four options (from most appropriate to least appropriate.

User's Persona: {persona}

Question: {question}

Options: {options}

请输出以下内容，使用标准 JSON 格式回复：

```json
{{
  "best_style": "A/B/C/D",（这里必须选一个最好的）
  "ranking": ["A", "B", "C", "D"],
  "reason": "<简要说明排序理由（100字以内）>"
}}
"""


gen_advice_template_604 = """
你是一位语言理解能力极强的助手，正在构建一个用于评估 AI 个性化理解能力的数据集。请基于常识，完成以下任务：

---

🎯 任务目标：

1. 针对给定的生活场景，分别列出该场景中，生成：不符合常识的建议类型，可行性差，**（使用抽象类别，如“限制级剧情片”、“危险地形”、“自己动手做饭”等），每条建议配上一句简洁理由，说明为什么它在该场景下不合适。

2. 为每条建议，生成**1–2 条现实、自然但在该场景中发生偏差的个人偏好**作为潜在来源。  
   这些偏好应在其他场景中是合理的，但在此处发生了**偏好错位**

---

✅ 要求：

- 建议类型应为抽象通用类别，不是具体实例
- 偏好必须真实自然，可来源于兴趣、习惯、行为风格、生活节奏等，偏好最好不要针对当前场景。
- 如果一个偏好在该场景中其实有一定适配性，请**明确指出冲突限定条件**（如“情绪过于压抑时才不适合”）
- 输出使用 Markdown 格式，结构清晰，便于结构化抽取
- 语言保持中立、自然、无讽刺色彩
- 有的场景是复合的，如去xxx旅游，要吃什么，一方面可以考虑xxx的饮食习惯，另一方面要考虑旅游时候不方便干什么，如去旅游，自己做就不合适。
- 对于每条偏好，不需要生成很具体的描述，只需要给出用户喜欢xxx就可以了。后面我们会进行额外的步骤把它的程度与范围进行具体化。

---

📘 示例：

场景：我和家里人打算周五晚上去看电影，有什么推荐吗？

**恐怖片 / 惊悚片**

- 类型：不合理的建议
- 原因：包括灵异、血腥、心理恐怖等，可能让孩子或老人感到不适。
- 偏好1: 用户喜欢看恐怖游戏直播，来享受肾上腺素飙升的感觉。
- 偏好2: 用户喜欢在聚会的时候，和朋友讲灵异小说xxxx

**限制级剧情片（R级）**  

- 类型：不合理的建议
- 原因：包含暴力、性、毒品等内容，不适合孩子或老人共同观看。  
- 偏好1：用户喜欢探讨社会极端议题，偏好暴力/犯罪类作品的张力。  
- 偏好2：用户喜欢成人向游戏和剧集，习惯高刺激性剧情。

**奇幻剧情片**  

- 类型：比较特别但是合理的建议
- 原因：虽然有些奇幻元素，但通常适合家庭观影，能引发想象力和讨论。
- 偏好1：用户喜欢奇幻小说和电影，认为能激发创造力。
- 偏好2：用户喜欢和家人一起看奇幻冒险类的动画片。

<例子结束>

---

请以相同结构完成以下输入：

输入场景：
{question}

输出格式：
[
  {{
    "Preference": "偏好",
    "question": "{question}",
    "Structure": {{
      "Who": "（偏好适用的参与角色）",
      "When": "（偏好适用的时间场景）",
      "Where": "（偏好适用的地点场景）",
      "What": "（偏好领域）",
      "Why": "（偏好动机）"
    }}
    "reason": "（错位分析）",
    "advice_type": "(不合理的建议内容)"
  }},
  ...
]
请直接输出可以被json.load解析的内容，不要添加任何解释文字、不要使用 markdown 代码块（如```json），只返回标准 JSON。
"""


personalized_cot_policy_responser = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_responser_cot_policy_template
)

personalized_responser = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_responser_template,
)

personalized_responser_reminder = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_responser_reminder_template,
)

personalized_responser_cot = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_responser_cot_template,
)

Personalized_responser_mcq = PromptTemplate(
    input_variables=["persona", "question", "options"],
    template=Personalized_Responser_mcq_template,
)

Personalized_intent_mcq = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_intent_mcq_template
)

Personalized_intent_mmcq = PromptTemplate(
    input_variables=["personas", "question"], 
    template=Personalized_intent_mmcq_template
)

Personalized_intent_mmcq_implicit = PromptTemplate(
    input_variables=["personas", "question"], 
    template=Personalized_intent_mmcq_implicit_template
)

Personalized_intent_mmcq_reminder = PromptTemplate(
    input_variables=["personas", "question"],
    template=Personalized_intent_mmcq_reminder_template
)

Personalized_intent_mmcq_reminder_implicit = PromptTemplate(
    input_variables=["personas", "question"],
    template=Personalized_intent_mmcq_reminder_implicit_template
)

Personalized_intent_mcq_reminder = PromptTemplate(
    input_variables=["persona", "question"],
    template=Personalized_intent_mcq_reminder_template
)


cot_reasoner_prompt = PromptTemplate(
    input_variables=["question", "persona"],
    template=Personalized_cot_mcq_template
)


Personalized_cot_mmcq_prompt = PromptTemplate(
    input_variables=["question", "personas"],
    template=Personalized_cot_mmcq_template
)

Personalized_cot_mmcq_implicit_prompt = PromptTemplate(
    input_variables=["question", "personas"],
    template=Personalized_cot_mmcq_implicit_template
)

Personalized_cot_mgen_responser = PromptTemplate(
    input=["question", "personas"],
    template=Personalized_responser_cot_policy_mgen_template
)