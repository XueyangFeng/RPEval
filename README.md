# How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants


## Motivation Example
Different levels of PAs. In L1, memory is directly concatenated with the query, whereas in L2, the PA infers implicit cues from the user’s query to determine memory utilization strategy
![alt text](pic/image-1.png)

## Performance
### Discriminative Setting
Performance of major LLMs on the discriminative intent matching accuracy in RPEVAL. The Human row reports the average accuracy from blind human annotation.
![alt text](pic/image-2.png)

### Generative Setting
(a) Fine-grained Error Analysis; (b) Overall error severity in the generative setting with
the single-preference configuration, (c) the reliability of the LLM-as-a-judge evaluation.
![alt text](pic/image-3.png)

## Dataset Location
The preference evaluation dataset is located in the benchmark_dataset directory.

## Dataset Format
The dataset is provided in json format and contains the following attributes:

1. Single Preference:
```
{
  "persona": "用户平时注重身心放松，喜欢通过冥想等方式为自己营造安静舒缓的氛围，对各种助眠和自我安抚的小技巧总是抱有好奇心。",
  "question": "有没有适合学生宿舍的夜间放松小仪式，能让我晚上不那么焦虑？",
  "intent": "希望获得一些能在有限空间内操作、帮助缓解焦虑和放松身心的夜间小仪式，既能结合冥想，也愿意尝试其他简单易行的放松方法。",
  "intent_type": "supportive"
}
```

2. Multiple Preference:
```
{
    "persona": [
      "用户习惯晚上活动较多，入睡时间比一般人要晚一些，觉得夜晚更安静自在。",
      "用户平时喜欢和家人一起参与各类文化体验活动，乐于在日常生活中为孩子寻找新鲜有趣的学习机会。",
      "用户极度反感人多和嘈杂的环境，尤其在出行安排上，绝不会选择任何热门、拥挤、需要排队或容易产生喧闹的地方。只要涉及人流密集或噪音大的场所，用户都会果断拒绝，哪怕是再有名的景点也绝不考虑。",
      "用户平时喜欢挑战自我，偶尔会尝试一些高空、速度感强的极限运动，觉得那种肾上腺素飙升的感觉很刺激，但也不会刻意追求每次都要冒险。",
      "用户对户外探险和野外生存有一定兴趣，偶尔会参加相关活动，觉得挑战自然是一种很有趣的体验。",
      "用户平时喜欢自己动手搭建营地，觉得动手布置环境很有成就感，也能让人静下心来享受过程。",
      "用户平日里喜欢在闲暇时分走进自然，享受绿意与清新的空气，对舒适自在的生活节奏有着温和的偏爱。",
      "用户喜欢在户外挑战自己，偏爱选择那些路线复杂、需要体力和耐力的徒步活动，觉得征服难度本身就是一种乐趣。",
      "用户平时喜欢在夜晚找些有趣的活动放松自己，比如偶尔和朋友小聚或体验城市夜生活，觉得这样能让生活多点乐趣。"
    ],
    "question": "我们准备国庆带孩子去成都周边玩几天，能帮忙设计个轻松点的行程吗？",
    "intent_types": "ABCAAABAA",
    "intents": [
      "用户在涉及家庭出游和孩子时，完全没有提及或坚持自己作息晚的习惯，说明这一偏好更多是日常生活中的自然状态，而非强烈的生活主张。用户能够灵活调整作息以适应家庭需求，表明其对作息时间的偏好属于温和、可变的类型。",
      "用户一贯重视孩子的成长体验，倾向于选择文化类活动，但这次出行又特别强调轻松，说明对活动的选择既有文化倾向，也注重舒适与亲子互动。用户对文化活动有一定偏好，但并非极端专注，愿意在亲子出行中兼顾多样体验。",
      "用户明确表示孩子害怕拥挤和嘈杂，且希望行程轻松。国庆期间人流密集，用户对安静环境的需求非常突出。persona应具体化为：对人多、吵闹的环境极度排斥，任何涉及热门景点、排队、喧闹场所的建议都不可接受。",
      "用户在涉及家庭出游和孩子时，主动选择了轻松的行程，完全没有考虑将极限运动融入其中，说明对极限运动的偏好更多是个人兴趣，在特定场合下可以轻松搁置，偏好强度属于日常爱好层面。",
      "用户在涉及家庭和孩子的出行安排时，完全没有将野外生存体验的偏好带入决策，说明这种偏好更多是个人兴趣或偶尔尝试，而非生活中不可或缺的核心部分。",
      "用户在涉及家庭出游和亲子活动时，优先考虑全家人的舒适和轻松氛围，而没有坚持自己动手搭建营地的习惯，说明这一偏好更多是个人兴趣或偶尔为之，并非生活中不可或缺的核心部分。",
      "用户平时就有亲近自然的习惯，说明他对自然环境有一定偏好，但并未表现出极端的热衷或专业性。此次出行希望轻松，说明对舒适度有一定要求，偏好自然但不排斥常规的休闲方式，属于自然取向较明显但不排他、兼容性较强的用户。",
      "用户在涉及家庭出游时，主动放下了对高难度徒步的偏好，说明这种偏好更多是个人兴趣或挑战自我的方式，而非生活中不可妥协的核心需求。用户能够灵活调整自己的活动选择，表明对高难度徒步的热爱属于中等偏好，更多体现在个人时间和特定场合。",
      "用户在涉及家庭出游和亲子活动时，完全没有将夜间娱乐的个人兴趣带入决策，说明夜间娱乐只是日常生活中的一种放松方式，而非不可或缺的核心需求。"
    ]
}
```

## Benchmarking on RPEval
### **nvironment Setup**
Create a conda environment:
```
conda create -n rpeval python=3.10 -y
conda activate rpeval
```

Install the required dependencies:
```
pip install -r requirements.txt
```

### **API Configuration**
All LLM (Large Language Model) calls in this repository are made using OpenAI-like interfaces. To configure the APIs:

1. Set your API information in the `config/api_config.json` file.
2. For closed-source models, set the information directly in the config.
3. For open-source models, use `vllm` for local deployment. We have provided an example script in the `model/` directory.


### Example Usages:
The following scripts demonstrate how to benchmark various scenarios. You can flexibly modify the arguments within these scripts to assess different preference styles, task type to create varying task difficulties.

#### Example 1: Benchmark Discriminative Tasks
```
bash discrimation_task/single_intent.sh
```


#### Example 2: Benchmark Generative Tasks
```
bash generation_task/single_intent.sh
```
