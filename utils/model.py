import time
import random
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # 没装tqdm也不影响使用

class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_path: str,
        request_timeout: int = 60,     # 请求超时（秒）
        max_retries: int = 3,          # 最大重试次数
        backoff_base: float = 0.8,     # 指数退避基数（秒）
        verbose: bool = True,          # ✅ 是否打印动态日志
        logger: Optional[Callable[[str], None]] = None,  # ✅ 自定义日志函数
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self.model_path = model_path
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.verbose = verbose
        self._log = logger if logger is not None else print

    def _vlog(self, msg: str):
        if self.verbose:
            self._log(msg)

    def _sleep_backoff(self, attempt: int):
        # 指数退避 + 抖动
        delay = (self.backoff_base * (2 ** attempt)) * (0.8 + 0.4 * random.random())
        self._vlog(f"[Backoff] attempt={attempt} sleep={delay:.2f}s")
        time.sleep(delay)

    def _chat_once(self, messages, temperature, response_format, stop):
        # 单次请求；尽量把 timeout 也显式传入
        if self.model_path.startswith("gpt"):
            if self.model_path.startswith("gpt-5"):
                if response_format is not None:
                    return self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        reasoning_effort="minimal",
                        response_format=response_format,
                        stop=stop,
                        temperature=temperature,
                        timeout=self.request_timeout,
                    )
                else:
                    return self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        reasoning_effort="minimal",
                        stop=stop,
                        temperature=temperature,
                        timeout=self.request_timeout,
                    )
            else:
                if response_format is not None:
                    return self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        response_format=response_format,
                        stop=stop,
                        temperature=temperature,
                        timeout=self.request_timeout,
                    )
                else:
                    return self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        stop=stop,
                        temperature=temperature,
                        timeout=self.request_timeout,
                    )
        else:
            extra_body = {"truncate_prompt_tokens": 9000}
            if response_format is not None:
                extra_body["guided_json"] = response_format
            return self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                extra_body=extra_body,
                stop=stop,
                temperature=temperature,
                timeout=self.request_timeout,
            )

    def get_single_chat_completion(
        self,
        user_message: str,
        sys_prompt: str = "You are a helpful assistant",
        temperature: float = 0.0,
        response_format=None,
        stop: Optional[List[str]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]

        msg_len = len(str(messages))
        self._vlog(f"[Request] model={self.model_path} len(messages)≈{msg_len} temp={temperature}")

        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                completion = self._chat_once(
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    stop=stop,
                )
                elapsed = time.time() - t0
                content = completion.choices[0].message.content or ""
                self._vlog(f"[OK] model={self.model_path} attempt={attempt} time={elapsed:.2f}s out_len={len(content)}")
                return content
            except Exception as e:
                elapsed = time.time() - t0
                last_err = e
                self._vlog(f"[Error] model={self.model_path} attempt={attempt} time={elapsed:.2f}s err={e}")
                if attempt >= self.max_retries:
                    self._log(f"[Fail] model={self.model_path} exhausted retries. returning empty string.")
                    return ""
                self._sleep_backoff(attempt)

        # 理论上不会到这里
        self._log(f"[Unexpected] model={self.model_path} last_err={last_err}")
        return ""

    def get_multi_chat_completions(
        self,
        user_messages: List[str],
        sys_prompt: str = "You are a helpful assistant",
        temperature: float = 0,
        response_format=None,
        stop: Optional[List[str]] = None,
        max_workers: int = 8,                 # 限制并发，避免被限流
        per_task_timeout: Optional[int] = None,  # 单任务 future 超时（秒）
        show_progress: bool = True,           # ✅ 是否显示进度条
    ) -> List[str]:
        """
        并发获取多条消息的模型回复（带日志 & 可选进度条）。
        - 单任务内部：请求超时与重试；
        - 外层：future 层面的超时防护；
        - 任一任务失败不会影响整体。
        """
        self._vlog(f"[Batch] total={len(user_messages)} max_workers={max_workers} timeout={per_task_timeout}s")

        def fetch(idx_msg):
            idx, msg = idx_msg
            self._vlog(f"[Task {idx}] send")
            out = self.get_single_chat_completion(
                user_message=msg,
                sys_prompt=sys_prompt,
                temperature=temperature,
                response_format=response_format,
                stop=stop,
            )
            self._vlog(f"[Task {idx}] done len={len(out)}")
            return idx, out

        results = [""] * len(user_messages)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch, (i, m)) for i, m in enumerate(user_messages)]

            iterator = as_completed(futures)
            if show_progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Chat tasks")

            for fut in iterator:
                try:
                    idx, res = fut.result(timeout=per_task_timeout)
                    results[idx] = res
                except Exception as e:
                    # 找到该 future 的索引（可能稍慢，但稳妥）
                    i = futures.index(fut) if fut in futures else -1
                    self._log(f"[Warn] task_index={i} failed/timeout: {e}")

        self._vlog("[Batch] finished")
        return results


class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str, model_path: str, response_format=None, request_timeout: int = 30):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self.model_path = model_path
        self.response_format = response_format
        self.request_timeout = request_timeout

    def get_single_chat_completion(self, chat_messages: list, sys_prompt: str = "You are a helpful assistant"):
        messages = [{"role": "system", "content": sys_prompt}] + chat_messages
        try:
            if self.response_format is not None:
                if self.model_path.startswith("gpt"):
                    completion = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        response_format=self.response_format,
                        timeout=self.request_timeout   # ✅ 请求超时
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        extra_body=dict(guided_json=json_template),
                        timeout=self.request_timeout   # ✅ 请求超时
                    )
            else:
                completion = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    timeout=self.request_timeout   # ✅ 请求超时
                )

            return completion.choices[0].message.content

        except requests.exceptions.Timeout:
            print(f"[Timeout] Model request exceeded {self.request_timeout}s.")
            return ""   # 返回空字符串，避免卡死

        except Exception as e:
            print(f"[Error] Chat completion failed: {e}")
            return ""   # 返回空字符串，避免卡死

    def get_multi_chat_completions(self, chat_messages: list[list], sys_prompt: str = "You are a helpful assistant"):
        """
        并发获取多条消息的模型回复。

        参数:
            messages (list): 包含用户输入的消息内容的列表。
            sys_prompt (str): 系统提示（默认是 "You are a helpful assistant"）。

        返回:
            list: 多条消息的模型回复列表。
        """
        # 为每个消息准备参数
        def fetch_response(user_message):
            return self.get_single_chat_completion(user_message, sys_prompt)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_response, chat_messages))

        return results
