import base64
import json
import logging
from pathlib import Path

import httpx
import openai
from poemai_utils.ai_model import AIApiType
from poemai_utils.basic_types_utils import linebreak, short_display
from poemai_utils.openai.llm_answer_cache import LLMAnswerCache
from poemai_utils.openai.openai_model import OPENAI_MODEL
from poemai_utils.utils_config import get_config_by_key

_logger = logging.getLogger(__name__)

DISABLE_GPT_LOG = get_config_by_key("DISABLE_GPT_LOG")


def current_unix_time():
    import datetime
    import time
    from types import SimpleNamespace

    unix_timestamp = int(time.time() * 1000)
    dt = datetime.datetime.fromtimestamp(unix_timestamp / 1000)
    # Convert datetime object to ISO date format
    iso_date = str(dt.isoformat())

    return SimpleNamespace(unix_timestamp=unix_timestamp, dt=dt, iso_date=iso_date)


class Ask:
    OPENAI_MODEL = (
        OPENAI_MODEL  # to make it easier to import / access, just use Ask.OPENAI_MODEL
    )

    def __init__(
        self,
        model=OPENAI_MODEL.GPT_3_5_TURBO,
        log_file="promptlog.db",
        openai=openai,
        gpt_log=None,
        openai_api_key=None,
        llm_answer_cache: LLMAnswerCache = None,
        raise_on_cache_miss=False,
        disable_prompt_log=None,
        async_openai=None,
        check_token_count=False,
        base_url=None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "You must install openai to use this function. Try: pip install openai"
            )

        if openai_api_key is None:
            home = str(Path.home())
            with open(home + "/openai_api_key.txt", "r") as f:
                self._openai_api_key = f.read().strip()

        else:
            self._openai_api_key = openai_api_key

        openai_args = {}

        if openai_api_key is not None:
            openai_args["api_key"] = self._openai_api_key

        if base_url is not None:
            openai_args["base_url"] = base_url
            self.base_url = base_url
        else:
            self.base_url = None

        self.client = OpenAI(**openai_args)

        self.model = model
        openai.api_key = self._openai_api_key

        self.gpt_log = gpt_log

        self.llm_answer_cache = llm_answer_cache
        self._openai = openai
        self.raise_on_cache_miss = raise_on_cache_miss

        self.disable_prompt_log = True

        if async_openai is None:
            self.async_openai = AsyncOpenai(
                model=self.model,
                openai_api_key=self._openai_api_key,
                check_token_count=check_token_count,
                base_url=self.base_url,
            )
        else:
            self.async_openai = async_openai

    def get_last_token_statistics(self):
        if self.async_openai is not None:
            return self.async_openai.last_token_statistics

    @classmethod
    def count_tokens(cls, text):
        import tiktoken  # only needed if you want to count tokens

        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def ask_chat(
        self,
        prompt,
        temperature=0,
        max_tokens=100,
        stop=None,
        suffix=None,
        system_prompt=None,
        messages=None,
        json_mode=False,
        additional_args=None,
        structured_json_schema=None,
    ):
        if not AIApiType.CHAT_COMPLETIONS in self.model.api_types:
            raise ValueError(f"Model {self.model} does not support chat completions")

        message_list = self._calculate_messages_for_chat(
            prompt, system_prompt, messages
        )
        args = {}
        if stop is not None:
            args["stop"] = stop

        if json_mode:
            args["response_format"] = {"type": "json_object"}

        # add later
        # elif structured_json_schema is not None:
        #     args["response_format"] = {
        #         "type": "json_schema",
        #         "json_schema": structured_json_schema,
        #     }

        if temperature is not None:
            # Check if the model supports temperature parameter
            if (
                hasattr(self.model, "supports_temperature")
                and not self.model.supports_temperature
            ):
                # Skip adding temperature for models that don't support it
                pass
            else:
                args["temperature"] = temperature
        if additional_args is not None:
            args.update(additional_args)

        try:

            if max_tokens is not None:
                if self.model.requires_max_completion_tokens:
                    args["max_completion_tokens"] = max_tokens
                else:
                    args["max_tokens"] = max_tokens

            _logger.debug(
                f"Calling chat completions with model {self.model}, messages: {message_list}, temperature: {temperature}, args: {args}"
            )
            response_raw = self.client.chat.completions.create(
                messages=message_list,
                model=self.model.model_key,
                **args,
            )
            _logger.debug(f"Response raw: {response_raw}")
            response = response_raw.choices[0].message.content

        except Exception as e:
            _logger.error(e)
            raise e

        return response

    @staticmethod
    def _calculate_messages_for_chat(prompt, system_prompt, messages):
        if prompt is None and messages is None:
            raise ValueError("prompt or messages must be provided")
        message_list = []
        if system_prompt is not None:
            message_list.append({"role": "system", "content": system_prompt})
        if messages is not None:
            message_list.extend(messages)
        if prompt is not None:
            if not isinstance(prompt, list):
                message_list.append({"role": "user", "content": prompt})
        return message_list

    def ask_completion(
        self,
        prompt,
        temperature=0,
        max_tokens=600,
        stop=None,
        suffix=None,
        system_prompt=None,
        messages=None,
    ):
        if system_prompt is not None:
            print(f"Warning: system_prompt is not supported for {self.model}")
        if messages is not None:
            print(f"Warning: messages is not supported for {self.model}")
        if AIApiType.COMPLETIONS not in self.model.api_types:
            raise ValueError(f"Model {self.model} does not support completions")
        try:
            response = self.client.completions.create(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model.model_key,
                stop=stop,
                suffix=suffix,
            )["choices"][0]["text"].strip(" \n")
        except Exception as e:
            _logger.error(f"Error in ask_completion: {e}")
            raise e

        return response

    def ask(
        self,
        prompt,
        temperature=0,
        max_tokens=600,
        stop=None,
        suffix=None,
        system_prompt=None,
        messages=None,
        metadata=None,
        json_mode=False,
        additional_args=None,
    ):
        if metadata is None:
            metadata = {}
        if prompt is None and messages is None:
            raise ValueError("prompt or messages must be provided")
        answer = None
        cache_hit = False
        cache_key = None
        if additional_args is None:
            additional_args = {}
        if "temperature" in additional_args:
            temperature = additional_args["temperature"]
        if self.llm_answer_cache is not None:
            answer, cache_key = self.llm_answer_cache.fetch_from_cache(
                self.model.name,
                prompt,
                temperature,
                max_tokens,
                stop,
                suffix,
                system_prompt,
                messages,
            )
            if answer is not None:
                cache_hit = True
                _logger.info(
                    f"Cache hit for model {self.model} cache_key {cache_key} prompt *'{short_display(prompt,45)}'* for model {self.model}"
                )

        if not cache_hit:
            if self.raise_on_cache_miss:
                raise ValueError(
                    f"Cache miss for cache_key {cache_key} prompt [{short_display(prompt)} ] for model {self.model}"
                )
            _logger.debug(
                f"Cache miss for for model {self.model} cache_key {cache_key}"
            )
            if temperature is not None:
                # Only add temperature if the model supports it
                if (
                    hasattr(self.model, "supports_temperature")
                    and not self.model.supports_temperature
                ):
                    # Skip adding temperature for models that don't support it
                    pass
                else:
                    additional_args["temperature"] = temperature
            if AIApiType.CHAT_COMPLETIONS in self.model.api_types:
                # Only pass temperature if the model supports it
                if (
                    hasattr(self.model, "supports_temperature")
                    and not self.model.supports_temperature
                ):
                    # For models that don't support temperature, don't pass it
                    answer = self.ask_chat(
                        prompt,
                        temperature=None,  # Don't pass temperature
                        max_tokens=max_tokens,
                        stop=stop,
                        suffix=suffix,
                        system_prompt=system_prompt,
                        messages=messages,
                        json_mode=json_mode,
                        additional_args=additional_args,
                    )
                else:
                    # For models that support temperature, pass it normally
                    answer = self.ask_chat(
                        prompt,
                        temperature,
                        max_tokens,
                        stop,
                        suffix,
                        system_prompt,
                        messages,
                        json_mode=json_mode,
                        additional_args=additional_args,
                    )
            elif AIApiType.COMPLETIONS in self.model.api_types:
                answer = self.ask_completion(
                    prompt,
                    temperature,
                    max_tokens,
                    stop,
                    suffix,
                    system_prompt,
                    messages,
                )
            else:
                raise ValueError(f"Model {self.model} does not support completions")
            if self.llm_answer_cache is not None:
                self.llm_answer_cache.store_in_cache(
                    self.model.name,
                    prompt,
                    temperature,
                    max_tokens,
                    stop,
                    suffix,
                    system_prompt,
                    messages,
                    answer,
                )

        current_time = current_unix_time()

        if not (DISABLE_GPT_LOG or self.disable_prompt_log):
            self.gpt_log[current_time.unix_timestamp] = {
                "prompt": prompt,
                "answer": answer,
                "model": self.model.name,
                "iso_date": current_time.iso_date,
                "unix_timestamp": current_time.unix_timestamp,
                "metadata": metadata,
            }

        return answer

    def ask_with_images(
        self,
        prompt,
        image_paths,
        detail="high",
        temperature=0,
        max_tokens=600,
        stop=None,
        suffix=None,
        system_prompt=None,
        metadata=None,
    ):
        if not self.model.supports_vision:
            raise ValueError(f"Model {self.model} does not support vision")

        if metadata is None:
            metadata = {}
        if prompt is None and messages is None:
            raise ValueError("prompt or messages must be provided")
        answer = None

        prompt_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

        for image_path in image_paths:
            base64_image = self.encode_image(image_path)

            prompt_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                }
            )

        messages = [prompt_message]

        answer = self.ask_chat(
            prompt,
            temperature,
            max_tokens,
            stop,
            suffix,
            system_prompt,
            messages,
        )

        return answer

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def ask_async(
        self,
        prompt,
        temperature=0,
        max_tokens=600,
        stop=None,
        system_prompt=None,
        messages=None,
        metadata=None,
    ):
        if metadata is None:
            metadata = {}
        suffix = None
        answer = None
        cache_hit = False

        messages = self._calculate_messages_for_chat(prompt, system_prompt, messages)
        cache_key = None
        answer = None
        if self.llm_answer_cache is not None and False:  # Cache is broken for async
            answer, cache_key = self.llm_answer_cache.fetch_from_cache(
                self.model.name,
                prompt,
                temperature,
                max_tokens,
                stop,
                suffix,
                system_prompt,
                messages,
            )
            if answer is not None:
                cache_hit = True
                _logger.info(
                    f"Cache hit for model {self.model} cache_key {cache_key} prompt *'{short_display(prompt,45)}'* for model {self.model}"
                )

        if cache_hit:
            for part in answer.splitlines():
                yield {"role": "assistant", "content": part}
        else:
            cache_miss_message = f"Cache miss for cache_key {cache_key} prompt [{short_display(prompt)} ] for model {self.model}"
            if self.raise_on_cache_miss:
                raise ValueError(cache_miss_message)
            _logger.info(cache_miss_message)
            if AIApiType.CHAT_COMPLETIONS in self.model.api_types:
                answer = ""
                async for part in self.async_openai.ask_chat_async(
                    temperature,
                    max_tokens,
                    stop,
                    messages,
                ):
                    _logger.debug("part: %s", part)
                    part_content = part.get("content")
                    if part_content is not None:
                        answer += part_content
                    yield part

                # Store the answer in the cache
                if self.llm_answer_cache is not None:
                    self.llm_answer_cache.store_in_cache(
                        self.model.name,
                        prompt,
                        temperature,
                        max_tokens,
                        stop,
                        suffix,
                        system_prompt,
                        messages,
                        answer,
                    )

            else:
                raise ValueError(f"Model {self.model} is not supported for async")

    def print_log(self, limit=10):
        for key in list(sorted(self.gpt_log.keys()))[-limit:]:
            entry = self.gpt_log[key]
            print(f"----------------------")
            print(f"model: {entry['model']} time: {entry['iso_date']} ")
            print(f"------ PROMPT  -------")
            print(f"{linebreak(entry['prompt'])}")
            print(f"------ ANSWER  -------")
            print(f"{linebreak(entry['answer'])}")
            print(f"**********************")


class AskGPT_4(Ask):
    def __init__(
        self,
        log_file="promptlog.db",
        openai=openai,
        gpt_log=None,
        openai_api_key=None,
        llm_answer_cache: LLMAnswerCache = None,
        raise_on_cache_miss=False,
        disable_prompt_log=None,
        async_openai=None,
    ):
        super().__init__(
            OPENAI_MODEL.GPT_4,
            log_file=log_file,
            openai=openai,
            gpt_log=gpt_log,
            openai_api_key=openai_api_key,
            llm_answer_cache=llm_answer_cache,
            raise_on_cache_miss=raise_on_cache_miss,
            disable_prompt_log=disable_prompt_log,
            async_openai=async_openai,
        )


class AskGPT_3(Ask):
    def __init__(
        self,
        log_file="promptlog.db",
        openai=openai,
        gpt_log=None,
        openai_api_key=None,
        llm_answer_cache: LLMAnswerCache = None,
        raise_on_cache_miss=False,
        disable_prompt_log=None,
        async_openai=None,
    ):
        super().__init__(
            OPENAI_MODEL.GPT_3,
            log_file=log_file,
            openai=openai,
            gpt_log=gpt_log,
            openai_api_key=openai_api_key,
            llm_answer_cache=llm_answer_cache,
            raise_on_cache_miss=raise_on_cache_miss,
            disable_prompt_log=disable_prompt_log,
            async_openai=async_openai,
        )


class AskGPT_3_5_TURBO(Ask):
    def __init__(
        self,
        log_file="promptlog.db",
        openai=openai,
        gpt_log=None,
        openai_api_key=None,
        llm_answer_cache: LLMAnswerCache = None,
        raise_on_cache_miss=False,
        disable_prompt_log=None,
        async_openai=None,
    ):
        super().__init__(
            model=OPENAI_MODEL.GPT_3_5_TURBO,
            log_file=log_file,
            openai=openai,
            gpt_log=gpt_log,
            openai_api_key=openai_api_key,
            llm_answer_cache=llm_answer_cache,
            raise_on_cache_miss=raise_on_cache_miss,
            disable_prompt_log=disable_prompt_log,
            async_openai=async_openai,
        )


class AsyncOpenai:
    def __init__(
        self, model, openai_api_key, check_token_count=False, base_url=None, timeout=60
    ):

        self._openai_api_key = openai_api_key
        self.model = model
        self.check_token_count = check_token_count
        self.last_token_statistics = None

        OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
        if base_url is None:
            self.base_url = OPENAI_API_URL
        else:
            self.base_url = base_url + "/v1/chat/completions"
        self.httpx = httpx  # for testing, can be overridden after instantiation
        self.timeout = timeout

    async def ask_chat_async(
        self,
        temperature=0,
        max_tokens=600,
        stop=None,
        messages=None,
    ):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._openai_api_key}",
        }

        data = {
            "model": self.model.model_key,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Only add temperature if the model supports it
        if (
            hasattr(self.model, "supports_temperature")
            and not self.model.supports_temperature
        ):
            # Skip adding temperature for models that don't support it
            pass
        else:
            data["temperature"] = temperature
        if stop is not None:
            data["stop"] = stop

        if self.model.requires_max_completion_tokens:
            data["max_completion_tokens"] = max_tokens
        else:
            data["max_tokens"] = max_tokens

        full_text = ""
        usage_data = None

        timeout_config = httpx.Timeout(
            self.timeout
        )  # timeout for connect, read, write, and pool

        async with self.httpx.AsyncClient(timeout=timeout_config) as client:
            if self.check_token_count:
                self.last_token_statistics = None
            async with client.stream(
                "POST", self.base_url, headers=headers, content=json.dumps(data)
            ) as response:
                if response.status_code != 200:
                    response_text = await response.aread()
                    _logger.info(
                        f"response.status_code: {response.status_code} on url {self.base_url}, repsonse text: {response_text}"
                    )
                    raise RuntimeError(
                        "OpenAI API is unavailable at url " + self.base_url
                    )

                buffer = ""
                async for chunk_text in response.aiter_text():
                    buffer += chunk_text
                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        message = message.strip()
                        if message.startswith("data:"):
                            message = message[5:].strip()
                        if message == "":
                            continue
                        try:
                            chunk = json.loads(message)
                            if chunk.get("choices"):
                                if len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if delta:
                                        full_text += delta.get("content", "")
                                        yield delta
                            if (
                                chunk.get("usage") is not None
                            ):  # Capture usage data in the last chunk
                                usage_data = chunk["usage"]
                        except json.JSONDecodeError:
                            continue

                # Verify token counts
                if self.check_token_count:
                    if usage_data:
                        if "completion_tokens" in usage_data:
                            our_token_count = Ask.count_tokens(full_text)
                            api_reported_tokens = usage_data["completion_tokens"]
                            discrepancy = our_token_count != api_reported_tokens

                            self.last_token_statistics = {
                                "our_answer_tokens": our_token_count,
                                "api_reported_tokens": api_reported_tokens,
                                "discrepancy": discrepancy,
                            }
                            if discrepancy:
                                _logger.warning(
                                    f"Token count discrepancy detected. Counted: {our_token_count}, Reported: {api_reported_tokens}"
                                )
                            else:
                                _logger.info(
                                    f"Token count verified. Counted: {our_token_count}, Reported: {api_reported_tokens}"
                                )
