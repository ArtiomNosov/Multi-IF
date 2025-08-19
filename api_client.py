# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import threading
from requests.exceptions import RequestException

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from mistralai import Mistral
from openai import OpenAI

from utils import GenerationSetting

# Global variables for GigaChat client management
CLIENT = None
LOCK = threading.Lock()
API_MAX_RETRY = 3
API_RETRY_SLEEP = 1
API_REQUEST_DELAY = 0.0  # Global delay between API requests in seconds
API_ERROR_OUTPUT = "Error: Failed to get response from GigaChat API"

def get_api_bot(model_name, generation_setting):
    if OpenAIBot.check_name(model_name):
        return OpenAIBot(model_name, generation_config=generation_setting)
    elif AnthropicBot.check_name(model_name):
        return AnthropicBot(model_name, generation_setting)
    elif GeminiBot.check_name(model_name):
        return GeminiBot(model_name, generation_setting)
    elif MistralBot.check_name(model_name):
        return MistralBot(model_name, generation_setting)
    elif GigaChatBot.check_name(model_name):
        return GigaChatBot(model_name, generation_setting)
    else:
        raise NotImplementedError(f"The model {model_name} is not supported yet.")

class APIBot:

    def __init__(self, model, generation_config):
        self.model_name = model
        self.generation_config = generation_config

    def generate(self, messages):
        ...

    def check_name(self, name):
        ...


class OpenAIBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        self.client = OpenAI()

    def generate(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.generation_config.max_new_tokens,
            seed=self.generation_config.seed,
            top_p=self.generation_config.top_p,
            temperature=self.generation_config.temperature
        )
        return response.choices[0].message.content

    @staticmethod
    def check_name(name):
        if name in ['o1-preview',
                    'o1-mini',
                    'o1-preview-2024-09-12',
                    'o1-mini-2024-09-12',
                    'gpt-4-turbo',
                    'gpt-4-turbo-2024-04-09',
                    'gpt-4-turbo-preview',
                    'gpt-4-0125-preview',
                    'gpt-4-1106-preview',
                    'gpt-4',
                    'gpt-4-0613',
                    'gpt-4o-2024-08-06'
                ]:
            return True
        return False


class AnthropicBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        # make sure ti set the API key for anthropic,
        # which will be accessed via os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic()

    def generate(self, messages):
        # Anthropic models don't support manual seed.
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.generation_config.max_new_tokens,
            messages=messages,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p
        )
        return response.content[0].text

    @staticmethod
    def check_name(name):
        if name in ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-5-sonnet-20240620']:
            return True
        return False


class GeminiBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        # Configure the genai client with the API key

        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel(model)
        self.generation_config = generation_config

    def generate(self, messages):
        role_map = {
            'user': 'user',
            'assistant': 'model'
        }
        history = []
        for m in messages:
            history.append({'role': role_map[m['role']], 'parts': m['content']})
        chat =  self.model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]['content'],
            generation_config={
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p,
                "max_output_tokens": self.generation_config.max_new_tokens
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text

    @staticmethod
    def check_name(name):
        if name in [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
        ]:
            return True
        return False


class MistralBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        self.model = model

    def generate(self, messages):
        response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.generation_config.temperature,
            random_seed=self.generation_config.seed,
            max_tokens=self.generation_config.max_new_tokens,
            top_p=self.generation_config.top_p
        )
        return response.choices[0].message.content

    @staticmethod
    def check_name(name):
        if name in [
            'mistral-large-latest',
            'mistral-small-latest',
            # Add any other supported model names here
        ]:
            return True
        return False


class GigaChatBot(APIBot):
    def __init__(self, model, generation_config):
        super().__init__(model, generation_config)
        # GigaChat client will be initialized lazily when first used

    def generate(self, messages):
        global CLIENT

        with LOCK:
            if CLIENT is None:
                try:
                    from giga import GigaChat
                    CLIENT = GigaChat()
                except ImportError:
                    raise ImportError("GigaChat module not found. Please install it with: pip install giga @ git+https://git@gitlab.com/ru.ush/lightweight-gigachat.git@module")

        assert CLIENT is not None
        output = API_ERROR_OUTPUT
        for _ in range(API_MAX_RETRY):
            try:
                response = CLIENT.chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.generation_config.temperature,
                    max_tokens=self.generation_config.max_new_tokens,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except RequestException as e:
                print(type(e), e)
                time.sleep(API_RETRY_SLEEP)

        return output

    @staticmethod
    def check_name(name):
        if name.lower() in [
            'gigachat',
            'giga',
            'giga-chat',
            # Add any other supported GigaChat model names here
        ]:
            return True
        return False


if __name__ == '__main__':
    generation_setting = GenerationSetting(max_new_tokens=1024, temperature=0.6, top_p=0.9)
    bot = get_api_bot('mistral-small-latest',generation_setting)
    history = [
        {'role': 'user', 'content': 'create an equation.'},
        {'role': 'assistant', 'content': 'x^2-4x+4=0'},
        {'role': 'user', 'content': 'solve the equation.'}
    ]
    print(bot.generate(history))
