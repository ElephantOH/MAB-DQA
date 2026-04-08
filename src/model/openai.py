from src.model.base import BaseVLMClient
from pydantic import BaseModel, Field
from openai import OpenAI
import base64
import os
CURRENT_FILE_PATH = os.path.abspath(__file__)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
class OpenaiConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model_name: str = Field(default="gpt-4o")
    api_key: str
    
    temperature: float = Field(default=0.0)
    max_new_tokens: int = Field(default=1024)


class OpenAIClient(BaseVLMClient):
    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        self.temperature = self.config.temperature
        self.max_new_tokens = self.config.max_new_tokens

        print(f"[Info] [{CURRENT_FILE_PATH}] Model Using: {self.config.model_name}.")
        assert self.config.api_key is not None, f"[Error] [{CURRENT_FILE_PATH}] API Key is Required for OpenAIClient."
        self.client = OpenAI(
            api_key=self.config.api_key,
        )
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
    
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    def predict(
            self,
            question,
            texts=None,
            images=None,
            history=None
        ):
        messages = self.process_message(question, texts, images, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        result = response.choices[0].message.content
        messages.append(self.create_ans_message(result))
        return result, messages
    
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
    