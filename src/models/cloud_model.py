import time
import google.generativeai as genai
from typing import Any, List, Optional, Dict

# Import các class cốt lõi của LangChain để kế thừa
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.pydantic_v1 import Field, PrivateAttr

class GeminiChatModel(BaseChatModel):
    """
    Wrapper Gemini 1.5 Flash chuẩn LangChain.
    Hỗ trợ toán tử '|', JSON Mode và Retry.
    """

    # Khai báo các biến config (Pydantic Fields)
    model_name: str = "gemini-1.5-flash"
    api_key: str = Field(..., alias="api_key") # Bắt buộc phải có
    temperature: float = 0.0
    max_retries: int = 3

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:

        # 1. Cấu hình Gemini
        genai.configure(api_key=self.api_key)

        generation_config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json" # BẮT BUỘC TRẢ VỀ JSON
        }

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )

        # 2. Chuyển đổi Messages của LangChain thành Prompt String cho Gemini
        # (Flash hoạt động tốt nhất với prompt string gộp cho tác vụ Router)
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System Instruction:\n{msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User Question:\n{msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Model Answer:\n{msg.content}")
            else:
                prompt_parts.append(msg.content)

        final_prompt = "\n\n".join(prompt_parts)

        # 3. Gọi API với cơ chế Retry
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(final_prompt)

                # Lấy text kết quả
                content = response.text.strip()

                # Trả về kết quả đúng chuẩn LangChain
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"⚠️ Gemini Retry ({attempt+1}/{self.max_retries}): {e}")
                time.sleep(wait_time)

        # 4. Xử lý khi lỗi toàn tập (Fallback)
        print(f"❌ Gemini Failed: {last_error}")
        # Trả về JSON mặc định để không crash pipeline
        fallback_json = '{"genre": "rag", "rewrite": "ERROR_FALLBACK", "keywords": ""}'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback_json))])

    @property
    def _llm_type(self) -> str:
        return "gemini-chat-model"
    
