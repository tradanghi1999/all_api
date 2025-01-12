from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import requests
import json


class AllApiChatModel(BaseChatModel):
    """A custom chat model that run with AllApi
    """
    api_key: str = Field(alias="api_key")
    api_url: str = Field(alias="endpoint")
    model_name: str = Field(alias="model")
    tools: List[str] = []
    knowledge: List[str] = []
    """The name of the model"""
    temperature: Optional[float] = None

    # các params all api không hỗ trợ
    #max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    #parrot_buffer_length: int
    """The number of characters from the last message of the prompt to be echoed."""
    
    def _get_messages(self, messages: List[BaseMessage]):
        json_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                json_messages.append({
                    "type": "system",
                    "content": message.content
                })
            elif isinstance(message, HumanMessage):
                json_messages.append({
                    "type": "human",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                json_messages.append({
                    "type": "ai",
                    "content": message.content
                })
        return json_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        #print("prompt1: ", prompt)
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "api_key": self.api_key,
            "stream": False,
            "model": self.model_name,
            "temperature": self.temperature,
            "tools": self.tools,
            "knowledge": self.knowledge,
            "messages": self._get_messages(messages),
#            "prompt": prompt
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        json_resposne = response.json()
        tokens = json_resposne.get('data', {}).get('content', '')
        ct_input_tokens = int(json_resposne.get('data', {}).get('usage_metadata', {}).get('input_tokens', 0))
        ct_output_tokens = int(json_resposne.get('data', {}).get('usage_metadata', {}).get('output_tokens', 0))
        
        #print(ct_input_tokens, ct_output_tokens)
        
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        #print("prompt1: ", prompt)
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "api_key": self.api_key,
            "stream": True,
            "model": self.model_name,
            "temperature": self.temperature,
            "tools": self.tools,
            "knowledge": self.knowledge,
            "messages": self._get_messages(messages),
#            "prompt": prompt
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        latest_metadata = None
        for line in response.iter_lines():
          if line:
              decoded_line = line.decode('utf-8')
              if decoded_line.startswith("data: "):
                  try:
                    json_data = json.loads(decoded_line[6:])  # Strip 'data: ' prefix
                    content = json_data.get("content", "")
                    latest_metadata = json_data.get("usage_metadata", {})
                    if not content == None:
                    # Print the content on the same line
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=content)
                        )
                        if run_manager:
                            # This is optional in newer versions of LangChain
                            # The on_llm_new_token will be called automatically
                            run_manager.on_llm_new_token(content, chunk=chunk)
                        yield chunk
                  except json.JSONDecodeError as e:
                    yield("")
                  except Exception as e:
                    yield("")
            
            
        last_chunk = ChatGenerationChunk(
            message=AIMessageChunk(content=""),
            metadata=latest_metadata,
        )
        
        if run_manager:
            run_manager.on_llm_new_token("", chunk=last_chunk)
            
        yield last_chunk
        
            
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "all_api"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
