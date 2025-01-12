from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field
import requests
import json

class AllApiLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    api_key: str = Field(alias="api_key")
    api_url: str = Field(alias="endpoint")
    model_name: str = Field(alias="model")
    tools: List[str] = []
    knowledge: List[str] = []
    """The name of the model"""
    temperature: Optional[float] = None

    
    

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
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
            "messages": [
                { "type": "human", "content": prompt }
            ],
#            "prompt": prompt
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        json_resposne = response.json()
        tokens = json_resposne.get('data', {}).get('content', '')
        #print(prompt)
        return str(tokens)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
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
            "messages": [
                { "type": "human", "content": prompt }
            ]
        }
        response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    try:
                        json_data = json.loads(decoded_line[6:])  # Strip 'data: ' prefix
                        content = json_data.get("content", "")
                        if content is not None:
                            chunk = GenerationChunk(text=str(content))
                            if run_manager:
                                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                            yield chunk
                    except json.JSONDecodeError:
                        yield GenerationChunk(text="")
                    except Exception:
                        yield GenerationChunk(text="")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "all_api"
