import unittest
from src.all_api_chat import AllApiChatModel
#from langchain_core.messages import AIMessage, HumanMessage
import os #Import thư viện os

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)

from dotenv import load_dotenv
load_dotenv()


class TestAllApiChatModel(unittest.TestCase):
    def test_llm_type(self):
        chat_model = AllApiChatModel(api_key='test_key', endpoint='https://test.api', model='test_model')
        self.assertEqual(chat_model._llm_type, "all_api")
    
    def test_identifying_params(self):
        chat_model = AllApiChatModel(api_key='test_key', endpoint='https://test.api', model='gpt-4o')
        self.assertEqual(chat_model._identifying_params, {'model_name': chat_model.model_name})

    def test_generate(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        chat_model = AllApiChatModel(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        system_message = SystemMessage(content="You are a helpful assistant.")
        human_message = HumanMessage(content="Who won the world series in 2020?")
        result = chat_model._generate(messages=[system_message, human_message])
        self.assertIsNotNone(result)

    def test_invoke(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        chat_model = AllApiChatModel(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        system_message = SystemMessage(content="You are a helpful assistant.")
        human_message = HumanMessage(content="Who won the world series in 2020?")
        result = chat_model.invoke([system_message, human_message])
        self.assertIsNotNone(result.content)
        
    def test_batch(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        chat_model = AllApiChatModel(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        result = chat_model.batch(["Who won the world series in 2020?", "Who is the president of the United States?"])
        self.assertIsNotNone(result[0].content)
        
    def test_stream(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        chat_model = AllApiChatModel(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        
        content = ""
        for chunk in chat_model.stream("How to make a cake?"):
            content += chunk.content
            #print(chunk.content, end="|")
        self.assertIsNotNone(content)