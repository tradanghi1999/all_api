import unittest
from src.all_api_llm import AllApiLLM
#from langchain_core.messages import AIMessage, HumanMessage
import os #Import thư viện os

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import json

from dotenv import load_dotenv
load_dotenv()


class TestAllApiLLM(unittest.TestCase):
    def test_llm_type(self):
        llm = AllApiLLM(api_key='test_key', endpoint='https://test.api', model='test_model')
        self.assertEqual(llm._llm_type, "all_api")
        
    def test_identifying_params(self):
        llm = AllApiLLM(api_key='test_key', endpoint='https://test.api', model='test_model', temperature=0.5)
        self.assertEqual(llm._identifying_params, {'model_name': llm.model_name})

    def test_call(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        result = llm._call("Who won the world series in 2020?")
        self.assertIsNotNone(result)
        
    def test_invoke(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        result = llm.invoke([HumanMessage(content="Who won the world series in 2020?")])
        self.assertIsNotNone(result)
    
    def test_batch(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        result = llm.batch(["who won the world series in 2020?", "who is the president of the United States?"])
        self.assertIsNotNone(result[0])
    
    def test_stream(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        
        tokens = ''
        
        for token in llm.stream("who won the world series in 2020?"):
            tokens += token
            #print(token)
        
        self.assertGreater(len(tokens), 0)
    
    def test_simple_chain(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')  
        model_name = 'gpt-4o-mini'  
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}")
        ])    
        
        chain = prompt | llm
        
        result = chain.invoke({
                "input": "who won the world series in 2020?",
        })
        
        self.assertIsNotNone(result)

    def test_complex_chain(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')
        model_name = 'gpt-4o-mini'
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        prompt = ChatPromptTemplate.from_messages(
            [("system", "you are a girl name Anna"), ("human", "Hi, how old are you?"), ("ai", "Hi, I am 21."), ("human", "{input}")]
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
                "input": "Who are you and tell me how old are you?",
        })
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_json_chain(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')
        model_name = 'gpt-4o-mini'
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        prompt = ChatPromptTemplate.from_messages(
            [("system", "you are a helpful AI assistant that responds in JSON format"), 
             ("human", "{input}")]
        )
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({
                "input": "Your are Anna and your are 18 years old. Return a JSON object with your name and age",
        })
        
        #print(result)
        self.assertIsNotNone(result)
        # Verify the result is valid JSON
        try:
            json.loads(json.dumps(result))
        except ValueError:
            self.fail("Output is not valid JSON")

    def test_complex_chain_with_messages_objects(self):
        url = os.getenv('ALL_API_URL')
        api_key = os.getenv('ALL_API_API_KEY')
        model_name = 'gpt-4o-mini'
        llm = AllApiLLM(api_key=api_key, endpoint=url, model=model_name, temperature=0.5)
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content="you are a helpful AI assistant that responds in JSON format"), 
             HumanMessage(content="Your are Anna and your are 18 years old. Return a JSON object with your name and age")]  
        )
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({})
        #print(result)
        self.assertIsNotNone(result)
        # Verify the result is valid JSON
        try:
            json.loads(json.dumps(result))
        except ValueError:
            self.fail("Output is not valid JSON")
        


