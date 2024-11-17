import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

# reference document
# https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

def display_properties(obj):
    print("Top-level properties of response object:")
    for prop in dir(obj):
        if not prop.startswith('_'):  # Skip internal/private properties
            print(f"- {prop}")

def display_json(obj):
    def convert_to_dict(item):
        # Define properties to serialize
        properties = ['content', 'additional_kwargs', 'response_metadata', 'usage_metadata']
        result = {}
        
        # Add all available properties
        for prop in properties:
            if hasattr(item, prop):
                result[prop] = getattr(item, prop)
        
        # Always include the type
        result['type'] = item.__class__.__name__
        return result
            
    # Use default parameter in json.dumps to handle non-serializable objects
    print(json.dumps(convert_to_dict(obj),
                    indent=2, 
                    ensure_ascii=False))


model = ChatOpenAI(model="gpt-4o-mini")
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

response = model.invoke(messages)

display_json(response)

