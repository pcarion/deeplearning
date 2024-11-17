import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator
from langchain_core.prompts import PromptTemplate


# define the OpenAI LLM model name
llmModelName = "gpt-4o-mini"

# reference document
# https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

def display_properties(obj):
    print("Top-level properties of response object:")
    for prop in dir(obj):
        if not prop.startswith('_'):  # Skip internal/private properties
            print(f"- {prop}")

def display_json_with_properties(obj, properties):
    def convert_to_dict(item):
        result = {}
        
        # Add all available properties
        for prop in properties:
            if hasattr(item, prop):
                result[prop] = getattr(item, prop)
        
        # Always include the type
        result['type'] = item.__class__.__name__
        return result
            
    # Use default parameter in json.dumps to handle non-serializable objects
    print(json.dumps(obj, 
                    default=convert_to_dict,    
                    indent=2, 
                    ensure_ascii=False))
    
def display_json(obj):
    print(json.dumps(obj, 
                    indent=2, 
                    ensure_ascii=False))
    
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")
    
class MovieReview(BaseModel):
    isPositive: bool = Field(description="is the review positive or negative")
    summary: str = Field(description="a summary of the review")
    actors: list[str] = Field(description="a list of actors in the movie")



model = ChatOpenAI(model=llmModelName)

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate(
    template="Analyze the following movie review.\n{format_instructions}\n{review}\n",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
for i in range(1, 5):
    review_file = f"review{i:02d}.txt"
    output = prompt_and_model.invoke({"review": read_text_file(review_file)})
    response = parser.invoke(output)
    print(f"\nResults for {review_file}:")
    print(response)
    display_json_with_properties(response, ['isPositive', 'summary', 'actors'])
