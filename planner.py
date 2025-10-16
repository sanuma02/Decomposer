from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

def load_prompt_template(file_path="prompts/planner_prompt.txt") -> str:
    content = ''
    print("load_prompt_template")
    with open(file_path, "r") as file:
        content = file.read()
    return content

def create_query_plan(user_query: str) -> str:
    print("create_query_plan")
    prompt = load_prompt_template().format(user_query)
    if prompt:

        response = client.responses.create(
            model="gpt-5",
            input=prompt
        )

    print(response.output_text)