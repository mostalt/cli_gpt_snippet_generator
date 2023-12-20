import argparse
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

def parse_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="")
  parser.add_argument("--lang", default="")
  args = parser.parse_args()

  return args

def run_chain():
  args = parse_cli_args()
  
  if not args.lang or not args.task:
    print("error: pass --task and --lang arguments. example: python main.py --lang='python' --task='print hello world string'")
    return

  llm = OpenAI()

  code_prompt = PromptTemplate(
      template="Write a very short {lang} function that will {task}",
      input_variables=["lang", "task"]
  )
  testing_prompt = PromptTemplate(
    input_variables=["language", "snippet"],
    template="Write a test for the following {lang} code:\n{snippet}"
  )

  code_chain = LLMChain(
      llm=llm,
      prompt=code_prompt,
      output_key="snippet"
  )
  testing_chain = LLMChain(
    llm=llm,
    prompt=testing_prompt,
    output_key="test"
  )

  sequence = SequentialChain(
    chains=[code_chain, testing_chain],
    input_variables=["task", "lang"],
    output_variables=["test", "snippet"]
  )

  result = sequence({
    "lang": args.lang,
    "task": args.task
  })

  print("GENERATED SNIPPET:")
  print(result["snippet"], "\n")

  print("GENERATED TEST:")
  print(result["test"])


run_chain()