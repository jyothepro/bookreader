from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
import sys
import glob
import os

os.environ['OPENAI_API_KEY'] = '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'

def construct_index(directory_path):
  # set maximum input size
  max_input_size = 4096
  # set number of output tokens
  num_outputs = 256
  # set maximum chunk overlap
  max_chunk_overlap = 20
  # set chunk size limit
  chunk_size_limit = 600

  prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))
  
  # file = open('./content/test1.txt',mode='r')
  # all_of_it = file.read()
  # file.close()

  documents = SimpleDirectoryReader('./content/').load_data()
  
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
  
  index.save_to_disk('index2.json')
  
  return index


def ask_bot(input_index = 'index.json'):
    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    while True:
        query = input('What do you want to ask the bot?   \n')
        response = index.query(query, response_mode="compact")
        print ("\nBot says: \n\n" + response.response + "\n\n\n")

#index = construct_index("/content/")

ask_bot('index2.json')

