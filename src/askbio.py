import fitz  # PyMuPDF
import os
from dotenv import load_dotenv, find_dotenv
import nest_asyncio

import torch
from transformers import BitsAndBytesConfig

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.llms import ChatMessage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor



nest_asyncio.apply()
load_dotenv(find_dotenv())

class AskBio:
    def __init__(self) -> None:
        self.askbio_prefix = self._init_askbio_template()
        self.embedder = self._init_embedding_model()
        self.llm = self._init_llm()
        self.index = self.get_index()
        self.retriver = VectorIndexRetriever(self.index, similarity_topk=3)
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriver,
            response_synthesizer=get_response_synthesizer(llm=self.llm),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)])

    def _init_embedding_model(self):
        """Initialize the embedding model for embeddings generation

        Returns:
            Embeddings model: HuggingFaceEmbedding
        """
        return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5",device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def _init_askbio_template(self):
        """Initialize the prompt template for asking bio related questions

        Returns:
            PromptTemplate: PromptTemplate object with template string.
        """

        template = ("Your name is AskBio. You are AI chatbot who can answer question by using provided context information from book named Concepts of Biology"
        "Be more specific and do not facricate the answers.\n" 
        "If you are unsure about answer, please ask for clarfications.\n"
        "Use the provided context information below to answer the user questions. \n"
        "-------------------------------------------\n"
        "{context_str}" 
        "\n -------------------------------------------\n"
        "Given this information, please answer user question: {query_str} \n")
        askbio_template = PromptTemplate(template)
        return askbio_template

    
    def _init_llm(self):
        """Initialize the Language Model for generating responses
        Returns:
            HuggingFaceLLM : Language Model object with model name and device type.
        """
        # quantize to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        def messages_to_prompt(messages):
            """Convert messages to a single prompt string for the model

            Args:
                messages (List[Message]):  List of Message objects to convert to a single string as a prompt.

            Returns:
                Prompt string for the model.
            """
            prompt = ""
            for message in messages:
                if message.role == 'system':
                    prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                    prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                    prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"
            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"
        
        llm = HuggingFaceLLM(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
            context_window=3900,
            max_new_tokens=512,
            model_kwargs={"quantization_config": quantization_config,
                          "trust_remote_code": True},
            generate_kwargs={"temperature": 0.0},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            device_map="auto"
        )
        return llm
    
    def get_embedding(self, text: str):
        """This function returns the embedding of a text string using the embedder model.

        Args:
            text (str): The text to be embedded

        Returns:
            The embedding of the text string.
        """
        return self.embedder.get_text_embedding(text)
    
    def get_response(self, prompt: str):
        """This function returns the response from the LLM model.
        
        Args:
            prompt (str): The prompt to be used for the LLM model.
        
        Returns:
            The response from the LLM model.
        """
        response = self.llm.complete(prompt)
        return response
    
    def get_index(self):
        """"This function creates the index of loaded documents.
        """
        docs = SimpleDirectoryReader(
            input_dir="../data/sample/",
            recursive=True,
            required_exts=[".pdf"]).load_data()
        splitter = SentenceSplitter(chunk_size=512,chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(docs)
        index = VectorStoreIndex(
                        nodes=nodes,
                        use_async=True,
                        embed_model=self.embedder,
                        show_progress=False,)
        return index
        
    def ask(self, user_query: str):
        """This function returns the response from the LLM model based on user query and retrieved documents.

        Args:
            user_query (str): The user's question or statement.

        Returns:
            The response from the LLM model based on user's question or statement and retrieved documents.
        """
        nodes = self.retriver.retrieve(user_query)
        context = " ".join(node.get_text() for node in nodes)
        prompt = self.askbio_prefix.format(context_str=context, query_str=user_query)
        response = self.query_engine.query(prompt)
        return str(response)
    
if __name__ == "__main__":
    # import traceback
    # try:
    #     askbio = AskBio()
    #     print("-x-"*20)
    #     response = askbio.ask("How does the sensitivity of this plant help it survive in its environment?")
    # except Exception as e:
    #     traceback.print_exc()
    #     print('An error occurred:', e)
    import traceback
    askbio = AskBio()
    print("-x-"*20)
    print("Welcome to AskBio. Enter your question or type '/bye' to exit.")
    try:
        while True:
            user_input = input("Enter your question: ")
            if user_input == "/bye":
                print("Exiting AskBio. Goodbye!")
                break
            try:
                response = askbio.ask(user_input)
                print("Response:", response)
            except Exception as e:
                print('An error occurred while processing your question:', e)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        traceback.print_exc()
        print('An error occurred:', e)
