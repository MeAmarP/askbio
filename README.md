# AskBio : LLM Powered chatbot for answering biology-related questions.

AskBio is a command-line chatbot designed to answer questions by leveraging the power of Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) technology. It is specially tuned to provide responses based on the context information extracted from the "Concepts of Biology" text.

## Features

- **Language Model**: Utilizes the `microsoft/Phi-3-mini-4k-instruct` model for generating responses.
- **Embedding Model**: Uses `BAAI/bge-large-en-v1.5` model for generating embeddings of text data. This is used for semantic similarity and information retrieval.
- **Retrieval-Augmented Generation**: Enhances response accuracy by retrieving relevant information from a vector-indexed database of documents.
- **Quantized Model Deployment**: Utilizes quantization for efficient memory usage, making it suitable for deployment on systems with limited resources.
- **Continuous Interaction**: Runs in a command-line interface (CLI) until the user decides to exit.

## Install Dependencies
```bash
pip install -q transformers
pip -q install sentence-transformers
pip install -q llama-index
pip install llama-index-llms-huggingface
pip install llama-index-embeddings-huggingface
pip3 install torch torchvision torchaudio
pip install accelerate
pip install -i https://pypi.org/simple/ bitsandbytes
pip install pymupdf
pip install python-dotenv
```
## Usage
To run the AskBio chatbot, execute the following command in your terminal:
```bash
python askbio.py
```
Follow the on-screen prompts to enter your questions. Type `/bye` to exit the chatbot interface.

## Example Interaction
```
$ Welcome to AskBio. Enter your question or type '/bye' to exit.
$ Enter your question: How does photosynthesis work?
$ Response: Photosynthesis is a process used by plants, algae, and certain bacteria to harness energy from sunlight into chemical energy.
$ Enter your question: /bye
$ Exiting AskBio. Goodbye!
```
---
## Action Items
  - [x] Select LLM Model
    - `microsoft/Phi-3-mini-4k-instruct`
  - [x] Select Embedding model
    - `BAAI/bge-large-en-v1.5`
      - Embedding Dimensions = 1024 
      - Max Tokens = 512.
### Index Phase
  - [x] Data Preperation
  - [x] Chunking/Spliting strategy
    - Sentence Splitter with chunking size of 512
  - [ ] Select VectorDB
    - Refer benchmark data
### Post Processing Phase
  - [ ] Guradrails (Out-of-context, Abusive, vulgar, etc.)
  - [ ] Retreiveing and Reranking techniques
- [X] Evaluate the performance
  - [x] Q-A Pair Generation for testing, evaluation
  - [ ] Response Evaluation
  - [ ] Retrieval Evaluation
### Others
  - [ ] Conversation History for context understanding
### Deployment Phase
- Deploy the chatbot on a platform (like Streamlit) for user interaction.

## Tools
- llamaindex
- PyMupdf/unstructred lib
- Gradio?
- Milvus


