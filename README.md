# AskBio : LLM Powered chatbot for answering biology-related questions.

### Features
- Uses Large Language Models (LLMs) to generate responses based on the context of the question asked.
- Provides detailed explanations for complex biology concepts.
- Can answer questions about genes, proteins, cell types, diseases etc.

## Action Items
### Index Phase
  - Data Preperation
  - Chunking/Spliting strategy
    - Knowledge Graph embedding strategy??
  - Select VectorDB
    - Refer benchmark data
  - Select Embedding model
    - Refer Leaderboard
  - Q-A Pair Generation for testing, evaluation
### Query Phase
  - Select LLM Model
  - Query Transformer(Query Routing, Translating)
  - Guradrails (Out-of-context, Abusive, vulgar, etc.)
  - Retreiveing and Reranking techniques
- Evaluate the performance of the model
- Deploy the chatbot on a platform (like Streamlit) for user interaction.

## Tools
- llamaindex
- PyMupdf/unstructred lib
- Gradio?
- Milvus

## Design:
- HuggingFace/LLM service
- Vector DB service
- Evauation framework
- Prompting and related stuff
- Frontend service (Streamlit)


