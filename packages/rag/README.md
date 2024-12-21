# Retrieval-Augmented Generation (RAG)

This package contains custom components and implementations for LlamaIndex framework for Retrieval-Augmented Generation.

Components with defaults are provided in `rag/components`.

The following custom components are implemented in this package:

* custom data reader
* custom LlamaIndex base node schemas which support multiple embeddings per node (e.g. dense and sparse embeddings)
* custom embeddings compatible with BGE-M3 model which generates both dense and sparse embeddings
* custom fusion functions, which includes distribution-score based fusion
* custom reranker using BGE-M3 reranker
* custom LLMs which extends LlamaIndex's OpenAILike class
* custom node parsers which perform semantic text chunk parsing and decomposition of text using sentence splitter and atomic claim parser
* custom qdrant vectorstore wrapper
