python3 qa_simple_RAG_test.py --exp_name "original_rag_prompt" --advanced_prompt False
python3 qa_simple_RAG_test.py --exp_name "few_shot_prompt"
python3 qa_simple_RAG_test.py --exp_name "small_chunk_higher_K" --advanced_prompt False  --chunk_size 250 --retrieve_k_docs 10