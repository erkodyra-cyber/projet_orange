#Toujours depuis la racine 

python agent_security_navigator/src/rag/pdf_to_chunks.py
python agent_security_navigator/src/rag/build_index.py

python agent_sensitive/src/rag/pdf_to_chunks_sensitive.py
python agent_sensitive/src/rag/build_index_sensitive.py


python -m uvicorn agent_security_navigator.src.api.atoa:app --reload --port 8101
python -m uvicorn agent_sensitive.src.api.atoa:app --reload --port 8102
python -m uvicorn orchestrator.src.api.app:app --reload --port 8000
python -m uvicorn fake_idp.server:app --reload --port 8080