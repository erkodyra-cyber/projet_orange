#Toujours depuis la racine 

python agent_security_navigator/src/rag/pdf_to_chunks.py
python agent_security_navigator/src/rag/build_index.py

python agent_sensitive/src/rag/pdf_to_chunks_sensitive.py
python agent_sensitive/src/rag/build_index_sensitive.py


python -m uvicorn agent_security_navigator.src.api.atoa:app --reload --port 8101
python -m uvicorn agent_sensitive.src.api.atoa:app --reload --port 8102
python -m uvicorn orchestrator.src.api.app:app --reload --port 8000
 .\vault.exe server -config="config.hcl"
 $env:VAULT_ADDR = "http://127.0.0.1:8200"
C:\vault\vault.exe operator unseal "UNSEAL_KEY"
