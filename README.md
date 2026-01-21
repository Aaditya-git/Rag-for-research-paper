# Project

Describe your project here.
=======
# rag_clean

## Setup
1. Create venv and install
   pip install -r requirements.txt

2. Put PDFs in:
   data/pdfs/

3. Set OpenAI key:
   export OPENAI_API_KEY="..."

## Run
1. Extract and chunk to cache
   python src/extract_chunk.py

2. Build Milvus index
   python src/index_milvus.py

3. Ask questions
   python src/qa_cli.py

