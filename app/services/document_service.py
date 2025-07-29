# --- app/services/document_service.py ---
import os
import uuid
import shutil
import hashlib
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    TextLoader
)

import plotly.express as px
from sqlalchemy import inspect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from app.services.database import DatabaseService as db_service


class DocumentService:
    def __init__(self, persist_dir: str = "vectorstore", registry_file: str = "index_registry.json"):
        self.persist_dir = persist_dir
        self.registry_file = registry_file
        self.structured_dir = os.path.join(persist_dir, "structured")
        self.unstructured_dir = os.path.join(persist_dir, "unstructured")
        os.makedirs(self.structured_dir, exist_ok=True)
        os.makedirs(self.unstructured_dir, exist_ok=True)
        self.db = db_service

        self.unstructured_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        self.structured_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", ",", " "]
        )

        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                json.dump([], f)

    def _select_embedding_model(self, structured: bool = False):
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if structured else "sentence-transformers/all-mpnet-base-v2"
        return HuggingFaceEmbeddings(model_name=model_name)

    def _is_structured_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in [".csv", ".xls", ".xlsx"]

    def _load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == ".csv":
                loader = CSVLoader(file_path, encoding='utf-8')
            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif ext == ".json":
                loader = JSONLoader(file_path, jq_schema=".", text_content=False)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"UNSUPPORTED_FILE_FORMAT: {ext}")
            return loader.load()
        except Exception as e:
            raise ValueError(f"FAILED_TO_LOAD_DOCUMENT: {str(e)}")

    def _process_structured_data(
            self,
            documents: List[Document],
            file_path: str,
            metadata_fields: Optional[List[str]] = None
    ) -> List[Document]:
        processed_docs = []

        try:
            # Load file
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            # Dataset-level summary
            summary_content = f"""
            Dataset Summary:
            - Total Rows: {len(df)}
            - Total Columns: {len(df.columns)}
            - Column Names: {', '.join(df.columns.tolist())}
            - Data Types: {df.dtypes.to_string()}
            """

            if len(df) > 0:
                summary_content += f"\n- Sample (Top 3 Rows):\n{df.head(3).to_string()}"

            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_content += f"\n- Stats:\n{df[numeric_cols].describe().to_string()}"

            summary_doc = Document(
                page_content=summary_content.strip(),
                metadata={"type": "summary", "source": os.path.basename(file_path)}
            )
            processed_docs.append(summary_doc)

            # Row-level documents
            for idx, row in df.iterrows():
                row_content = ", ".join(f"{col}: {row[col]}" for col in df.columns)

                # Custom metadata fields
                filtered_meta = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in row and pd.notnull(row[field]):
                            filtered_meta[field] = str(row[field])

                row_doc = Document(
                    page_content=row_content,
                    metadata={
                        "type": "row",
                        "row_index": idx,
                        "source": os.path.basename(file_path),
                        "row_metadata": json.dumps(filtered_meta, ensure_ascii=False)
                    }
                )
                processed_docs.append(row_doc)

            processed_docs = filter_complex_metadata(processed_docs)

        except Exception as e:
            print(f"❌ Error processing structured data: {e}")
            return documents  # Fallback

        return processed_docs

    def _compute_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_duplicate(self, file_hash: str) -> bool:
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
        return any(entry['file_hash'] == file_hash for entry in registry)

    def _update_registry(self, username: str, docname: str, file_path: str,
                         instruction: str, file_hash: str, is_structured: bool):
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)

        registry.append({
            "username": username,
            "docname": docname,
            "file_path": file_path,
            "instruction": instruction,
            "file_hash": file_hash,
            "is_structured": is_structured,
            "timestamp": datetime.now().isoformat()
        })

        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def process_and_store(self, file_path: str, username: str, docname: str,
                          instruction: str = "", metadata_fields: Optional[List[str]] = None) -> Dict[str, Any]:

        file_hash = self._compute_file_hash(file_path)
        if self._is_duplicate(file_hash):
            raise ValueError("DUPLICATE_DOCUMENT")

        is_structured = self._is_structured_file(file_path)
        raw_docs = self._load_document(file_path)

        if is_structured:
            processed_docs = self._process_structured_data(
                documents=raw_docs,
                file_path=file_path,
                metadata_fields=metadata_fields or []  # ← pass user-defined metadata
            )
            splitter = self.structured_splitter
            base_dir = self.structured_dir
            vectorstore_type = "structured"
        else:
            processed_docs = raw_docs
            splitter = self.unstructured_splitter
            base_dir = self.unstructured_dir
            vectorstore_type = "unstructured"

        embedding_model = self._select_embedding_model(structured=is_structured)
        chunks = splitter.split_documents(processed_docs)

        for doc in chunks:
            doc.metadata.update({
                "username": username,
                "docname": docname,
                "instruction": instruction,
                "is_structured": is_structured,
                "vectorstore_type": vectorstore_type,
                "source_document": docname
            })

        index_path = os.path.join(base_dir, f"{username}_{docname}")

        if is_structured:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=index_path
            )
            vectorstore.persist()
        else:
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            vectorstore.save_local(index_path)

        self._update_registry(username, docname, file_path, instruction, file_hash, is_structured)

        return {
            "index_path": index_path,
            "vectorstore_type": vectorstore_type,
            "is_structured": is_structured,
            "chunk_count": len(chunks)
        }

    def get_retriever(self, username: str, docname: str, search_type: str = "similarity"):
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)

        doc_info = next((e for e in registry if e['username'] == username and e['docname'] == docname), None)
        if not doc_info:
            raise ValueError(f"Document not found: {username}_{docname}")

        is_structured = doc_info.get('is_structured', False)
        embedding_model = self._select_embedding_model(structured=is_structured)

        index_path = os.path.join(
            self.structured_dir if is_structured else self.unstructured_dir,
            f"{username}_{docname}"
        )

        if is_structured:
            vectorstore = Chroma(persist_directory=index_path, embedding_function=embedding_model)
            return vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": 5})
        else:
            vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.6})

    def delete_document_vector(self, docname: str=""):
        username = "admin"
        structured_path = os.path.join(self.structured_dir, f"{username}_{docname}")
        unstructured_path = os.path.join(self.unstructured_dir, f"{username}_{docname}")
        deleted = False

        if os.path.exists(structured_path):
            shutil.rmtree(structured_path)
            deleted = True

        if os.path.exists(unstructured_path):
            shutil.rmtree(unstructured_path)
            deleted = True

        if deleted:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            registry = [r for r in registry if not (r['username'] == username and r['docname'] == docname)]
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)

        return deleted

    def list_documents(self, username: Optional[str] = None) -> List[Dict]:
        if not os.path.exists(self.registry_file):
            return []

        with open(self.registry_file, 'r') as f:
            registry = json.load(f)

        return [r for r in registry if r['username'] == username] if username else registry

    def get_document_stats(self, username: str, docname: str) -> Dict[str, Any]:
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)

        doc_info = next((e for e in registry if e['username'] == username and e['docname'] == docname), None)
        if not doc_info:
            raise ValueError(f"Document not found: {username}_{docname}")

        return {
            "username": username,
            "docname": docname,
            "is_structured": doc_info.get('is_structured', False),
            "timestamp": doc_info.get('timestamp'),
            "file_path": doc_info.get('file_path'),
            "instruction": doc_info.get('instruction', "")
        }
    def delete_all_documents(self):
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        os.makedirs(self.structured_dir, exist_ok=True)
        os.makedirs(self.unstructured_dir, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump([], f)

    def search_documents(self, query: str, username: str, docname: str, top_k: int = 5) -> List[Dict[str, Any]]:
        retriever = self.get_retriever(username, docname)
        results = retriever.get_relevant_documents(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None)
            } for doc in results[:top_k]
        ]

# --- Session Management (Requires DatabaseService injected externally) ---

    def view_all_sessions(self, username: str) -> List[Dict[str, Any]]:
        """Return all stored chat sessions"""
        return self.db.get_chat_history(username=username)

    def view_session_by_id(self, session_id: str) -> List[Dict[str, Any]]:
        """Return specific session by session ID"""
        all_sessions = self.db.get_chat_history(username="admin")
        return [s for s in all_sessions if s['session_id'] == session_id]

    def delete_session_by_id(self, session_id: int):
        """Delete a specific session record"""
        self.db.delete_session(session_id=session_id)

    def delete_all_sessions(self, username: Optional[str] = None):
        """Delete all sessions or sessions by user"""
        self.db.delete_all_sessions(username=username)
        
    def preview_table(self, table_name: str):
        self.db.getpreview(table_name=table_name)
    def generate_chart(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str):
        if chart_type == "line":
         return px.line(df, x=x_col, y=y_col)
        elif chart_type == "bar":
            return px.bar(df, x=x_col, y=y_col)
        elif chart_type == "scatter":
            return px.scatter(df, x=x_col, y=y_col)
        elif chart_type == "pie":
            return px.pie(df, names=x_col, values=y_col)
        else:
            raise ValueError("Invalid chart type.")
