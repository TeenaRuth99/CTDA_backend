# --- app/routes/api_router.py --- (Enhanced Version with Better LLM Integration)
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query,Depends
from typing import List, Optional, Dict, Any
import os
import shutil
import pandas as pd
import logging
from fastapi.responses import Response,JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.database import DatabaseService
from app.services.cachemanager import CacheManager
import json
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBasic()
# Initialize services with proper dependencies
document_service = DocumentService()
llm_service = LLMService(document_service=document_service)  # Pass document_service to LLM
db_service = DatabaseService()
cache_manager = CacheManager()

@router.get("/auth")
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    valid_users = {
        "admin": "admin",  # Admin role
        "guest": "guest"  # Guest role
    }

    if credentials.username in valid_users and credentials.password == valid_users[credentials.username]:
        return {"username": credentials.username, "role": "Admin" if credentials.username == "admin" else "Guest"}

    raise HTTPException(status_code=401, detail="Invalid credentials")

# --- Document Upload with Enhanced Summary ---
# @router.post("/upload_document/")
# def upload_document(
#         username: str = Form(...),
#         docname: str = Form(...),
#         instruction: str = Form(""),
#         file: UploadFile = File(...),
#         include_summary: bool = Form(False),
#         summary_length: int = Form(4)  # Configurable summary length
# ):
#     """Upload document with optional AI-generated summary"""
#     try:
#         # Create upload directory and save file
#         file_path = os.path.join("uploads", f"{username}_{docname}_{file.filename}")
#         os.makedirs("uploads", exist_ok=True)

#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Process and store document
#         document_service.process_and_store(file_path, username, docname, instruction)

#         response = {
#             "status": "success",
#             "message": "Document uploaded and indexed.",
#             "document_info": {
#                 "filename": file.filename,
#                 "username": username,
#                 "docname": docname
#             }
#         }

#         # Generate enhanced summary if requested
#         if include_summary:
#             try:
#                 # Get document statistics to understand the data type
#                 doc_stats = document_service.get_document_stats(username, docname)

#                 if doc_stats.get('is_structured', False):
#                     # For structured data, provide data overview
#                     ext = os.path.splitext(file_path)[1].lower()
#                     if ext == ".csv":
#                         df = pd.read_csv(file_path)
#                     elif ext in [".xls", ".xlsx"]:
#                         df = pd.read_excel(file_path)
#                     else:
#                         df = None

#                     if df is not None:
#                         preview = df.head(5).to_markdown()
#                         summary_prompt = f"""Analyze this dataset and provide insights about its structure and contents:\n\n{preview}\n\nDataset Info: {df.shape[0]} rows, {df.shape[1]} columns"""
#                         summary = llm_service.invoke(summary_prompt)
#                         response["summary"] = summary
#                         response["data_info"] = {
#                             "rows": df.shape[0],
#                             "columns": df.shape[1],
#                             "column_names": df.columns.tolist()
#                         }
#                 else:
#                     # For unstructured data, use the enhanced summarization
#                     raw_docs = document_service._load_document(file_path)
#                     if raw_docs:
#                         content = raw_docs[0].page_content[:3000]  # Increased content length
#                         summary = llm_service.summarize(content, max_length=summary_length)
#                         response["summary"] = summary

#             except Exception as e:
#                 logger.error(f"Summary generation failed: {str(e)}")
#                 response["summary"] = f"Summary generation failed: {str(e)}"

#         return response

#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
 
# --- Document Upload with Enhanced Summary ---
@router.post("/upload_document/")
def upload_document(
        username: str = Form(...),
        docname: str = Form(...),
        instruction: str = Form(""),
        file: UploadFile = File(...),
        include_summary: bool = Form(False),
        summary_length: int = Form(4)  # Configurable summary length
):
    """Upload document with optional AI-generated summary"""
    try:
        # Create upload directory and save file
        file_path = os.path.join("uploads", f"{username}_{docname}_{file.filename}")
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Process and store document
        document_service.process_and_store(file_path, username, docname, instruction)
        response = {
            "status": "success",
            "message": "Document uploaded and indexed.",
            "document_info": {
                "filename": file.filename,
                "username": username,
                "docname": docname
            }
        }
        # Generate enhanced summary if requested
        if include_summary:
            try:
                # Get document statistics to understand the data type
                doc_stats = document_service.get_document_stats(username, docname)
                if doc_stats.get('is_structured', False):
                    # For structured data, provide data overview
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == ".csv":
                        df = pd.read_csv(file_path)
                    elif ext in [".xls", ".xlsx"]:
                        df = pd.read_excel(file_path)
                    else:
                        df = None
                    if df is not None:
                        preview = df.head(5).to_markdown()
                        summary_prompt = f"""Analyze this dataset and provide insights about its structure and contents:\n\n{preview}\n\nDataset Info: {df.shape[0]} rows, {df.shape[1]} columns"""
                        summary = llm_service.invoke(summary_prompt)
                        response["summary"] = summary
                        response["data_info"] = {
                            "rows": df.shape[0],
                            "columns": df.shape[1],
                            "column_names": df.columns.tolist()
                        }
                else:
                    # For unstructured data, use the enhanced summarization
                    raw_docs = document_service._load_document(file_path)
                    if raw_docs:
                        content = raw_docs[0].page_content[:3000]  # Increased content length
                        summary = llm_service.summarize(content, max_length=summary_length)
                        response["summary"] = summary
            except Exception as e:
                logger.error(f"Summary generation failed: {str(e)}")
                response["summary"] = f"Summary generation failed: {str(e)}"
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
 
 

# --- Conversational QA Endpoint ---
@router.post("/conversational_qa/")
def conversational_qa(
    username: str = Form(...),
    session_id: str = Form(...),
    query: str = Form(...),
    docnames: List[str] = Form(...),
    feedback_username: Optional[str] = Form(None),
    feedback_comment: Optional[str] = Form(None)
):
    """Chat-like multi-turn QA across documents"""
    try:
        feedback_data = None
        if feedback_username and feedback_comment:
            feedback_data = {
                "username": feedback_username,
                "feedback": feedback_comment
            }

        result = llm_service.ask_documents_conversational(
            query=query,
            username=username,
            session_id=session_id,
            docnames=docnames,
            feedback_data=feedback_data
        )

        # Store session history after getting result
        db_service.save_session(
            session_id=session_id,
            username=username,
            query=query,
            docname=docnames,
            response=result["answer"]
        )

        print(result)

        return {
            "session_id": session_id,
            "query": query,
            "answer": result["answer"],
            "query_type": result.get("query_type"),
            "context_length": result.get("context_length"),
            "source_documents": result.get("source_documents", [])
        }

    except Exception as e:
        logger.error(f"Conversational QA error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversational QA failed: {str(e)}")

# --- Enhanced Document Summarization ---
@router.post("/summarize_document/")
def summarize_document(
        username: str = Form(...),
        docname: str = Form(...),
        summary_length: int = Form(4),
        summary_type: str = Form("general")  # general, technical, executive
):
    """Generate enhanced document summary with configurable options"""
    try:
        # Get document statistics first
        doc_stats = document_service.get_document_stats(username, docname)

        # Load retriever and fetch relevant chunks
        retriever = document_service.get_retriever(username, docname)
        docs = retriever.get_relevant_documents("provide a comprehensive summary of this document")

        if not docs:
            # Fallback to raw document
            upload_dir = "uploads"
            matched_files = [f for f in os.listdir(upload_dir) if f.startswith(f"{username}_{docname}")]
            if not matched_files:
                raise HTTPException(status_code=404, detail="Document not found.")

            file_path = os.path.join(upload_dir, matched_files[0])
            raw_docs = document_service._load_document(file_path)
            docs = raw_docs[:3] if raw_docs else []

        if not docs:
            raise HTTPException(status_code=404, detail="No content found in document.")

        # Combine content from multiple chunks
        combined_content = "\n\n".join([doc.page_content for doc in docs[:5]])

        # Customize summary based on type
        if summary_type == "executive":
            summary_prompt = f"""Create an executive summary of the following content focusing on key insights, conclusions, and actionable items:\n\n{combined_content}"""
        elif summary_type == "technical":
            summary_prompt = f"""Create a technical summary highlighting methodologies, data, and technical details:\n\n{combined_content}"""
        else:
            summary_prompt = combined_content

        # Generate summary with specified length
        summary = llm_service.summarize(summary_prompt, max_length=summary_length)

        return {
            "document": docname,
            "summary": summary,
            "summary_type": summary_type,
            "summary_length": summary_length,
            "chunks_used": len(docs[:5]),
            "is_structured": doc_stats.get('is_structured', False)
        }

    except Exception as e:
        logger.error(f"Document summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# --- Enhanced Text Summarization ---
@router.post("/summarize_text/")
def summarize_text(
        text: str = Form(...),
        max_length: int = Form(4),
        summary_type: str = Form("general")
):
    """Enhanced text summarization with configurable options"""
    try:
        # Customize based on summary type
        if summary_type == "executive":
            modified_text = f"Create an executive summary focusing on key points and actionable insights:\n\n{text}"
        elif summary_type == "technical":
            modified_text = f"Create a technical summary highlighting methodologies and technical details:\n\n{text}"
        else:
            modified_text = text

        summary = llm_service.summarize(modified_text, max_length=max_length)

        return {
            "original_length": len(text),
            "summary": summary,
            "summary_type": summary_type,
            "compression_ratio": len(summary) / len(text) if text else 0
        }

    except Exception as e:
        logger.error(f"Text summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {str(e)}")


# --- Enhanced SQL Explanation ---
@router.post("/explain_sql/")
def explain_sql_query(
        sql_query: str = Form(...),
        detail_level: str = Form("standard")  # basic, standard, detailed
):
    """Enhanced SQL query explanation with configurable detail levels"""
    try:
        # Customize explanation based on detail level
        if detail_level == "basic":
            custom_prompt = f"""Provide a simple, non-technical explanation of this SQL query:\n\n{sql_query}"""
        elif detail_level == "detailed":
            custom_prompt = f"""Provide a comprehensive, technical explanation of this SQL query including performance considerations:\n\n{sql_query}"""
        else:
            custom_prompt = sql_query

        explanation = llm_service.explain_sql(custom_prompt if detail_level != "standard" else sql_query)

        return {
            "query": sql_query,
            "explanation": explanation,
            "detail_level": detail_level
        }

    except Exception as e:
        logger.error(f"SQL explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL explanation failed: {str(e)}")


# --- NEW: Answer Refinement Endpoint ---
@router.post("/refine_answer/")
def refine_answer(
        question: str = Form(...),
        initial_answer: str = Form(...),
        additional_context: Optional[str] = Form(None),
        username: Optional[str] = Form(None)
):
    """Refine and improve an existing answer"""
    try:
        feedback_data = {"username": username} if username else None

        refined_answer = llm_service.refine_answer(
            question=question,
            initial_answer=initial_answer,
            context=additional_context,
            feedback_data=feedback_data
        )

        return {
            "question": question,
            "initial_answer": initial_answer,
            "refined_answer": refined_answer,
            "improvement_applied": True
        }

    except Exception as e:
        logger.error(f"Answer refinement error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer refinement failed: {str(e)}")


# --- NEW: Health Check Endpoint ---
@router.get("/health/")
def health_check():
    """Comprehensive health check for all services"""
    try:
        # Get LLM service health
        llm_health = llm_service.health_check()

        # Check other services
        doc_health = "healthy" if document_service else "unhealthy"
        db_health = "healthy" if db_service else "unhealthy"
        cache_health = "healthy" if cache_manager else "unhealthy"

        overall_status = "healthy" if all([
            llm_health["llm_service"] == "healthy",
            doc_health == "healthy",
            db_health == "healthy"
        ]) else "unhealthy"

        return {
            "overall_status": overall_status,
            "llm_service": llm_health,
            "document_service": doc_health,
            "database_service": db_health,
            "cache_service": cache_health,
            "timestamp": llm_health["timestamp"]
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "overall_status": "unhealthy",
            "error": str(e)
        }


# --- Enhanced Query Assistant ---
# @router.post("/query_assistant/")
# def query_assistant(
#         user_request: str = Form(...),
#         context_type: str = Form("general")  # general, sql, document, data
# ):
#     """Enhanced intelligent assistant with context awareness"""
#     try:
#         # Enhanced SQL detection
#         sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'alter', 'drop', 'with', 'from', 'where',
#                         'join']
#         is_sql = any(keyword in user_request.lower() for keyword in sql_keywords)

#         if is_sql or context_type == "sql":
#             explanation = llm_service.explain_sql(user_request)
#             return {
#                 "type": "sql_explanation",
#                 "query": user_request,
#                 "explanation": explanation,
#                 "context_type": context_type
#             }
#         else:
#             # Use context-aware prompting
#             if context_type == "document":
#                 prompt = f"As a document analysis assistant, help with this request: {user_request}"
#             elif context_type == "data":
#                 prompt = f"As a data analysis assistant, help with this request: {user_request}"
#             else:
#                 prompt = f"As a helpful assistant, provide guidance for: {user_request}"

#             response = llm_service.invoke(prompt)
#             return {
#                 "type": "assistance",
#                 "response": response,
#                 "context_type": context_type
#             }

#     except Exception as e:
#         logger.error(f"Query assistant error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Query assistant failed: {str(e)}")


# --- All existing endpoints remain unchanged ---
@router.get("/list_documents/")
def list_documents():
    """List all available documents"""
    try:
        return document_service.list_documents()
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/delete_document/{docname}")
def delete_document( docname: str):
    """Delete a specific document"""
    try:
        success = document_service.delete_document_vector(docname)
        if success:
            return {"status": "success", "message": f"Deleted document: {docname}"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/delete_all_documents/")
def delete_all_documents():
    """Delete all documents"""
    try:
        document_service.delete_all_documents()
        return {"status": "success", "message": "All documents deleted."}
    except Exception as e:
        logger.error(f"Delete all documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all documents: {str(e)}")


@router.get("/session_history/{username}")
def view_all_sessions(username: str):
    """Get session history"""
    try:
        response = db_service.get_chat_history(username=username)
        print(response)
        return response

    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

# @router.get("/session_by_id/")
# def view_session_by_id(session_id: str):
#     """Get session history"""
#     try:
#         return document_service.view_session_by_id(session_id=session_id)
#     except Exception as e:
#         logger.error(f"Get sessions error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@router.get("/session_by_id/{session_id}")
def get_session(session_id: str):
    session = db_service.get_session(session_id)
    if not session:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
    return session
@router.delete("/delete_session/{session_id}")
def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        db_service.delete_session_by_session_id(session_id)
        return {"status": "success", "message": "Session deleted."}
    except Exception as e:
        logger.error(f"Delete session error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.delete("/delete_all_sessions/")
def delete_all_sessions(username: Optional[str] = None):
    """Delete all sessions"""
    try:
        db_service.delete_all_sessions(username)
        return {"status": "success", "message": "Sessions deleted."}
    except Exception as e:
        logger.error(f"Delete all sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete sessions: {str(e)}")


@router.post("/submit_feedback/")
def submit_feedback(
        username: str = Form(...),
        prompt: str = Form(...),
        response: str = Form(...),
        feedback: str = Form(...)
):
    """Submit user feedback"""
    try:
        db_service.store_feedback(username, prompt, response, feedback)
        logger.info(f"Feedback stored: {username}, {prompt}, {feedback}")
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        logger.error(f"Submit feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/feedback_stats/")
def feedback_stats():
    """Get feedback statistics"""
    try:
        return db_service.get_feedback_stats()
    except Exception as e:
        logger.error(f"Feedback stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")


@router.delete("/clear_cache/")
def clear_cache():
    """Clear the cache"""
    try:
        cache_manager.clear()
        return {"status": "success", "message": "Cache cleared."}
    except Exception as e:
        logger.error(f"Clear cache error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/export_sessions/{username}")
def export_sessions(
    username: str,
    format: str = Query("csv", enum=["csv", "json"])
):
    """Export session history in CSV or JSON format"""
    try:
        sessions = document_service.view_all_sessions(username=username)

        if not sessions:
            raise HTTPException(status_code=404, detail="No sessions found.")

        if format == "json":
            return sessions

        # Convert to CSV
        import io
        import csv

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=sessions[0].keys())
        writer.writeheader()
        writer.writerows(sessions)
        output.seek(0)

        return Response(
            content=output.read(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sessions.csv"}
        )

    except Exception as e:
        logger.error(f"Export session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export sessions.")

@router.get("/chat_history/{session_id}")
def chat_history(session_id: int):
    """Get full chat history of a session"""
    try:
        history = document_service.view_session_by_id(session_id=session_id)
        if not history:
            raise HTTPException(status_code=404, detail="No chat found for this session.")

        # Optionally, enrich with source doc info or LLM annotations
        enriched_history = []
        for row in history:
            enriched = {
                "timestamp": row["timestamp"],
                "prompt": row["prompt"],
                "response": row["response"],
                "document_used": row.get("docname"),
                "source_documents": row.get("source_documents", []),
            }
            enriched_history.append(enriched)

        return enriched_history

    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")
    
@router.get("/structured_documents/")

def get_structured_documents():
    """Fetch all structured documents and their column names for the Data Visualizer"""
    try:
        # Get list of all documents for the user
        documents = document_service.list_documents()
        
        # Filter for structured documents (CSV, Excel)
        structured_docs = [
            doc for doc in documents 
            if doc.get('is_structured', False)
        ]
        
        result = []
        for doc in structured_docs:
            file_path = doc.get('file_path')
            if not file_path or not os.path.exists(file_path):
                continue
                
            try:
                # Load the file to get column names
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                    
                result.append({
                    "docname": doc['docname'],
                    "filename": os.path.basename(file_path),
                    "columns": df.columns.tolist()
                })
            except Exception as e:
                logger.warning(f"Error loading columns for {doc['docname']}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "documents": result
        }
        
    except Exception as e:
        logger.error(f"Error fetching structured documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@router.post("/generate_from_file/", summary="Generate chart from existing file")
def generate_from_file(
    filename: str = Form(...),
    x_col: str = Form(...),
    y_col: str = Form(...),
    chart_type: str = Form(...)
):
    # Find the structured document with matching filename
    documents = document_service.list_documents()

    doc = next(
        (d for d in documents if d.get("docname")== filename),
        None
    )
    print(doc)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No structured document found for filename '{filename}'")

    file_path = doc["file_path"]
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    # Load into DataFrame
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Generate Plotly chart
    chart = document_service.generate_chart(df, chart_type, x_col, y_col)
    from plotly.utils import PlotlyJSONEncoder
    return JSONResponse(content=json.loads(json.dumps(chart, cls=PlotlyJSONEncoder)))
