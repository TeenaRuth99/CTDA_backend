# --- app/services/llm_service.py ---
import os
import hashlib
import json
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import Input, Output
from app.services.cachemanager import CacheManager
from app.services.database import DatabaseService
from app.services.document_service import DocumentService
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self,
                 model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                 cache_enabled: bool = True,
                 document_service: Optional[DocumentService] = None):
        """
        Initialize LLM Service with enhanced RAG capabilities

        Args:
            model_name: Groq model name
            cache_enabled: Whether to enable caching
            document_service: DocumentService instance for RAG operations
        """
        # Load Groq API key from environment
        self.api_key = os.getenv("GROQ_API_KEY")

        # Initialize Groq LLM with enhanced parameters
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            max_tokens=1024,  # Increased for better responses
            temperature=0.7,
            timeout=60,
            max_retries=3
        )

        # Initialize services
        self.cache_enabled = cache_enabled
        self.cache = CacheManager() if cache_enabled else None
        self.db = DatabaseService()
        self.document_service = document_service or DocumentService()

        # Output parser
        self.parser = StrOutputParser()

        # Enhanced prompt templates registry
        self.prompt_templates = self._initialize_prompt_templates()

        # RAG configuration
        self.rag_config = {
            "structured_k": 5,  # Number of chunks for structured data
            "unstructured_k": 4,  # Number of chunks for unstructured data
            "similarity_threshold": 0.7,
            "max_context_length": 8000,
            "chunk_overlap_strategy": "smart"
        }

        # Conversation memory storage
        self.conversation_memory = {}

    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive prompt templates"""
        return {
            "conversational_qa": {
                "template": """You are a helpful AI assistant engaged in a conversation with a user. You have access to document context and conversation history.

Previous Conversation:
{conversation_history}

Current Document Context:
{context}

Current Question: {question}

Instructions:
- Consider the conversation history to maintain context and continuity
- Use the document context to provide accurate, relevant answers
- If the question relates to previous conversation, acknowledge and build upon it
- If the document context doesn't contain relevant information, say so clearly
- Provide helpful, conversational responses that feel natural
- Reference specific parts of the documents when relevant
- If the user asks follow-up questions, connect them to previous responses

Answer:""",
                "input_variables": ["conversation_history", "context", "question"]
            },

            "structured_qa": {
                "template": """You are a data analyst assistant. Answer the question based on the structured data provided.

Data Context:
{context}

Question: {question}

Instructions:
- Use only the data provided in the context
- If the answer requires calculations, show your work
- If data is missing, clearly state what information is unavailable
- Format numerical answers appropriately
- If the question cannot be answered with the given data, say "Information not available in the provided data"

Answer:""",
                "input_variables": ["context", "question"]
            },

            "unstructured_qa": {
                "template": """You are a knowledgeable research assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the provided context
- If the context doesn't contain relevant information, say "Information not available in the provided context"
- Provide specific details when available
- If multiple perspectives are presented, acknowledge them
- Cite specific parts of the context when relevant

Answer:""",
                "input_variables": ["context", "question"]
            },

            "hybrid_qa": {
                "template": """You are an intelligent assistant capable of working with both structured and unstructured data.

Context Information:
{context}

Question: {question}

Instructions:
- Analyze both structured data (tables, statistics) and unstructured text
- Provide a comprehensive answer that integrates information from all sources
- Distinguish between quantitative insights (from structured data) and qualitative insights (from text)
- If information is contradictory, mention the discrepancy
- If some information is missing, specify what additional data would be helpful

Answer:""",
                "input_variables": ["context", "question"]
            },

            "summarization": {
                "template": """You are a skilled summarization assistant. Create a concise summary of the provided content.

Content to summarize:
{content}

Requirements:
- Provide a clear, concise summary in 3-4 sentences
- Capture the main points and key insights
- Maintain the original meaning and context
- Use clear, professional language

Summary:""",
                "input_variables": ["content"]
            },

            "sql_explanation": {
                "template": """You are a database expert. Explain the following SQL query in simple, clear terms.

SQL Query:
{sql_query}

Explanation Requirements:
- Break down the query into logical components
- Explain what each part does
- Describe the expected output
- Mention any important considerations or potential issues
- Use non-technical language when possible

Explanation:""",
                "input_variables": ["sql_query"]
            },

            "answer_refinement": {
                "template": """You are a quality assurance assistant. Improve the given answer based on the question and additional context.

Original Question: {question}
Initial Answer: {initial_answer}
Additional Context: {context}

Your task:
- Enhance the answer's accuracy and completeness
- Incorporate relevant information from the additional context
- Ensure the answer directly addresses the question
- Maintain clarity and readability
- If the initial answer was incorrect, provide the correct information

Refined Answer:""",
                "input_variables": ["question", "initial_answer", "context"]
            }
        }

    def _generate_cache_key(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """Generate unique cache key including parameters"""
        cache_data = {"prompt": prompt, "params": params or {}}
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    def _format_context(self, documents: List[Document], max_length: int = 8000) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents):
            # Add document separator and metadata info
            doc_info = f"Document {i + 1}"
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'content')
                doc_info += f" (Source: {source}, Type: {doc_type})"

            doc_content = f"\n--- {doc_info} ---\n{doc.page_content}\n"

            # Check if adding this document would exceed max length
            if current_length + len(doc_content) > max_length:
                if context_parts:  # Only break if we have at least one document
                    break
                else:  # If first document is too long, truncate it
                    doc_content = doc_content[:max_length - 100] + "...\n"

            context_parts.append(doc_content)
            current_length += len(doc_content)

        return "\n".join(context_parts)

    def _determine_query_type(self, query: str, context: str = "") -> str:
        """Determine the type of query and context for better prompt selection"""
        query_lower = query.lower()

        # Check for structured data indicators
        structured_indicators = ["calculate", "sum", "average", "count", "total", "statistics",
                                 "table", "column", "row", "data analysis", "compare numbers"]

        # Check for unstructured data indicators
        unstructured_indicators = ["explain", "describe", "what is", "tell me about",
                                   "summarize", "analysis", "opinion", "concept"]

        # Check context for structured data patterns
        has_structured_context = any(indicator in context for indicator in ["|", "Row ", "Column ", "Total Rows:"])

        if has_structured_context or any(indicator in query_lower for indicator in structured_indicators):
            return "structured"
        elif any(indicator in query_lower for indicator in unstructured_indicators):
            return "unstructured"
        else:
            return "hybrid"

    def _format_conversation_history(self, session_id: str, limit: int = 3) -> str:
        """Format conversation history for context"""
        try:
            # Get conversation history from database
            conversations = self.db.get_conversation_history(session_id, limit)

            if not conversations:
                return "No previous conversation."

            history_parts = []
            for i, conv in enumerate(conversations):
                history_parts.append(f"Q{i + 1}: {conv.get('question', '')}")
                history_parts.append(f"A{i + 1}: {conv.get('answer', '')}")

            return "\n".join(history_parts)

        except Exception as e:
            logger.error(f"Error formatting conversation history: {str(e)}")
            return "Unable to retrieve conversation history."


    def register_prompt_template(self, name: str, template: str, input_variables: List[str]):
        """Register a new prompt template"""
        self.prompt_templates[name] = {
            "template": template,
            "input_variables": input_variables
        }

    def get_prompt_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered prompt template"""
        return self.prompt_templates.get(name)

    def invoke(self,
               prompt: str,
               stream: bool = False,
               feedback_data: Optional[Dict[str, str]] = None,
               cache_params: Optional[Dict[str, Any]] = None) -> Union[str, Any]:
        """
        Enhanced invoke method with better caching and error handling
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, cache_params)

            # Check cache
            if self.cache_enabled and self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info("Retrieved response from cache")
                    return cached_result

            # Execute LLM call
            if stream:
                return self.llm.stream(prompt)

            response = self.llm.invoke(prompt)
            result = self.parser.parse(response.content.strip())

            # Cache result
            if self.cache_enabled and self.cache:
                self.cache.set(cache_key, result)

            # Store feedback if provided
            if feedback_data:
                self.db.store_feedback(
                    username=feedback_data.get("username", "unknown"),
                    prompt=prompt,
                    response=result,
                    feedback=feedback_data.get("feedback", "")
                )

            return result

        except Exception as e:
            logger.error(f"Error in LLM invoke: {str(e)}")
            raise

    def ask_documents_conversational(self,
                                     query: str,
                                     username: str,
                                     session_id: str,
                                     docnames: List[str],
                                     feedback_data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Enhanced conversational QA across multiple documents without storing the conversation
        """
        try:
            # Get conversation history
            conversation_history = self._format_conversation_history(session_id)

            # Collect documents from all sources
            all_relevant_docs = []
            doc_types = []

            for docname in docnames:
                try:
                    doc_stats = self.document_service.get_document_stats(username, docname)
                    doc_types.append(doc_stats.get('is_structured', False))

                    retriever = self.document_service.get_retriever(username, docname)
                    docs = retriever.get_relevant_documents(query)

                    for doc in docs:
                        doc.metadata['source_document'] = docname

                    all_relevant_docs.extend(docs)

                except Exception as e:
                    logger.warning(f"Error querying document {docname}: {str(e)}")
                    continue

            if not all_relevant_docs:
                return {
                    "answer": "No relevant information found in any of the specified documents.",
                    "source_documents": [],
                    "query_type": "no_results",
                    "session_id": session_id
                }

            all_relevant_docs.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            top_docs = all_relevant_docs[:self.rag_config["structured_k"] + self.rag_config["unstructured_k"]]
            context = self._format_context(top_docs, self.rag_config["max_context_length"])
            query_type = self._determine_query_type(query, context)

            prompt_config = self.prompt_templates["conversational_qa"]
            prompt_template = PromptTemplate(
                template=prompt_config["template"],
                input_variables=prompt_config["input_variables"]
            )

            formatted_prompt = prompt_template.format(
                conversation_history=conversation_history,
                context=context,
                question=query
            )

            answer = self.invoke(
                formatted_prompt,
                feedback_data=feedback_data,
                cache_params={"username": username, "session_id": session_id, "docnames": docnames}
            )

            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, 'score', None)
                    }
                    for doc in top_docs
                ],
                "query_type": query_type,
                "document_types": doc_types,
                "context_length": len(context),
                "documents_queried": len(docnames),
                "session_id": session_id,
                "conversation_turn": len(conversation_history.split(
                    '\n')) // 2 + 1 if conversation_history != "No previous conversation." else 1
            }

        except Exception as e:
            logger.error(f"Error in conversational document query: {str(e)}")
            return {
                "answer": f"Error processing conversational query: {str(e)}",
                "source_documents": [],
                "query_type": "error",
                "session_id": session_id
            }

    def summarize(self, text: str, max_length: int = 4) -> str:
        """Enhanced summarization with configurable length"""
        try:
            prompt_config = self.prompt_templates["summarization"]
            prompt_template = PromptTemplate(
                template=prompt_config["template"].replace("3-4 sentences", f"{max_length} sentences"),
                input_variables=prompt_config["input_variables"]
            )

            formatted_prompt = prompt_template.format(content=text)
            return self.invoke(formatted_prompt)

        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def explain_sql(self, sql_query: str) -> str:
        """Enhanced SQL explanation"""
        try:
            prompt_config = self.prompt_templates["sql_explanation"]
            prompt_template = PromptTemplate(
                template=prompt_config["template"],
                input_variables=prompt_config["input_variables"]
            )

            formatted_prompt = prompt_template.format(sql_query=sql_query)
            return self.invoke(formatted_prompt)

        except Exception as e:
            logger.error(f"Error in SQL explanation: {str(e)}")
            return f"Error explaining SQL: {str(e)}"

    def refine_answer(self,
                      question: str,
                      initial_answer: str,
                      context: Optional[str] = None,
                      feedback_data: Optional[Dict[str, str]] = None) -> str:
        """Enhanced answer refinement"""
        try:
            prompt_config = self.prompt_templates["answer_refinement"]
            prompt_template = PromptTemplate(
                template=prompt_config["template"],
                input_variables=prompt_config["input_variables"]
            )

            formatted_prompt = prompt_template.format(
                question=question,
                initial_answer=initial_answer,
                context=context or "No additional context provided"
            )

            return self.invoke(formatted_prompt, feedback_data=feedback_data)

        except Exception as e:
            logger.error(f"Error in answer refinement: {str(e)}")
            return f"Error refining answer: {str(e)}"

    def generate_prompt(self, query: str, documents: List[Document]) -> str:
        """Generate context-aware prompt from documents"""
        try:
            context = self._format_context(documents)
            query_type = self._determine_query_type(query, context)

            # Determine if we have structured data
            has_structured = any(
                doc.metadata.get('is_structured', False) for doc in documents
            )

            if has_structured and query_type in ["structured", "hybrid"]:
                template_name = "structured_qa"
            elif query_type == "unstructured":
                template_name = "unstructured_qa"
            else:
                template_name = "hybrid_qa"

            prompt_config = self.prompt_templates[template_name]
            prompt_template = PromptTemplate(
                template=prompt_config["template"],
                input_variables=prompt_config["input_variables"]
            )

            return prompt_template.format(context=context, question=query)

        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return f"Error generating prompt: {str(e)}"

    def rerank_answers(self, query: str, answers: List[str]) -> str:
        """Enhanced answer reranking"""
        try:
            if not answers:
                return "No answers provided for reranking."

            if len(answers) == 1:
                return answers[0]

            context = "\n\n".join([
                f"Answer {i + 1}: {answer}"
                for i, answer in enumerate(answers)
            ])

            prompt = f"""You are an expert answer evaluator. Given a question and multiple answers, select and return the most accurate, complete, and helpful answer.

Question: {query}

Available Answers:
{context}

Instructions:
- Evaluate each answer for accuracy, completeness, and relevance
- Consider factual correctness and clarity
- Choose the answer that best addresses the question
- Return only the selected answer, not your reasoning

Best Answer:"""

            return self.invoke(prompt)

        except Exception as e:
            logger.error(f"Error in answer reranking: {str(e)}")
            return answers[0] if answers else "Error in reranking process."


    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a conversation session"""
        try:
            conversations = self.db.get_chat_history(session_id)
            print(conversations)

            if not conversations:
                return {
                    "session_id": session_id,
                    "total_turns": 0,
                    "summary": "No conversation history found."
                }

            # Create summary
            all_questions = [conv.get('question', '') for conv in conversations]
            all_answers = [conv.get('answer', '') for conv in conversations]

            summary_text = f"Session had {len(conversations)} conversation turns. "
            summary_text += f"Topics discussed: {', '.join(all_questions[:3])}"
            if len(all_questions) > 3:
                summary_text += f" and {len(all_questions) - 3} more topics."

            return {
                "session_id": session_id,
                "total_turns": len(conversations),
                "summary": summary_text,
                "latest_timestamp": conversations[0].get('timestamp', '') if conversations else None
            }

        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {
                "session_id": session_id,
                "total_turns": 0,
                "summary": f"Error retrieving session summary: {str(e)}"
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service"""
        try:
            # Test LLM connection
            test_response = self.invoke("Hello, this is a test.")
            llm_status = "healthy" if test_response else "unhealthy"

            # Test document service
            doc_status = "healthy" if self.document_service else "unhealthy"

            # Test cache
            cache_status = "healthy" if not self.cache_enabled or self.cache else "unhealthy"

            return {
                "llm_service": llm_status,
                "document_service": doc_status,
                "cache_service": cache_status,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "llm_service": "unhealthy",
                "document_service": "unhealthy",
                "cache_service": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }