# --- app/services/database_service.py --- (FIXED VERSION)
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatabaseService:
    def __init__(self, db_path: str = "rag_app.db"):
        # Connect to the SQLite database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-style row access
        self._create_tables()

    def _create_tables(self):
        # Create session history table with session_id
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                username TEXT NOT NULL,
                query TEXT NOT NULL,
                document_name TEXT,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        # Create feedback table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_session(self, session_id: str, username: str, query: str, docname: str, response: str):
        """Store a query session into the database"""
        try:
            # Convert docname to string if it's a list
            if isinstance(docname, list):
                docname = json.dumps(docname)

            self.conn.execute("""
                INSERT INTO session_history (session_id, username, query, document_name, response, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, username, query, docname, response, datetime.now().isoformat()))
            self.conn.commit()
            logger.info(f"Session saved successfully: {session_id}")
        except Exception as e:
            logger.error(f"Failed to save session: {str(e)}")
            raise e

    def get_chat_history(self, username: str) -> List[Dict[str, Any]]:
        """
        Fetch all prompts/responses for a given username.
        """
        try:
            query = """
                SELECT id, session_id, query, response, document_name, timestamp
                FROM session_history
                WHERE username = ?
                ORDER BY timestamp DESC
            """
            cursor = self.conn.execute(query, (username,))
            rows = cursor.fetchall()

            # Convert rows to dictionaries
            result = []
            for row in rows:
                row_dict = {
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'query': row['query'],
                    'response': row['response'],
                    'document_name': row['document_name'],
                    'timestamp': row['timestamp']
                }
                result.append(row_dict)

            logger.info(f"Retrieved {len(result)} chat history records for user: {username}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch chat history: {str(e)}")
            raise e

    def get_session_by_id(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all messages for a specific session_id.
        """
        try:
            query = """
                SELECT id, session_id, username, query, response, document_name, timestamp
                FROM session_history
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            cursor = self.conn.execute(query, (session_id,))
            rows = cursor.fetchall()

            # Convert rows to dictionaries
            result = []
            for row in rows:
                row_dict = {
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'username': row['username'],
                    'query': row['query'],
                    'response': row['response'],
                    'document_name': row['document_name'],
                    'timestamp': row['timestamp']
                }
                result.append(row_dict)

            logger.info(f"Retrieved {len(result)} messages for session: {session_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch session by ID: {str(e)}")
            raise e

    def delete_session(self, session_id: int):
        """Delete a specific session record by ID"""
        try:
            cursor = self.conn.execute("DELETE FROM session_history WHERE id = ?", (session_id,))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Session deleted successfully: {session_id}")
            else:
                logger.warning(f"No session found with ID: {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            raise e
    def get_session(self, session_id):
            cursor = self.conn.execute(
                  "SELECT * FROM session_history WHERE session_id = ?", (session_id,)
                        )
            return cursor.fetchone() 

    def delete_session_by_session_id(self, session_id: str):
        """Delete all session records by session_id"""
        try:
            cursor = self.conn.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Session deleted successfully: {session_id}")
            else:
                logger.warning(f"No session found with session_id: {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            raise e

    def delete_all_sessions(self, username: str = None):
        """Delete all sessions or those for a specific user"""
        try:
            if username:
                cursor = self.conn.execute("DELETE FROM session_history WHERE username = ?", (username,))
                logger.info(f"Deleted {cursor.rowcount} sessions for user: {username}")
            else:
                cursor = self.conn.execute("DELETE FROM session_history")
                logger.info(f"Deleted {cursor.rowcount} sessions")
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete sessions: {str(e)}")
            raise e

    def store_feedback(self, username: str, prompt: str, response: str, feedback: str):
        """Save feedback entry linked to prompt/response"""
        try:
            # Ensure all inputs are strings and not empty
            if not all([username, prompt, response, feedback]):
                raise ValueError("All feedback fields must be non-empty strings")
            
            self.conn.execute("""
                INSERT INTO feedback (username, prompt, response, feedback, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, prompt, response, feedback, datetime.now().isoformat()))
            self.conn.commit()
            logger.info(f"Feedback stored for user: {username}, prompt: {prompt[:50]}...")
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            raise e

    def get_feedback_by_user(self, username: str) -> List[Dict[str, Any]]:
        """Retrieve feedback for a specific user"""
        try:
            result = self.conn.execute("SELECT * FROM feedback WHERE username = ?", (username,)).fetchall()
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get feedback by user: {str(e)}")
            raise e

    def get_feedback_stats(self) -> List[Dict[str, Any]]:
        """Get count of feedback grouped by type"""
        try:
            result = self.conn.execute("""
                SELECT feedback, COUNT(*) AS count
                FROM feedback
                GROUP BY feedback
            """).fetchall()
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {str(e)}")
            raise e

    def close(self):
        """Close database connection"""
        try:
            self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    def getpreview(self, table_name: str):
        return self.conn.execute(f"SELECT * FROM {table_name} LIMIT 2;").fetchall()

    

    