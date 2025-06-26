"""
Database Manager for ALEX - SQLite with async support
"""

import aiosqlite
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.schemas import ConversationHistory, MessageRole, ConversationSummary

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages conversation history and app data"""
    
    def __init__(self, database_url: str):
        # Extract database path from URL
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove "sqlite:///"
        else:
            self.db_path = database_url
        
        self.db = None
        
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            # Create database and tables
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables"""
        
        # Conversations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_used TEXT,
                tokens_used INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
        """)
        
        # Model stats table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_time REAL DEFAULT 0.0,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
    
    async def is_connected(self) -> bool:
        """Check if database is accessible"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def save_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None
    ):
        """Save a message to the database"""
        
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure conversation exists
            await db.execute("""
                INSERT OR IGNORE INTO conversations (conversation_id, created_at, updated_at)
                VALUES (?, ?, ?)
            """, (conversation_id, datetime.utcnow(), datetime.utcnow()))
            
            # Save message
            await db.execute("""
                INSERT INTO messages (conversation_id, role, content, model_used, tokens_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (conversation_id, role, content, model_used, tokens_used, datetime.utcnow()))
            
            # Update conversation timestamp
            await db.execute("""
                UPDATE conversations 
                SET updated_at = ? 
                WHERE conversation_id = ?
            """, (datetime.utcnow(), conversation_id))
            
            await db.commit()
    
    async def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[ConversationHistory]:
        """Get conversation history"""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT id, conversation_id, role, content, model_used, tokens_used, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (conversation_id, limit)) as cursor:
                
                rows = await cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append(ConversationHistory(
                        id=row['id'],
                        conversation_id=row['conversation_id'],
                        role=MessageRole(row['role']),
                        content=row['content'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        model_used=row['model_used'],
                        tokens_used=row['tokens_used']
                    ))
                
                return history
    
    async def list_conversations(self, limit: int = 20) -> List[ConversationSummary]:
        """List recent conversations with summaries"""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute("""
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.updated_at,
                    COUNT(m.id) as message_count,
                    m.content as last_message
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                LEFT JOIN (
                    SELECT conversation_id, MAX(timestamp) as max_timestamp
                    FROM messages
                    GROUP BY conversation_id
                ) latest ON c.conversation_id = latest.conversation_id AND m.timestamp = latest.max_timestamp
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (limit,)) as cursor:
                
                rows = await cursor.fetchall()
                
                conversations = []
                for row in rows:
                    conversations.append(ConversationSummary(
                        conversation_id=row['conversation_id'],
                        title=row['title'],
                        last_message=row['last_message'] or "No messages",
                        last_updated=datetime.fromisoformat(row['updated_at']),
                        message_count=row['message_count']
                    ))
                
                return conversations
    
    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages"""
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            await db.commit()
    
    async def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE conversations 
                SET title = ?, updated_at = ?
                WHERE conversation_id = ?
            """, (title, datetime.utcnow(), conversation_id))
            await db.commit()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get conversation count
            async with db.execute("SELECT COUNT(*) as count FROM conversations") as cursor:
                conv_count = (await cursor.fetchone())['count']
            
            # Get message count
            async with db.execute("SELECT COUNT(*) as count FROM messages") as cursor:
                msg_count = (await cursor.fetchone())['count']
            
            # Get total tokens used
            async with db.execute("SELECT SUM(tokens_used) as total FROM messages WHERE tokens_used IS NOT NULL") as cursor:
                total_tokens = (await cursor.fetchone())['total'] or 0
            
            return {
                "total_conversations": conv_count,
                "total_messages": msg_count,
                "total_tokens_used": total_tokens
            }
    
    async def close(self):
        """Close database connection"""
        # aiosqlite handles connection cleanup automatically
        logger.info("Database manager closed")