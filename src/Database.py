import psycopg2
import json
from datetime import datetime
from Logger import Logger, LogComponent
import os


class Database:
    def __init__(self):
        self.logger = Logger()
        self.connection = None
        self.connect()
        self.create_tables()

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname="dialai",
                user=os.getenv('USER'),  # Uses your system username
                password="",             # Postgres.app doesn't need password by default
                host="localhost",
                port="5432"
            )
            self.logger.info(LogComponent.DATABASE, "Successfully connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error connecting to database: {e}")
            raise e

    def create_tables(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations_history (
                        id SERIAL PRIMARY KEY,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        duration_seconds INTEGER NOT NULL,
                        conversation_history JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.connection.commit()
                self.logger.info(LogComponent.DATABASE, "Database tables created successfully")
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error creating tables: {e}")
            raise e

    def store_conversation(self, start_time, end_time, conversation_history):
        try:
            total_duration = int((end_time - start_time).total_seconds())
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO conversations_history 
                    (start_time, end_time, duration_seconds, conversation_history)
                    VALUES (%s, %s, %s, %s)
                """, (
                    start_time,
                    end_time,
                    total_duration,
                    json.dumps(conversation_history)
                ))
                self.connection.commit()
                self.logger.info(LogComponent.DATABASE, "Conversation stored successfully")
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error storing conversation: {e}")
            raise e

    def close(self):
        if self.connection:
            self.connection.close()

    def get_conversation_by_id(self, conversation_id: int):
        """Retrieve a specific conversation from the database by ID."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, start_time, end_time, duration_seconds, conversation_history
                    FROM conversations_history
                    WHERE id = %s
                """, (conversation_id,))
                results = cursor.fetchall()
                return results[0] if results else None
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error retrieving conversation {conversation_id}: {e}")
            raise e

    def get_all_conversations(self):
        """Retrieve all conversations from the database."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, start_time, end_time, duration_seconds, conversation_history
                    FROM conversations_history
                    ORDER BY start_time DESC
                """)
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error retrieving conversations: {e}")
            raise e

    def get_conversation_metrics(self):
        """Get basic metrics about conversations."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_calls,
                        AVG(duration_seconds) as avg_duration,
                        MIN(duration_seconds) as min_duration,
                        MAX(duration_seconds) as max_duration
                    FROM conversations_history
                """)
                return cursor.fetchone()
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error retrieving conversation metrics: {e}")
            raise e

    def get_conversations_by_ids(self, conversation_ids: list) -> list:
        """Retrieve specific conversations from the database by their IDs."""
        try:
            with self.connection.cursor() as cursor:
                # Using format() here is safe as we're converting the list to a string
                id_list = ','.join(str(id) for id in conversation_ids)
                cursor.execute(f"""
                    SELECT id, start_time, end_time, duration_seconds, conversation_history
                    FROM conversations_history
                    WHERE id IN ({id_list})
                    ORDER BY start_time DESC
                """)
                results = cursor.fetchall()
                print(f"Retrieved {len(results)} conversations")
                return results
        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error retrieving conversations {conversation_ids}: {e}")
            raise e

    def export_to_json(self, output_dir: str = "exports") -> str:
        """
        Export all conversations from the database to a JSON file.
        
        Args:
            output_dir (str): Directory where the JSON file will be saved. Defaults to "exports".
            
        Returns:
            str: Path to the created JSON file
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(LogComponent.DATABASE, f"Created output directory: {output_dir}")

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"conversations_export_{timestamp}.json")

            # Get all conversations
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, start_time, end_time, duration_seconds, conversation_history
                    FROM conversations_history
                    ORDER BY start_time DESC
                """)
                conversations = cursor.fetchall()

            # Format data for JSON export
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "total_conversations": len(conversations)
                },
                "conversations": [
                    {
                        "id": conv[0],
                        "start_time": conv[1].isoformat(),
                        "end_time": conv[2].isoformat(),
                        "duration_seconds": conv[3],
                        "conversation_history": json.loads(conv[4]) if isinstance(conv[4], str) else conv[4]
                    }
                    for conv in conversations
                ]
            }

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(LogComponent.DATABASE, f"Successfully exported {len(conversations)} conversations to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(LogComponent.DATABASE, f"Error exporting conversations to JSON: {e}")
            raise e 

