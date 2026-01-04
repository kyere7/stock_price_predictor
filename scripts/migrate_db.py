"""
Database migration script to add missing columns.
Run this once to update the schema to support model versioning.
"""
import sys
import os
from pathlib import Path

# Add parent directory to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import psycopg2
from backend.config import load_config

def migrate():
    """Apply all pending migrations."""
    config = load_config()
    
    try:
        conn = psycopg2.connect(
            host=config.get('host', 'localhost'),
            port=int(config.get('port', '5432')),
            database=config.get('database', 'stock_predictor'),
            user=config.get('user', 'postgres'),
            password=config.get('password', ''),
        )
        cursor = conn.cursor()
        
        print("Checking if model_versions table exists...")
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'model_versions'
            )
        """)
        model_versions_exists = cursor.fetchone()[0]
        
        if not model_versions_exists:
            print("Creating model_versions table...")
            cursor.execute("""
                CREATE TABLE model_versions (
                    id SERIAL PRIMARY KEY,
                    version_tag VARCHAR(100) NOT NULL UNIQUE,
                    model_type VARCHAR(50),
                    artifact_path VARCHAR(500),
                    trained_at TIMESTAMP NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    notes VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            print("✓ Created model_versions table")
        
        print("Checking if predictions table has model_version_id column...")
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'predictions' AND column_name = 'model_version_id'
            )
        """)
        has_model_version_id = cursor.fetchone()[0]
        
        if not has_model_version_id:
            print("Adding model_version_id column to predictions...")
            # First, add the column as nullable
            cursor.execute("""
                ALTER TABLE predictions 
                ADD COLUMN model_version_id INTEGER
            """)
            
            # Create a default model version if predictions exist
            cursor.execute("SELECT COUNT(*) FROM predictions")
            pred_count = cursor.fetchone()[0]
            
            if pred_count > 0:
                print(f"Found {pred_count} existing predictions. Creating default model version...")
                cursor.execute("""
                    INSERT INTO model_versions (version_tag, model_type, trained_at, is_active)
                    VALUES ('legacy_v1', 'Unknown', NOW(), 0)
                    ON CONFLICT (version_tag) DO NOTHING
                    RETURNING id
                """)
                result = cursor.fetchone()
                if result:
                    default_version_id = result[0]
                else:
                    cursor.execute("SELECT id FROM model_versions WHERE version_tag = 'legacy_v1'")
                    default_version_id = cursor.fetchone()[0]
                
                print(f"Using model_version_id {default_version_id} for existing predictions")
                cursor.execute(
                    "UPDATE predictions SET model_version_id = %s WHERE model_version_id IS NULL",
                    (default_version_id,)
                )
            
            # Now add the NOT NULL constraint and foreign key
            cursor.execute("""
                ALTER TABLE predictions 
                ALTER COLUMN model_version_id SET NOT NULL,
                ADD CONSTRAINT fk_pred_model_version_id 
                    FOREIGN KEY (model_version_id) 
                    REFERENCES model_versions(id)
            """)
            
            conn.commit()
            print("✓ Added model_version_id column with foreign key")
        
        print("Checking if trading_day_offset column exists...")
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'predictions' AND column_name = 'trading_day_offset'
            )
        """)
        has_offset = cursor.fetchone()[0]
        
        if not has_offset:
            print("Adding trading_day_offset column...")
            cursor.execute("""
                ALTER TABLE predictions 
                ADD COLUMN trading_day_offset INTEGER DEFAULT 0
            """)
            conn.commit()
            print("✓ Added trading_day_offset column")
        
        print("\n✓ Migration complete! Database schema is now up to date.")
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    migrate()
