import sqlite3
from pathlib import Path

def merge_databases(db1_path: str, db2_path: str, output_path: str):
    """Merge two SQLite databases with identical structures, combining their data"""
    # Initialize connections as None
    conn1 = None
    conn2 = None
    conn_out = None
    
    try:
        # Verify input paths exist
        if not Path(db1_path).exists():
            raise FileNotFoundError(f"First database not found: {db1_path}")
        if not Path(db2_path).exists():
            raise FileNotFoundError(f"Second database not found: {db2_path}")
            
        # Connect to databases
        conn1 = sqlite3.connect(db1_path)
        conn2 = sqlite3.connect(db2_path)
        conn_out = sqlite3.connect(output_path)
        
        # Get list of tables from both databases
        tables1 = set(row[0] for row in conn1.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        tables2 = set(row[0] for row in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        
        # Combine all unique table names
        all_tables = tables1.union(tables2)
        
        for table in all_tables:
            print(f"Processing table: {table}")
            
            try:
                # Get table structure from whichever database has it
                if table in tables1:
                    create_table_sql = conn1.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()[0]
                else:
                    create_table_sql = conn2.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()[0]
                
                # Create table in output database
                conn_out.execute(create_table_sql)
                
                # Copy data from first database if table exists there
                if table in tables1:
                    data = conn1.execute(f"SELECT * FROM {table}").fetchall()
                    if data:
                        placeholders = ','.join(['?' for _ in range(len(data[0]))])
                        conn_out.executemany(f"INSERT INTO {table} VALUES ({placeholders})", data)
                
                # Copy data from second database if table exists there
                if table in tables2:
                    data = conn2.execute(f"SELECT * FROM {table}").fetchall()
                    if data:
                        placeholders = ','.join(['?' for _ in range(len(data[0]))])
                        conn_out.executemany(f"INSERT INTO {table} VALUES ({placeholders})", data)
                
                # Remove duplicates if any
                conn_out.execute(f"CREATE TABLE temp_table AS SELECT DISTINCT * FROM {table}")
                conn_out.execute(f"DROP TABLE {table}")
                conn_out.execute(f"ALTER TABLE temp_table RENAME TO {table}")
                
                conn_out.commit()
                print(f"Successfully merged table: {table}")
                
            except sqlite3.Error as e:
                print(f"Error processing table {table}: {e}")
                continue
        
        print(f"Successfully merged databases into: {output_path}")
        
    except Exception as e:
        print(f"Error merging databases: {e}")
        
    finally:
        # Safely close connections if they were opened
        if conn1:
            conn1.close()
        if conn2:
            conn2.close()
        if conn_out:
            conn_out.close()

if __name__ == "__main__":
    # Replace these with your actual database paths
    db1_path = "results/S2_results.db"  # Update with your first database path
    db2_path = "results/S2_additional.db"  # Update with your second database path
    output_path = "results/S2C.db"  # Update with your desired output path
    
    merge_databases(db1_path, db2_path, output_path) 