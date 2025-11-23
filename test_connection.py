"""
Quick test script to retrieve data from EDS_FACTORS_FUNDAMENTALS_NTM_LTM table
"""
import data_retrieval

def test_table_structure():
    """Test connection and get table structure"""
    retriever = data_retrieval.SnowflakeDataRetriever()
    
    try:
        # Connect to Snowflake
        retriever.connect()
        print("✓ Successfully connected to Snowflake\n")
        
        # First, let's see what databases are available
        print("Checking available databases...")
        db_query = "SHOW DATABASES"
        db_df = retriever.execute_query(db_query)
        print("Available databases:")
        print(db_df[['name']].to_string() if 'name' in db_df.columns else db_df.to_string())
        print("\n" + "="*80 + "\n")
        
        # Use the correct database name
        db_name = "EDS_DEV_BERKELEY"
        
        # Try to use the database
        print("Setting database context...")
        cursor = retriever.conn.cursor()
        try:
            cursor.execute(f'USE DATABASE "{db_name}"')
            print(f"✓ Database context set to {db_name}\n")
        except Exception as e:
            print(f"Note: Could not set database context: {e}\n")
            print("Trying with fully qualified table names...\n")
        cursor.close()
        
        # Check available schemas
        print("Checking available schemas...")
        try:
            schema_query = f'SHOW SCHEMAS IN DATABASE "{db_name}"'
            schemas_df = retriever.execute_query(schema_query)
            if 'name' in schemas_df.columns:
                print("Available schemas:")
                print(schemas_df[['name']].to_string())
            else:
                print("Schemas:")
                print(schemas_df.to_string())
        except Exception as e:
            print(f"Could not list schemas: {e}")
        print("\n" + "="*80 + "\n")
        
        # First, let's check the table structure - try different qualified names
        print("Getting table structure...")
        structure_query = None
        structure_df = None
        
        # Try fully qualified with different schema possibilities (BERKELEY is the schema)
        schema_name = "BERKELEY"
        for query_attempt in [
            f'DESCRIBE TABLE "{db_name}"."{schema_name}"."EDS_FACTORS_FUNDAMENTALS_NTM_LTM"',
            f'DESCRIBE TABLE {db_name}.{schema_name}.EDS_FACTORS_FUNDAMENTALS_NTM_LTM',
            f'DESCRIBE TABLE "{db_name}"."EDS_FACTORS_FUNDAMENTALS_NTM_LTM"',
            f'DESCRIBE TABLE {db_name}.EDS_FACTORS_FUNDAMENTALS_NTM_LTM',
            f'DESCRIBE TABLE "{schema_name}"."EDS_FACTORS_FUNDAMENTALS_NTM_LTM"',
            f'DESCRIBE TABLE {schema_name}.EDS_FACTORS_FUNDAMENTALS_NTM_LTM',
            'DESCRIBE TABLE EDS_FACTORS_FUNDAMENTALS_NTM_LTM'
        ]:
            try:
                structure_query = query_attempt
                structure_df = retriever.execute_query(structure_query)
                print(f"✓ Successfully queried with: {query_attempt}\n")
                break
            except Exception as e:
                continue
        
        if structure_query is None or structure_df is None:
            raise Exception("Could not find table with any qualified name")
        print("\nTable Structure:")
        print(structure_df.to_string())
        print("\n" + "="*80 + "\n")
        
        # Get a sample of data (first 10 rows) - use same qualified name that worked
        print("Getting sample data (first 10 rows)...")
        # Extract table name from the successful structure query
        table_name = structure_query.split('DESCRIBE TABLE ')[1].strip()
        sample_query = f"""
        SELECT * 
        FROM {table_name}
        LIMIT 10
        """
        sample_df = retriever.execute_query(sample_query)
        print(f"\nSample Data ({len(sample_df)} rows):")
        print(sample_df.to_string())
        print(f"\nShape: {sample_df.shape}")
        print(f"Columns: {list(sample_df.columns)}")
        
        # Get some statistics
        print("\n" + "="*80 + "\n")
        print("Getting row count and date range...")
        # Use same table name that worked for structure query
        table_name = structure_query.split('DESCRIBE TABLE ')[1].strip()
        stats_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            MIN(DATE) as earliest_date,
            MAX(DATE) as latest_date
        FROM {table_name}
        """
        stats_df = retriever.execute_query(stats_query)
        print("\nTable Statistics:")
        print(stats_df.to_string())
        
        retriever.disconnect()
        print("\n✓ Test completed successfully!")
        
        return sample_df, structure_df
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if retriever.conn:
            retriever.disconnect()
        raise

if __name__ == "__main__":
    test_table_structure()

