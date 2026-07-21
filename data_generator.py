import sqlite3
import random
import os
from datetime import datetime, timedelta

def generate_database(db_path="data/analytics.db", num_rows=100_000):
    print(f"Generating synthetic database at {db_path} with {num_rows} rows...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect and create table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS transactions;")
    cursor.execute("""
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount REAL,
            timestamp TEXT,
            category TEXT
        );
    """)
    
    # Setup generator parameters
    categories = ["Electronics", "Grocery", "Clothing", "Books", "Entertainment", "Travel"]
    base_date = datetime.now() - timedelta(days=90)
    
    # Generate data in batches for speed
    batch_size = 10000
    data = []
    
    # We will simulate ~500 users
    user_avg_spend = {user_id: random.uniform(10.0, 100.0) for user_id in range(1, 501)}
    
    for i in range(num_rows):
        user_id = random.randint(1, 500)
        # Standard transaction amount centered around user's average spend
        amount = round(random.normalvariate(user_avg_spend[user_id], user_avg_spend[user_id] * 0.2), 2)
        amount = max(1.0, amount) # Ensure no negative purchases
        
        # Inject anomalies (~0.5% chance)
        if random.random() < 0.005:
            amount = round(user_avg_spend[user_id] * random.uniform(10.0, 25.0), 2) # 10x to 25x normal spend
            
        timestamp = (base_date + timedelta(
            seconds=random.randint(0, 90 * 24 * 60 * 60)
        )).strftime("%Y-%m-%d %H:%M:%S")
        
        category = random.choice(categories)
        
        data.append((user_id, amount, timestamp, category))
        
        if len(data) >= batch_size:
            cursor.executemany(
                "INSERT INTO transactions (user_id, amount, timestamp, category) VALUES (?, ?, ?, ?);",
                data
            )
            data = []
            
    if data:
        cursor.executemany(
            "INSERT INTO transactions (user_id, amount, timestamp, category) VALUES (?, ?, ?, ?);",
            data
        )
        
    # Create indexes to support fast queries
    cursor.execute("CREATE INDEX idx_user ON transactions(user_id);")
    cursor.execute("CREATE INDEX idx_category ON transactions(category);")
    cursor.execute("CREATE INDEX idx_timestamp ON transactions(timestamp);")
    
    conn.commit()
    conn.close()
    print("Database generation completed successfully.")

if __name__ == "__main__":
    generate_database()
