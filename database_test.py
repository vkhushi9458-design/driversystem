
import sqlite3
from datetime import datetime

DATABASE_NAME = "driver_monitor.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()

    print("Database and events table created successfully.")


def save_event(event_type):
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO events (event_type, event_time)
        VALUES (?, ?)
    """, (event_type, event_time))

    connection.commit()
    connection.close()

    print(f"Event saved: {event_type} at {event_time}")


def show_events():
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, event_type, event_time
        FROM events
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    print("\n--- DRIVER MONITORING EVENT HISTORY ---")

    if not events:
        print("No events recorded yet.")

    for event in events:
        print(
            f"ID: {event[0]} | "
            f"Event: {event[1]} | "
            f"Time: {event[2]}"
        )


if __name__ == "__main__":
    create_database()

    save_event("EYES_CLOSED")
    save_event("YAWNING")
    save_event("HEAD_DOWN")

    show_events()
