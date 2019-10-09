"""Database interface function definitions."""

import csv
import glob
import sqlite3
import os

import cobmo.config


def create_database(
        sqlite_path,
        sql_path,
        csv_path
):
    """Create SQLITE database from SQL schema file and CSV files."""

    # Connect SQLITE database (creates file, if none).
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Remove old data, if any.
    cursor.executescript(""" 
        PRAGMA writable_schema = 1; 
        DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger'); 
        PRAGMA writable_schema = 0; 
        VACUUM; 
        """)

    # Recreate SQLITE database (schema) from SQL file.
    cursor.executescript(open(sql_path, 'r').read())
    conn.commit()

    # Import CSV files into SQLITE database.
    conn.text_factory = str  # Allows utf-8 data to be stored.
    cursor = conn.cursor()
    for file in glob.glob(os.path.join(csv_path, '*.csv')):
        table_name = os.path.splitext(os.path.basename(file))[0]

        with open(file, 'r') as file:
            first_row = True
            for row in csv.reader(file):
                if first_row:
                    cursor.execute("delete from {}".format(table_name))
                    insert_sql_query = \
                        "insert into {} VALUES ({})".format(table_name, ', '.join(['?' for column in row]))

                    first_row = False
                else:
                    cursor.execute(insert_sql_query, row)
            conn.commit()
    cursor.close()
    conn.close()


def connect_database(
        data_path=cobmo.config.data_path,
        overwrite_database=True
):
    """Connect to the database at given `data_path` and return connection handle."""

    # Create database, if `overwrite_database` or no database exists.
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        create_database(
            sqlite_path=os.path.join(data_path, 'data.sqlite'),
            sql_path=os.path.join(cobmo.config.cobmo_path, 'cobmo', 'database_schema.sql'),
            csv_path=data_path
        )

    # Obtain connection.
    conn = sqlite3.connect(os.path.join(data_path, 'data.sqlite'))
    return conn
