
import os

import mysql.connector

from mysql.connector import pooling
from dotenv import load_dotenv


# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

load_dotenv()


MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE", "ROOTCAUSE_AI"),
    "connection_timeout": 10
}


# --------------------------------------------------
# CONNECTION POOL
# --------------------------------------------------

_connection_pool = None


def get_connection_pool():
    """
    Initialize the MySQL connection pool once
    and reuse it across application requests.
    """

    global _connection_pool

    if _connection_pool is None:

        if not MYSQL_CONFIG["user"] or not MYSQL_CONFIG["password"]:
            raise RuntimeError(
                "MySQL credentials are missing. "
                "Check your .env file."
            )

        _connection_pool = pooling.MySQLConnectionPool(
            pool_name="rootcause_pool",
            pool_size=5,
            pool_reset_session=True,
            **MYSQL_CONFIG
        )

        print("MySQL connection pool initialized.")

    return _connection_pool


# --------------------------------------------------
# GET DATABASE CONNECTION
# --------------------------------------------------

def get_db_connection():
    """
    Get a connection from the pool.
    """

    try:
        pool = get_connection_pool()
        return pool.get_connection()

    except mysql.connector.Error as error:
        raise RuntimeError(
            f"MySQL connection failed: {error}"
        ) from error


# --------------------------------------------------
# TEST DATABASE CONNECTION
# --------------------------------------------------

def test_connection():

    connection = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            "SELECT DATABASE() AS database_name, "
            "VERSION() AS mysql_version"
        )

        result = cursor.fetchone()

        cursor.close()

        return {
            "connected": True,
            "database": result["database_name"],
            "mysql_version": result["mysql_version"]
        }

    except Exception as error:
        return {
            "connected": False,
            "error": str(error)
        }

    finally:
        if connection and connection.is_connected():
            connection.close()
