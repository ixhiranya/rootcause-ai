
from app.database import test_connection


def main():

    print("=" * 55)
    print("ROOTCAUSE AI - MYSQL CONNECTION TEST")
    print("=" * 55)

    result = test_connection()

    if result["connected"]:

        print("\nMySQL connection successful!")

        print("Database:", result["database"])
        print("MySQL version:", result["mysql_version"])

        print("\nALL DATABASE TESTS PASSED")

    else:

        print("\nMySQL connection failed!")
        print("Error:", result["error"])


if __name__ == "__main__":
    main()
