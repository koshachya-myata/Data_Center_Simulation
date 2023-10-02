import clickhouse_connect
import os

class DBApi():
    def __init__(self, host=None, port=None,
                 database=None, login=None, password=None) -> None:
        if host is None:
            host = os.environ['CLICKHOUSE_HOST']
        if port is None:
            port = os.environ['CLICKHOUSE_PORT']
        self.database = database
        if self.database is None:
            self.database = 'default'
        if login is None:
            login = os.environ['CLICKHOUSE_LOGIN']
        if password is None:
            password = os.environ['CLICKHOUSE_PASSWORD']
        self.connection = clickhouse_connect.get_client(
            host=host,
            port=int(port),
            username=login,
            password=password,
            send_receive_timeout=12000,
            query_limit=0)

    def create_db(self, database_name=None):
        if database_name is None:
            database_name = os.environ['CLICKHOUSE_DATABASE']
        sql = f'create database if not exists {database_name}'
        result = self.connection.command(sql)
        return result

    def run_sql(self, sql_command):
        result = self.connection.command(sql_command)
        return result

    def close_connection(self):
        self.connection.close()

    def __del__(self):
        self.close_connection()
