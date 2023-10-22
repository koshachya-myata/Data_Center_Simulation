"""Class for interaction whith clickhouse database."""
from typing import Union
import clickhouse_connect
import os


class DBApi():
    """Class for interaction with clickhouse database."""

    def __init__(
            self,
            host: Union[str, None] = None,
            port: Union[int, None] = None,
            database: Union[str, None] = None,
            login: Union[str, None] = None,
            password: Union[str, None] = None) -> None:
        """
        Init DBAPI class, set self.database and self.connection.

        If some parametr is None, it setted from environ.

        Args:
            host (Union[str, None], optional): host. Defaults to None.
            port (Union[int, None], optional): port. Defaults to None.
            database (Union[str, None], optional): databse name.
                                                   Defaults to None.
            login (Union[str, None], optional): username. Defaults to None.
            password (Union[str, None], optional): password. Defaults to None.
        """
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

    def create_db(self, database_name: str = None):
        """
        Create database with database_name.

        If database_name is None create db with name as
        CLICKHOUSE_DATABASE environ.

        Args:
            database_name (str, optional): Database name. Defaults to None.

        Returns:
            (str | int | Sequence[str] | QuerySummary): connection sql-command
                                                        info.
        """
        if database_name is None:
            database_name = os.environ['CLICKHOUSE_DATABASE']
        sql = f'create database if not exists {database_name}'
        result = self.connection.command(sql)
        return result

    def run_sql(self, sql_command: str):
        """
        Run sql_command on self.connection.

        Args:
            sql_command (str): sql command.

        Returns:
            (str | int | Sequence[str] | QuerySummary): connection sql-command
                                                        info.
        """
        result = self.connection.command(sql_command)
        return result

    def close_connection(self):
        """Close self.connection."""
        self.connection.close()

    def __del__(self):
        """Close self.connection."""
        self.close_connection()
