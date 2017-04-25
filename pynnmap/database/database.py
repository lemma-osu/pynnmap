import pyodbc


def get_db_object_name(name):
    """
    Given a database object name, break up the name into pieces and return
    a tuple of the fully qualified name parts (catalog, schema, object name)

    Parameters
    ----------
    name : str
        The name of the database object

    Returns
    -------
    out_tuple: tuple
        The components of the database object name
    """

    # Default output list
    out_list = ['lemma', 'dbo', 'udb_plt']

    # Replace the elements of out_list if and only if there exists a
    # replacement for it
    parts = name.split('.')
    for (i, j) in enumerate(range(len(parts) - 1, -1, -1)):
        if parts[j]:
            out_list[(len(out_list) - 1) - i] = parts[j]
    return tuple(out_list)


# Function for quoting stored procedure parameters
def quote_string(field):
    return "'" + str(field) + "'"


class Database(object):

    def __init__(self, dsn):
        """
        Initialize the Database instance with the DSN name

        Parameters
        ----------
        dsn : str
            Name of DSN
        """

        self.dsn = dsn

    def execute_many(self, sql, rows):
        db = pyodbc.connect('DSN=' + self.dsn)
        cursor = db.cursor()
        try:
            cursor.executemany(sql, rows)
            db.commit()
        except:
            err_msg = 'Error caused by sql statement "' + sql + '"'
            raise Exception(err_msg)
        finally:
            cursor.close()
            db.close()

    def update_data(self, sql):
        """
        Execute an sql string that updates or deletes data rather than
        returning a recordset

        Parameters
        ----------
        sql : str
            The SQL string to execute

        Returns
        -------
        None
        """

        db = pyodbc.connect('DSN=' + self.dsn)
        cursor = db.cursor()
        try:
            cursor.execute(sql)
            db.commit()
        except:
            err_msg = 'Error caused by sql statement "' + sql + '"'
            raise Exception(err_msg)
        finally:
            cursor.close()
            db.close()

    def get_data(self, sql):
        """
        Execute an SQL string against this class' database that should
        result in a recordset. Return the results as a list of pyodbc
        records and return the pyodbc cursor description.

        Parameters
        ----------
        sql : str
            The SQL string to execute

        Returns
        -------
        records: list of pyodbc records
            data resulting from execution of sql string
        description: pyodbc cursor description
            field names and data types
        """

        db = pyodbc.connect('DSN=' + self.dsn)
        cursor = db.cursor()
        try:
            records = cursor.execute(sql).fetchall()
        except pyodbc.DataError:
            err_msg = 'caused by the sql statement "' + sql + '"'
            raise pyodbc.DataError(err_msg)
        except pyodbc.ProgrammingError:
            err_msg = 'caused by the sql statement "' + sql + '"'
            raise pyodbc.ProgrammingError(err_msg)
        except AttributeError:
            err_msg = 'caused by the sql statement "' + sql + '"'
            raise AttributeError(err_msg)
        except pyodbc.DataError:
            err_msg = 'caused by the sql statement "' + sql + '"'
            raise pyodbc.DataError(err_msg)
        except Exception:
            err_msg = 'caused by the sql statement "' + sql + '"'
            raise Exception(err_msg)
        finally:
            db.close()

        return records, cursor.description
        cursor.close()
