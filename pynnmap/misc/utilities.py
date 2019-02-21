import datetime
import decimal
import pandas as pd
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import numpy
from lxml import etree
from lxml import objectify
from matplotlib import mlab


# Exact replicate class of FormatFloat to keep four-decimal precision for
# CSV output (mlab.rec2csv puts out full precision)
class FormatDecimal(mlab.FormatFormatStr):
    def __init__(self, precision=4, scale=1.):
        mlab.FormatFormatStr.__init__(self, '%%1.%df' % precision)
        self.precision = precision
        self.scale = scale

    def __hash__(self):
        return hash((self.__class__, self.precision, self.scale))

    def toval(self, x):
        if x is not None:
            x = x * self.scale
        return x

    def fromstr(self, s):
        return float(s)/self.scale


def csv2rec(csv_file, upper_case_field_names=True, **kwargs):
    """
    Convenience wrapper function on top of mlab.csv2rec to optionally upper
    case field names

    Parameters
    ----------
    csv_file : str
        CSV file name

    upper_case_field_names: bool
        Flag whether or not to upper-case the field names which mlab.csv2rec
        leaves lower-case without an option to change them.  Defaults to True.

    kwargs : dict
        Keyword arguments to pass through to mlab.csv2rec

    Returns
    -------
    records : numpy.recarray
        The resulting recarray with (possibly) amended field names
    """
    try:
        records = mlab.csv2rec(csv_file, **kwargs)

    except ValueError:
        if len(open(csv_file, 'r').readlines()) == 1:
            return None
        else:
            raise ValueError

    except IOError:
        err_msg = 'Can''t open input file: ' + csv_file
        raise IOError(err_msg)

    if upper_case_field_names:
        names = [str(x).upper() for x in records.dtype.names]
        records.dtype.names = names
    return records


# def rec2csv(rec_array, csv_file, formatd=None, **kwargs):
#     """
#     Convenience wrapper function on top of mlab.rec2csv to allow fixed-
#     precision output to CSV files
#
#     Parameters
#     ----------
#     rec_array : numpy 1-d recarray
#         The recarray to be written out
#     csv_file : str
#         CSV file name
#     kwargs : dict
#         Keyword arguments to pass through to mlab.rec2csv
#
#     Returns
#     -------
#     None
#     """
#
#     # Get the formatd objects associated with each field
#     formatd = mlab.get_formatd(rec_array, formatd)
#
#     # For all FormatFloat objects, switch to FormatDecimal objects
#     for k, v in formatd.items():
#         if isinstance(v, mlab.FormatFloat):
#             formatd[k] = FormatDecimal()
#
#     # Pass this specification to mlab.rec2csv
#     mlab.rec2csv(rec_array, csv_file, formatd=formatd, **kwargs)


def df2csv(df, csv_file):
    """
    Specialization of pd.to_csv for LEMMA options

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write out
    csv_file : str
        Output file path
    """
    # Fill all missing data with zeros
    df.fillna(0.0, inplace=True)

    # Convert all boolean fields to integer
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype('int')

    # Strip any whitespace from character fields
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert to CSV
    df.to_csv(csv_file, index=False, float_format='%.4f')


# def pyodbc2rec(records, description):
#     """
#     Convert a list of pyodbc records into a numpy recarray.
#     This function handles all type conversion from pyodbc to numpy
#     and dynamically types string fields.
#
#     Parameters
#     ----------
#     records: list of pyodbc records
#         data to convert
#     description: pyodbc cursor description
#         field names and data types
#
#     Returns
#     -------
#     recarray : numpy recarray
#         Data result from the executed SQL string
#     """
#     # Dictionary to set the conversion between pyodbc and numpy for numeric
#     # types. Note that string types get handled dynamically below based
#     # on string length.  If this is going into a class, this dictionary
#     # should be defined at class level to avoid reinitialization on each call
#     pyodbc_x_numpy = {
#         bool: 'int8',
#         datetime.datetime: 'datetime64',
#         decimal.Decimal: 'float64',
#         float: 'float64',
#         int: 'int32',
#         long: 'int64',
#     }
#
#     # Dictionary to set the correct default 'missing' values when NULL
#     # records exist in pyodbc records
#     pyodbc_null = {
#         bool: False,
#         datetime.datetime: 0,
#         decimal.Decimal: 0.0,
#         float: 0.0,
#         int: 0,
#         long: 0,
#         str: '',
#         unicode: '',
#     }
#
#     # Maximum field size allowed (also place at class level)
#     max_size = 5000
#
#     # Figure out the data types for each field.  If the pyodbc type
#     # exists in the pyodbc_x_numpy dict, use that; else, create the
#     # correct numpy data type based on the length of the pyodbc type
#     dtype = []
#     for x in description:
#         field_name = str(x[0]).upper()
#         type_code = x[1]
#         size = x[3]
#         try:
#             dtype.append((field_name, pyodbc_x_numpy[type_code]))
#         except KeyError:
#             try:
#                 # Assert that any data field is less than max_size bytes
#                 assert(size < max_size)
#
#                 # Based on the pyodbc type, either create an appropriately
#                 # size field or raise an error
#                 if type_code == str:
#                     dtype.append((field_name, 'S' + str(size)))
#                 elif type_code == unicode:
#                     dtype.append((field_name, 'U' + str(size)))
#                 elif type_code == buffer:
#                     err_msg = 'No conversion of pyodbc buffer type for %s'
#                     err_msg = err_msg % (field_name)
#                     raise NotImplementedError(err_msg)
#                 else:
#                     err_msg = 'Unknown data type: %s' % (type_code)
#                     raise ValueError(err_msg)
#
#             # Fields larger than max_size bytes
#             except AssertionError:
#                 err_msg = 'The requested field size (%d) for %s '
#                 err_msg += 'is too large'
#                 err_msg = err_msg % ((size, field_name))
#                 raise ValueError(err_msg)
#
#     # For some reason, sending pyodbc.Rows to numpy.rec.fromrecords
#     # doesn't work well with string types.  For now, rewrite records
#     # as a list of lists instead of a list of pyodbc.Rows.
#     records = [list(r) for r in records]
#
#     # Sanitize these records to get rid of None values
#     # We also want to convert the list of lists to a list of tuples
#     # to make the conversion to recarray faster
#     type_codes = [x[1] for x in description]
#     new_records = []
#     for r in records:
#         for i in xrange(len(r)):
#             if r[i] is None:
#                 try:
#                     r[i] = pyodbc_null[type_codes[i]]
#                 except KeyError:
#                     err_msg = 'No default NULL type for %s'
#                     err_msg = err_msg % (type_codes[i])
#                     raise NotImplementedError(err_msg)
#             elif type_codes[i] in (str, unicode):
#                 r[i] = r[i].strip()
#         new_records.append(tuple(r))
#
#     # Convert from pyodbc to numpy recarray and return
#     try:
#         recarray = numpy.rec.fromrecords(new_records, dtype=dtype)
#     except:
#         err_msg = 'Error converting pyodbc cursor to numpy recarray'
#         raise Exception(err_msg)
#     return recarray

def validate_xml(xml_tree, xml_schema_file):
    """
    Verify that xml_tree validates against xml_schema_file

    Parameters
    ----------
    xml_tree : lxml ElementTree
        XML tree to validate

    xml_schema_file : str
        Name of the XML Schema Document to validate this tree against

    Returns
    -------
    None
    """

    xml_schema_doc = etree.parse(urlopen(xml_schema_file))
    xml_schema = etree.XMLSchema(xml_schema_doc)
    xml_schema.assertValid(xml_tree)


def pretty_print(node):
    """
    Pretty prints the specified node as structured XML

    Parameters
    ----------
    node : lxml.ObjectifiedNode
        The node to print (including descendents)

    Returns
    -------
    out_str : str
        The returned formatted string
    """
    tree = node.getroottree()

    # Deannotate the tree
    objectify.deannotate(tree)
    etree.cleanup_namespaces(tree)
    return etree.tostring(node, pretty_print=True)


def subset_lines_from_regex(
        all_lines, start_re, end_re, skip_lines=0, flush=False):
        """
        Read a subset of a list (usually a file held in memory) based on a
        start and end regular expression.  The function returns all chunks
        that were bracketed by the two regular expressions.

        Parameters
        ----------
        all_lines : list
            List of lines over which to check for chunks

        start_re : re.RegexObject
            The starting regular expression to search for

        end_re : re.RegexObject
            The ending regular expression to search for (not included)

        skip_lines : int
            The number of lines to skip after the start_re has been found.
            Defaults to 0

        flush : bool
            Flag for whether or not to write out the last chunk if the
            start_re has been found but the end of file has been reached
            before the chunk was appended to the master list

        Returns
        -------
        chunks : list of lists
            The set of all chunks found by the parser
        """
        pos = 0
        chunks = []
        chunk_lines = []
        while pos < len(all_lines):
            line = all_lines[pos]
            if start_re.match(line):
                # Skip header lines if requested
                pos += skip_lines

                # Keep pushing lines to the chunk until the end_re is found
                while pos < len(all_lines):
                    line = all_lines[pos]
                    if end_re.match(line):
                        # Write this chunk and empty the temporary container
                        chunks.append(chunk_lines)
                        chunk_lines = []
                        break
                    chunk_lines.append(line)
                    pos += 1
            pos += 1

        # If we have anything in chunk_lines, this means we have met the
        # start_re and hit the end of the file without hitting the end_re
        # If flush is True, write this chunk as well
        if flush and len(chunk_lines):
            chunks.append(chunk_lines)
        return chunks
