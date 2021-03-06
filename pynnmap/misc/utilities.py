# import datetime
# import decimal
import os

import numpy
import pandas as pd
from lxml import etree
from lxml import objectify
from matplotlib import mlab
from six.moves.urllib.request import urlopen


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


def df_to_csv(df, csv_file, index=False, n_dec=4):
    """
    Specialization of pd.DataFrame.to_csv for LEMMA options

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write out
    csv_file : str
        Output file path
    index : bool, optional
        Flag to specify whether to include the dataframe index on export
    n_dec : int, optional
        Number of decimals to include for float formats
    """
    # Fill all missing data with zeros
    df.fillna(0.0, inplace=True)

    # Convert all boolean fields to integer
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype('int')

    # Strip any whitespace from character fields
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert to CSV
    frmt = '%.{}f'.format(n_dec)
    df.to_csv(csv_file, index=index, float_format=frmt)


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
    all_lines = list(all_lines)
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


class MissingConstraintError(Exception):
    def __init__(self, message):
        self.message = message


def check_missing_files(files):
    missing_files = []
    for f in files:
        if not os.path.exists(f):
            missing_files.append(f)
    if len(missing_files) > 0:
        err_msg = ''
        for f in missing_files:
            err_msg += '\n' + f + ' does not exist'
        raise MissingConstraintError(err_msg)


def is_continuous(attr):
    return (
        attr.is_continuous_attr() and
        attr.is_project_attr() and
        attr.is_accuracy_attr() and
        not attr.is_species_attr()
    )


def get_continuous_attrs(mp):
    return [x for x in mp.attributes if is_continuous(x)]


def assert_columns_in_df(df, cols):
    try:
        assert(set(cols).issubset(df.columns))
    except AssertionError:
        msg = 'Columns do not appear in dataframe'
        raise NameError(msg)


def assert_valid_attr_values(df, attr):
    try:
        assert df[attr].isnull().sum() == 0
    except AssertionError:
        msg = 'Data frame has null attribute values'
        raise ValueError(msg)


def assert_same_len_ids(merged_df, df1, df2):
    try:
        merge_num = len(merged_df)
        assert(merge_num == len(df1) or merge_num == len(df2))
    except AssertionError:
        msg = 'Merged data frame does not have same length as originals'
        raise ValueError(msg)


def _list_like(x):
    return x if type(x) in [list, tuple] else [x]


def build_paired_dataframe(obs_df, prd_df, join_field, attr_fields):
    # Ensure we have the columns we want
    attr_fields = _list_like(attr_fields)
    columns = [join_field] + _list_like(attr_fields)
    assert_columns_in_df(obs_df, columns)
    assert_columns_in_df(prd_df, columns)

    # Ensure that all attr_fields values are filled in
    for attr_field in attr_fields:
        assert_valid_attr_values(obs_df, attr_field)
        assert_valid_attr_values(prd_df, attr_field)

    # Subset down to just the columns
    obs_df = obs_df[columns]
    prd_df = prd_df[columns]

    # Merge the data frames using an inner join.  Observed column will
    # have an '_O' suffix and predicted column will have a '_P' suffix
    merged_df = obs_df.merge(prd_df, on=join_field, suffixes=('_O', '_P'))

    # Ensure that the length of the merged dataset matches either the
    # original observed or predicted dataframes
    assert_same_len_ids(merged_df, obs_df, prd_df)
    return merged_df


def build_paired_dataframe_from_files(obs_fn, prd_fn, join_field, attr_fields):
    columns = [join_field] + _list_like(attr_fields)
    obs_df = pd.read_csv(obs_fn, usecols=columns)
    prd_df = pd.read_csv(prd_fn, usecols=columns)
    return build_paired_dataframe(obs_df, prd_df, join_field, attr_fields)
