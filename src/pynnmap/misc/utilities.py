import os
from urllib.request import urlopen

import numpy as np
import pandas as pd
from lxml import etree, objectify


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
    df.fillna(0.0, inplace=True)

    # Convert all boolean fields to integer
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int")

    # Strip any whitespace from character fields
    def strip_ws(x):
        return (
            x.str.strip() if x.dtype == "object" and isinstance(x.iloc[0], str) else x
        )

    df = df.apply(strip_ws)

    # Convert to CSV
    frmt = f"%.{n_dec}f"
    df.to_csv(csv_file, index=index, float_format=frmt)


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
    objectify.deannotate(tree)
    etree.cleanup_namespaces(tree)
    return etree.tostring(node, pretty_print=True)


def subset_lines_from_regex(all_lines, start_re, end_re, skip_lines=0, flush=False):
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
    if flush and len(chunk_lines) > 0:
        chunks.append(chunk_lines)
    return chunks


class MissingConstraintError(Exception):
    """Raised when required files for a diagnostic are missing."""

    ...


def check_missing_files(files):
    missing_files = [fn for fn in files if not os.path.exists(fn)]
    if missing_files:
        err_msg = "".join(f"\n{fn} does not exist" for fn in missing_files)
        raise MissingConstraintError(err_msg)


def is_continuous(attr):
    return (
        attr.is_continuous_attr()
        and attr.is_project_attr()
        and attr.is_accuracy_attr()
        and not attr.is_species_attr()
    )


def is_area(attr):
    return (
        attr.is_project_attr()
        and attr.is_accuracy_attr()
        and not attr.is_species_attr()
    )


def get_continuous_attrs(mp):
    return [x for x in mp.attributes if is_continuous(x)]


def get_area_attrs(mp):
    return [x for x in mp.attributes if is_area(x)]


def assert_columns_in_df(df, cols):
    try:
        assert set(cols).issubset(df.columns)
    except AssertionError as e:
        msg = "Columns do not appear in dataframe"
        raise NameError(msg) from e


def assert_valid_attr_values(df, attr):
    try:
        assert df[attr].isnull().sum() == 0
    except AssertionError as e:
        msg = "Data frame has null attribute values"
        raise ValueError(msg) from e


def assert_same_len_ids(merged_df, df1, df2):
    try:
        merge_num = len(merged_df)
        assert merge_num in [len(df1), len(df2)]
    except AssertionError as e:
        msg = "Merged data frame does not have same length as originals"
        raise ValueError(msg) from e


def _list_like(x):
    return x if isinstance(x, (list, tuple)) else [x]


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
    merged_df = obs_df.merge(prd_df, on=join_field, suffixes=("_O", "_P"))

    # Ensure that the length of the merged dataset matches either the
    # original observed or predicted dataframes
    assert_same_len_ids(merged_df, obs_df, prd_df)
    return merged_df


def build_paired_dataframe_from_files(obs_fn, prd_fn, join_field, attr_fields):
    columns = [join_field] + _list_like(attr_fields)
    obs_df = pd.read_csv(obs_fn, usecols=columns)
    prd_df = pd.read_csv(prd_fn, usecols=columns)
    return build_paired_dataframe(obs_df, prd_df, join_field, attr_fields)


def build_obs_prd_dataframes(obs_fn, prd_fn, common_field):
    # Read the observed and predicted files into dataframes
    obs_df = pd.read_csv(obs_fn, low_memory=False)
    prd_df = pd.read_csv(prd_fn, low_memory=False)

    # Subset the dataframes just to the IDs that are in both data frames
    obs_ids = getattr(obs_df, common_field)
    prd_ids = getattr(prd_df, common_field)
    obs_keep = np.in1d(obs_ids, prd_ids)
    prd_keep = np.in1d(prd_ids, obs_ids)
    return obs_df[obs_keep], prd_df[prd_keep]
