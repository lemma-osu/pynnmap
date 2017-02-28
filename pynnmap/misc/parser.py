import numpy as np
import re


class ParserError(Exception):
    pass


# General utility functions for size and set checking
def _assert_same_size(a, b):
    if a.size != b.size:
        err_msg = 'The two arrays are not the same size'
        raise ParserError(err_msg)


def _assert_same_set(a, b):
    if not np.all(a == b):
        err_msg = 'The two arrays are not the same set'
        raise ParserError(err_msg)


class Parser(object):
    def __init__(self):
        # Set up commonly used regular expressions for parsing
        self.blank_re = re.compile('^\s*$')

    def read_chunks(self, all_lines, start_re, end_re, skip_lines=0,
            flush=False):
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

    def parse(self):
        raise NotImplementedError
