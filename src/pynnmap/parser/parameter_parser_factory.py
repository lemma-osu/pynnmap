import os

from .xml_parameter_parser import XMLParameterParser


def get_parameter_parser(parameter_file):
    ext = os.path.splitext(parameter_file)[1]
    if ext.lower() == ".xml":
        parser = XMLParameterParser(parameter_file)
    elif ext.lower() == ".ini":
        err_msg = ".ini files are not yet supported"
        raise NotImplementedError(err_msg)
    else:
        err_msg = f"{ext.lower()} file types are unsupported"
        raise NotImplementedError(err_msg)

    return parser
