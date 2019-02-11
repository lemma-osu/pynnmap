import os


class MissingConstraintError(Exception):
    def __init__(self, message):
        self.message = message


class Diagnostic(object):
    @staticmethod
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

    def run_diagnostic(self):
        raise NotImplementedError
