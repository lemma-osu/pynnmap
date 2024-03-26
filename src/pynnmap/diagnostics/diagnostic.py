from pynnmap.misc import utilities


class Diagnostic(object):
    _required = None

    def run_diagnostic(self):
        raise NotImplementedError

    def check_missing_files(self):
        files = [getattr(self, attr) for attr in self._required]
        try:
            utilities.check_missing_files(files)
        except utilities.MissingConstraintError as e:
            raise utilities.MissingConstraintError(
                f"Skipping {self.__class__.__name__}"
            ) from e
