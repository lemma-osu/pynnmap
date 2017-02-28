import re
import os
from reportlab import platypus as p


class MissingConstraintError(Exception):
    def __init__(self, message):
        self.message = message


class ReportFormatter(object):

    # Set up an enumeration for the different pages
    (TITLE, PORTRAIT, LANDSCAPE) = ('title', 'portrait', 'landscape')

    def __init__(self):
        pass

    def check_missing_files(self, files):
        missing_files = []
        for f in files:
            if not os.path.exists(f):
                missing_files.append(f)
        if len(missing_files) > 0:
            err_msg = ''
            for f in missing_files:
                err_msg += '\n' + f + ' does not exist'
            raise MissingConstraintError(err_msg)

    def _make_page_break(self, story, orientation):

        # Set up the next page template
        story.append(p.NextPageTemplate(orientation))

        # Add the page break
        story.append(p.PageBreak())

        # Return this
        return story

    def txt_to_html(self, in_str):
        replace_list = {
            '>': '&gt;',
            '<': '&lt;',
        }

        for i in replace_list:
            in_str = re.sub(i, replace_list[i], in_str)

        return in_str

    def run_formatter(self):
        raise NotImplementedError

    def clean_up(self):
        pass
