from reportlab.lib import units as u

from models.diagnostics.report.report_formatter import MissingConstraintError
from models.diagnostics.report import accuracy_report
from models.diagnostics.report import introduction_formatter as intro
from models.diagnostics.report import data_dictionary_formatter as ddf
from models.diagnostics.report import local_accuracy_formatter as laf
from models.diagnostics.report import references_formatter as rf
from models.diagnostics.report import regional_accuracy_formatter as raf
from models.diagnostics.report import riemann_accuracy_formatter as riemann
from models.diagnostics.report import species_accuracy_formatter as saf
from models.diagnostics.report import vegetation_class_formatter as vcf
from models.diagnostics.report import report_styles


class LemmaAccuracyReport(accuracy_report.AccuracyReport):

    # Dictionary of diagnostic name to report type
    report_type = {
        'local_accuracy': laf.LocalAccuracyFormatter,
        'regional_accuracy': raf.RegionalAccuracyFormatter,
        'riemann_accuracy': riemann.RiemannAccuracyFormatter,
        'species_accuracy': saf.SpeciesAccuracyFormatter,
        'vegclass_accuracy': vcf.VegetationClassFormatter,
    }

    def __init__(self, parameter_parser):
        self.parameter_parser = parameter_parser

    def create_accuracy_report(self):
        p = self.parameter_parser

        # Get the name of the output accuracy report
        out_report = p.accuracy_assessment_report

        # Set up the document template
        pdf = report_styles.GnnDocTemplate(out_report,
            leftMargin=0.75 * u.inch, rightMargin=0.75 * u.inch,
            topMargin=0.6 * u.inch, bottomMargin=0.6 * u.inch)

        # Begin the story ...
        self.story = []

        # Make a list of formatters which are separate subsections of the
        # report
        formatters = []

        # Add the introduction to the list of formatters if the report
        # metadata is present
        if p.report_metadata_file:
            try:
                f = intro.IntroductionFormatter(self.parameter_parser)
                formatters.append(f)
            except MissingConstraintError as e:
                print e.message

        # Get instances of the separate components needed to build the
        # accuracy assessment report
        for d in p.include_in_report:
            try:
                f = (self.report_type[d])(self.parameter_parser)
                formatters.append(f)
            except MissingConstraintError as e:
                print e.message

        # Add the data dictionary and references formatters at the end of the
        # report
        try:
            f = ddf.DataDictionaryFormatter(self.parameter_parser)
            formatters.append(f)
        except MissingConstraintError as e:
            print e.message

        try:
            f = rf.ReferencesFormatter()
            formatters.append(f)
        except MissingConstraintError as e:
            print e.message

        # Run each instance's formatter
        for f in formatters:
            sub_story = f.run_formatter()
            if sub_story is not None:
                self.story.extend(sub_story[:])

        # Explicitly garbage collect sub_story
        sub_story = None

        # Write out the story
        if len(self.story) > 0:
            pdf.build(self.story,
                onTitle=report_styles.title,
                onPortrait=report_styles.portrait,
                onLandscape=report_styles.landscape)

        # Clean up if necessary for each formatter
        for f in formatters:
            f.clean_up()
