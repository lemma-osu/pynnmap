import os
import numpy as np
from reportlab import platypus as p
from reportlab.lib import colors
from reportlab.lib import units as u

from models.diagnostics.report import report_formatter
from models.diagnostics.report import report_styles
from models.misc import mpl_figures as mplf
from models.misc import utilities
from models.parser import xml_stand_metadata_parser as xsmp


class LocalAccuracyFormatter(report_formatter.ReportFormatter):

    def __init__(self, parameter_parser):
        pp = parameter_parser
        self.observed_file = pp.stand_attribute_file
        self.predicted_file = pp.independent_predicted_file
        self.stand_metadata_file = pp.stand_metadata_file
        self.id_field = pp.summary_level + 'ID'

        # Ensure all input files are present
        files = [self.observed_file, self.predicted_file,
            self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except report_formatter.MissingConstraintError as e:
            e.message += '\nSkipping LocalAccuracyFormatter\n'
            raise e

    def run_formatter(self):

        # Create the scatterplots
        self.scatter_files = self._create_scatterplots()

        # Format the scatterplots into the main story
        story = self._create_story(self.scatter_files)

        # Return the finished story
        return story

    def clean_up(self):

        # Remove the scatterplots
        for fn in self.scatter_files:
            if os.path.exists(fn):
                os.remove(fn)

    def _create_scatterplots(self):

        # Open files into recarrays
        obs_data = utilities.csv2rec(self.observed_file)
        prd_data = utilities.csv2rec(self.predicted_file)

        # Subset the obs_data to just those IDs in the predicted data
        ids1 = getattr(obs_data, self.id_field)
        ids2 = getattr(prd_data, self.id_field)
        common_ids = np.in1d(ids1, ids2)
        obs_data = obs_data[common_ids]

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # Subset the attributes to those that are continuous, are accuracy
        # attributes, are identified to go into the report, and are not
        # species variables
        attrs = []
        for attr in mp.attributes:
            if attr.field_type == 'CONTINUOUS' and attr.project_attr == 1 and \
                    attr.accuracy_attr == 1 and attr.species_attr == 0:
                attrs.append(attr.field_name)

        # Iterate over the attributes and create a scatterplot file of each
        scatter_files = []
        for attr in attrs:

            # Metadata for this attribute
            metadata = mp.get_attribute(attr)

            # Observed and predicted data matrices for this attribute
            obs_vals = getattr(obs_data, attr)
            prd_vals = getattr(prd_data, attr)

            # Create the output file name
            output_file = attr.lower() + '_scatter.png'

            # Create the scatterplot
            mplf.draw_scatterplot(obs_vals, prd_vals, metadata,
                output_type=mplf.FILE, output_file=output_file)

            # Add this to the list of scatterplot files
            scatter_files.append(output_file)

        # Return the list of scatterplots just created
        return scatter_files

    def _create_story(self, scatter_files):

        # Set up an empty list to hold the story
        story = []

        # Import the report styles
        styles = report_styles.get_report_styles()

        # Create a page break
        story = self._make_page_break(story, self.PORTRAIT)

        # Section title
        title_str = '<strong>Local-Scale Accuracy Assessment: '
        title_str += 'Scatterplots of Observed vs. Predicted '
        title_str += 'Values for Continuous Variables at '
        title_str += 'Plot Locations</strong>'

        para = p.Paragraph(title_str, styles['section_style'])
        t = p.Table([[para]], colWidths=[7.5 * u.inch])
        t.setStyle(
            p.TableStyle([
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 0), (-1, -1), '#957348'),
                ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ]))
        story.append(t)
        story.append(p.Spacer(0, 0.2 * u.inch))

        # Scatter explanation
        scatter_str = '''
            These scatterplots compare the observed plot values against
            predicted (modeled) values for each plot used in the GNN model.
            We use a modified leave-one-out (LOO) approach.  In traditional
            LOO accuracy assessment, a model is run with <i>n</i>-1
            plots and then accuracy is determined at the plot left out of
            modeling, for all plots used in modeling.  Because of computing
            limitations, we use a 'second-nearest-neighbor' approach.  We
            develop our models with all plots, but in determining accuracy, we
            don't allow a plot to assign itself as a neighbor at the plot
            location.  This yields similar accuracy assessment results as
            a true cross-validation approach, but probably slightly
            underestimates the true accuracy of the distributed
            (first-nearest-neighbor) map.<br/><br/>
            The observed value comes directly from the plot data,
            whereas the predicted value comes from the GNN prediction
            for the plot location.  The GNN prediction is the mean of
            pixel values for a window that approximates the
            field plot configuration.<br/><br/>
            The correlation coefficients, normalized Root Mean Squared
            Errors (RMSE), and coefficients of determination (R-square) are
            given. The RMSE is normalized by dividing the RMSE by the
            observed mean value.
        '''

        para = p.Paragraph(scatter_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Add the scatterplot images to a list of lists
        table_cols = 2
        scatter_table = []
        scatter_row = []
        for (i, fn) in enumerate(scatter_files):
            scatter_row.append(p.Image(fn, 3.4 * u.inch, 3.0 * u.inch))
            if (i % table_cols) == (table_cols - 1):
                scatter_table.append(scatter_row)
                scatter_row = []

        # Determine if there are any scatterplots left to print
        if len(scatter_row) != 0:
            for i in range(len(scatter_row), table_cols):
                scatter_row.append(p.Paragraph('', styles['body_style']))
            scatter_table.append(scatter_row)

        # Style this into a reportlab table and add to the story
        width = 3.75 * u.inch
        t = p.Table(scatter_table, colWidths=[width, width])
        t.setStyle(
            p.TableStyle([
                ('ALIGNMENT', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 6.0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6.0),
            ]))
        story.append(t)

        # Return this story
        return story
