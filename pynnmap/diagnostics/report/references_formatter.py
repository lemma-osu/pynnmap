from reportlab import platypus as p
from reportlab.lib import colors
from reportlab.lib import units as u

from pynnmap.diagnostics.report import report_formatter
from pynnmap.diagnostics.report import report_styles
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class ReferencesFormatter(report_formatter.ReportFormatter):

    def __init__(self):
        pass

    def run_formatter(self):

        # Format the scatterplots into the main story
        story = self._create_story()

        # Return the finished story
        return story

    def _create_story(self):

        # Set up an empty list to hold the story
        story = []

        # Import the report styles
        styles = report_styles.get_report_styles()

        # Create a page break
        story = self._make_page_break(story, self.PORTRAIT)

        # Section title
        title_str = '<strong>References</strong>'
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
        story.append(p.Spacer(0, 0.1 * u.inch))

        # List of references used
        references = [
            '''
            Cohen, J. 1960. "A coefficient of agreement for
            nominal scales." Educational and Psychological Measurement
            20: 37-46.
            ''',
            '''
            Kennedy, RE, Z Yang and WB Cohen. 2010. "Detecting trends
            in forest disturbance and recovery using yearly Landsat
            time series: 1. Landtrendr -- Temporal segmentation
            algorithms." Remote Sensing of Environment 114(2010): 
            2897-2910.
            ''',
            '''
            Ohmann, JL, MJ Gregory and HM Roberts. 2014 (in press). "Scale
            considerations for integrating forest inventory plot data and
            satellite image data for regional forest mapping." Remote
            Sensing of Environment.
            ''',
            '''
            O'Neil, TA, KA Bettinger, M Vander Heyden, BG Marcot, C Barrett,
            TK Mellen, WM Vanderhaegen, DH Johnson, PJ Doran, L Wunder, and
            KM Boula. 2001. "Structural conditions and habitat elements of
            Oregon and Washington. Pages 115-139 in: Johnson, DH and TA
            O'Neil, editors. 2001. "Wildlife-habitat relationships in Oregon
            and Washington." Oregon State University Press, Corvallis, OR.
            ''',
        ]

        # Print all references
        for reference in references:
            para = p.Paragraph(reference, styles['body_style'])
            story.append(para)
            story.append(p.Spacer(0, 0.10 * u.inch))

        # Return this story
        return story
