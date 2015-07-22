from reportlab import platypus as p
from reportlab.lib import colors
from reportlab.lib import units as u

from models.diagnostics.report import report_formatter
from models.diagnostics.report import report_styles
from models.parser import xml_stand_metadata_parser as xsmp


class DataDictionaryFormatter(report_formatter.ReportFormatter):

    def __init__(self, parameter_parser):
        pp = parameter_parser
        self.stand_metadata_file = pp.stand_metadata_file
        self.model_type = pp.model_type

        # Ensure all input files are present
        files = [self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except report_formatter.MissingConstraintError as e:
            e.message += '\nSkipping DataDictionaryFormatter\n'
            raise e

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
        title_str = '<strong>Data Dictionary</strong>'
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

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # Subset the attributes to those that are accuracy attributes, are
        # identified to go into the report, and are not species variables
        attrs = []
        for attr in mp.attributes:
            if attr.accuracy_attr == 1 and attr.project_attr == 1 and \
                    attr.species_attr == 0:
                attrs.append(attr.field_name)

        # Set up the master dictionary table
        dictionary_table = []

        # Iterate through the attributes and print out the field information
        # and codes if present
        for attr in attrs:
            metadata = mp.get_attribute(attr)
            field_name = metadata.field_name
            units = metadata.units
            description = metadata.description

            field_para = p.Paragraph(field_name, styles['body_style_10'])
            if units != 'none':
                description += ' (' + units + ')'
            field_desc_para = p.Paragraph(description, styles['body_style_10'])

            # If this field has codes, create a sub table underneath the
            # field description
            if metadata.codes:

                # Set up a container to hold the code rows
                code_table = []

                # Iterate over all code rows and append to the code_table
                for code in metadata.codes:
                    code_para = \
                        p.Paragraph(code.code_value, styles['code_style'])
                    description = self.txt_to_html(code.description)
                    code_desc_para = \
                        p.Paragraph(description, styles['code_style'])
                    code_table.append([code_para, code_desc_para])

                # Convert this to a reportlab table
                t = p.Table(code_table, colWidths=[0.75 * u.inch, 4.5 * u.inch])
                t.setStyle(
                    p.TableStyle([
                        ('TOPPADDING', (0, 0), (-1, -1), 3),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                        ('BACKGROUND', (0, 0), (-1, -1), '#f7f7ea'),
                        ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('GRID', (0, 0), (-1, -1), 0.25, colors.white),
                    ]))

                # Create a stack of the field description and field codes
                elements = \
                    [[field_desc_para], [t]]

            # If no codes exist, just add the field description
            else:
                elements = [[field_desc_para]]

            # Create a reportlab table of the field description and
            # (if present) field codes
            description_table = \
                p.Table(elements, colWidths=[5.25 * u.inch])
            description_table.setStyle(
                p.TableStyle([
                    ('TOPPADDING', (0, 0), (-1, 0), 0),
                    ('BOTTOMPADDING', (0, -1), (-1, -1), 0),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))

            dictionary_table.append([field_para, description_table])

        # Format the dictionary table into a reportlab table
        t = p.Table(dictionary_table, colWidths=[1.6 * u.inch, 5.4 * u.inch])
        t.setStyle(
            p.TableStyle([
                ('TOPPADDING', (0, 0), (0, -1), 5),
                ('BOTTOMPADDING', (0, 0), (0, -1), 5),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BACKGROUND', (0, 0), (-1, -1), '#f1efe4'),
            ]))
        story.append(t)

        # Description of the species information that is attached to ArcInfo
        # grids.  We don't enumerate the codes here, but just give this
        # summary information
        spp_str = """
            Individual species abundances are attached to ArcInfo grids that
            LEMMA distributes.  For this model, fields designate species
            codes based on the <link color="#0000ff"
            href="http://plants.usda.gov/">USDA PLANTS database</link> from
            the year 2000, and values represent species
        """
        if self.model_type in ['sppsz', 'sppba']:
            spp_str += " basal area (m^2/ha)."
        elif self.model_type in ['trecov', 'wdycov']:
            spp_str += " percent cover."

        para = p.Paragraph(spp_str, styles['body_style'])
        story.append(p.Spacer(0, 0.1 * u.inch))
        story.append(para)

        # Return this story
        return story
