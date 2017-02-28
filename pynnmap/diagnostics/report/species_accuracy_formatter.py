from reportlab import platypus as p
from reportlab.lib import colors
from reportlab.lib import units as u

from models.diagnostics.report import report_formatter
from models.diagnostics.report import report_styles
from models.misc import utilities
from models.parser import xml_stand_metadata_parser as xsmp
from models.parser import xml_report_metadata_parser as xrmp


class SpeciesAccuracyFormatter(report_formatter.ReportFormatter):

    def __init__(self, parameter_parser):
        pp = parameter_parser
        self.species_accuracy_file = pp.species_accuracy_file
        self.stand_metadata_file = pp.stand_metadata_file

        # Get the report metadata if it exists
        self.report_metadata_file = pp.report_metadata_file

        # Ensure all input files are present
        files = [self.species_accuracy_file, self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except report_formatter.MissingConstraintError as e:
            e.message += '\nSkipping SpeciesAccuracyFormatter\n'
            raise e

    def run_formatter(self):

        # Format the species accuracy information into the main story
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
        title_str = '<strong>Local-Scale Accuracy Assessment:<br/>'
        title_str += 'Species Accuracy at Plot Locations'
        title_str += '</strong>'

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

        # Kappa explanation
        kappa_str = '''
            Cohen's kappa coefficient (Cohen, 1960) is a statistical measure
            of reliability, accounting for agreement occurring by chance.  
            The equation for kappa is: 
        '''
        para = p.Paragraph(kappa_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.05 * u.inch))

        kappa_str = '''
           kappa = (Pr(a) - Pr(e)) / (1.0 - Pr(e))
        '''
        para = p.Paragraph(kappa_str, styles['indented'])
        story.append(para)
        story.append(p.Spacer(0, 0.05 * u.inch))

        kappa_str = '''
            where Pr(a) is the relative observed agreement among
            raters, and Pr(e) is the probability that agreement is
            due to chance.<br/><br/>

            <strong>Abbreviations Used:</strong><br/>
            OP/PP = Observed Present / Predicted Present<br/>
            OA/PP = Observed Absent / Predicted Present
            (errors of commission)<br/>
            OP/PA = Observed Present / Predicted Absent
            (errors of ommission)<br/>
            OA/PA = Observed Absent / Predicted Absent
        '''
        para = p.Paragraph(kappa_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.2 * u.inch))

        # Create a list of lists to hold the species accuracy information
        species_table = []

        # Header row
        header_row = []

        spp_str = '<strong>Species PLANTS Code<br/>'
        spp_str += 'Scientific Name / Common Name</strong>'
        para = p.Paragraph(spp_str, styles['body_style_10'])
        header_row.append(para)

        spp_str = '<strong>Species prevalence</strong>'
        para = p.Paragraph(spp_str, styles['body_style_10'])
        header_row.append(para)

        p1 = p.Paragraph('<strong>OP/PP</strong>',
            styles['body_style_10_right'])
        p2 = p.Paragraph('<strong>OP/PA</strong>',
            styles['body_style_10_right'])
        p3 = p.Paragraph('<strong>OA/PP</strong>',
            styles['body_style_10_right'])
        p4 = p.Paragraph('<strong>OA/PA</strong>',
            styles['body_style_10_right'])
        header_cells = [[p1, p2], [p3, p4]]
        t = p.Table(header_cells, colWidths=[0.75 * u.inch, 0.75 * u.inch])
        t.setStyle(
            p.TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
        header_row.append(t)

        kappa_str = '<strong>Kappa coefficient</strong>'
        para = p.Paragraph(kappa_str, styles['body_style_10'])
        header_row.append(para)
        species_table.append(header_row)

        # Open the species accuracy file into a recarray
        spp_data = utilities.csv2rec(self.species_accuracy_file)

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # Read in the report metadata if it exists
        if self.report_metadata_file:
            rmp = xrmp.XMLReportMetadataParser(self.report_metadata_file)
        else:
            rmp = None

        # Subset the attributes to just species
        attrs = []
        for attr in mp.attributes:
            if attr.species_attr == 1 and 'NOTALY' not in attr.field_name:
                attrs.append(attr.field_name)

        # Iterate over the species and print out the statistics
        for spp in attrs:

            # Empty row to hold the formatted output
            species_row = []

            # Get the scientific and common names from the report metadata
            # if it exists; otherwise, just use the species symbol
            if rmp is not None:

                # Strip off any suffix if it exists
                try:
                    spp_plain = spp.split('_')[0]
                    spp_info = rmp.get_species(spp_plain)
                    spp_str = spp_info.spp_symbol + '<br/>'
                    spp_str += spp_info.scientific_name + ' / '
                    spp_str += spp_info.common_name
                except IndexError:
                    spp_str = spp
            else:
                spp_str = spp
            para = p.Paragraph(spp_str, styles['body_style_10'])
            species_row.append(para)

            # Get the statistical information
            data = spp_data[spp_data.SPECIES == spp][0]
            counts = [data.OP_PP, data.OP_PA, data.OA_PP, data.OA_PA]
            prevalence = data.PREVALENCE
            kappa = data.KAPPA

            # Species prevalence
            prevalence_str = '%.4f' % prevalence
            para = p.Paragraph(prevalence_str, styles['body_style_10_right'])
            species_row.append(para)

            # Capture the plot counts in an inner table
            count_cells = []
            count_row = []
            for i in range(0, 4):
                para = p.Paragraph(
                    '%d' % counts[i], styles['body_style_10_right'])
                count_row.append(para)
                if i % 2 == 1:
                    count_cells.append(count_row)
                    count_row = []
            t = p.Table(count_cells, colWidths=[0.75 * u.inch, 0.75 * u.inch])
            t.setStyle(
                p.TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.white),
                    ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
            species_row.append(t)

            # Print out the kappa statistic
            kappa_str = '%.4f' % kappa
            para = p.Paragraph(kappa_str, styles['body_style_10_right'])
            species_row.append(para)

            # Push this row to the master species table
            species_table.append(species_row)

        # Style this into a reportlab table and add to the story
        col_widths = [(x * u.inch) for x in [4.0, 0.75, 1.5, 0.75]]
        t = p.Table(species_table, colWidths=col_widths)
        t.setStyle(
            p.TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), '#f1efe4'),
                ('GRID', (0, 0), (-1, -1), 2, colors.white),

                ('TOPPADDING', (0, 0), (0, -1), 2),
                ('BOTTOMPADDING', (0, 0), (0, -1), 2),
                ('LEFTPADDING', (0, 0), (0, -1), 6),
                ('RIGHTPADDING', (0, 0), (0, -1), 6),
                ('ALIGNMENT', (0, 0), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (0, -1), 'TOP'),

                ('TOPPADDING', (1, 0), (1, -1), 2),
                ('BOTTOMPADDING', (1, 0), (1, -1), 2),
                ('LEFTPADDING', (1, 0), (1, -1), 6),
                ('RIGHTPADDING', (1, 0), (1, -1), 6),
                ('ALIGNMENT', (1, 0), (1, -1), 'RIGHT'),
                ('VALIGN', (1, 0), (1, 0), 'TOP'),
                ('VALIGN', (1, 1), (1, -1), 'MIDDLE'),

                ('TOPPADDING', (2, 0), (2, -1), 0),
                ('BOTTOMPADDING', (2, 0), (2, -1), 0),
                ('LEFTPADDING', (2, 0), (2, -1), 0),
                ('RIGHTPADDING', (2, 0), (2, -1), 0),
                ('ALIGNMENT', (2, 0), (2, -1), 'LEFT'),
                ('VALIGN', (2, 0), (2, -1), 'TOP'),

                ('TOPPADDING', (3, 0), (3, -1), 2),
                ('BOTTOMPADDING', (3, 0), (3, -1), 2),
                ('LEFTPADDING', (3, 0), (3, -1), 6),
                ('RIGHTPADDING', (3, 0), (3, -1), 6),
                ('ALIGNMENT', (3, 0), (3, -1), 'RIGHT'),
                ('VALIGN', (3, 0), (3, 0), 'TOP'),
                ('VALIGN', (3, 1), (3, -1), 'MIDDLE'),
            ]))
        story.append(t)
        story.append(p.Spacer(0, 0.1 * u.inch))

        rare_species_str = """
            Note that some very rare species do not appear in this accuracy
            report, because these species were not included when building
            the initial ordination model.  The full set of species is
            available upon request.
        """
        para = p.Paragraph(rare_species_str, styles['body_style'])
        story.append(para)

        # Return this story
        return story
