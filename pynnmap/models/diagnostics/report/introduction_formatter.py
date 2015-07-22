import locale
from datetime import datetime
from reportlab import platypus as p
from reportlab.lib import colors
from reportlab.lib import units as u

from models.diagnostics.report import report_formatter
from models.diagnostics.report import report_styles
from models.parser import xml_report_metadata_parser as xrmp


class IntroductionFormatter(report_formatter.ReportFormatter):

    # Constants
    ACRES_PER_HECTARE = 2.471

    def __init__(self, parameter_parser):
        pp = parameter_parser
        self.report_metadata_file = pp.report_metadata_file
        self.mr = pp.model_region
        self.model_type = pp.model_type
        self.model_year = pp.model_year

        # Ensure all input files are present
        files = [self.report_metadata_file]
        try:
            self.check_missing_files(files)
        except report_formatter.MissingConstraintError as e:
            e.message += '\nSkipping IntroductionFormatter\n'
            raise e

    def run_formatter(self):

        # Create the story
        story = self._create_story()

        # Return the finished story
        return story

    def _create_story(self):

        # Set up an empty list to hold the story
        story = []

        # Import the report styles
        styles = report_styles.get_report_styles()

        # Open the report metadata
        rmp = xrmp.XMLReportMetadataParser(self.report_metadata_file)

        # Offset on this first page
        story.append(p.Spacer(0.0, 0.43 * u.inch))

        # Report title
        title_str = 'GNN Accuracy Assessment Report'
        title = p.Paragraph(title_str, styles['title_style'])
        story.append(title)

        # Model region name and number
        mr_name = rmp.model_region_name
        subtitle_str = mr_name + ' (Modeling Region ' + str(self.mr) + ')'
        para = p.Paragraph(subtitle_str, styles['sub_title_style'])
        story.append(para)

        # Model type information
        model_type_dict = {
            'sppsz': 'Basal-Area by Species-Size Combinations',
            'trecov': 'Tree Percent Cover by Species',
            'wdycov': 'Woody Percent Cover by Species',
            'sppba': 'Basal-Area by Species',
        }
        model_type_str = 'Model Type: '
        model_type_str += model_type_dict[self.model_type]
        para = p.Paragraph(model_type_str, styles['sub_title_style'])
        story.append(para)
        story.append(p.Spacer(0.0, 0.7 * u.inch))

        # Image and flowable to hold MR image and region description
        substory = []
        mr_image_path = rmp.image_path
        image = p.Image(mr_image_path, 3.0 * u.inch, 3.86 * u.inch,
             mask='auto')

        para = p.Paragraph('Overview', styles['heading_style'])
        substory.append(para)
        overview = rmp.model_region_overview
        para = p.Paragraph(overview, styles['body_style'])
        substory.append(para)

        image_flowable = p.ImageAndFlowables(image, substory, imageSide='left',
            imageRightPadding=6)

        story.append(image_flowable)
        story.append(p.Spacer(0.0, 0.2 * u.inch))

        # Contact information
        para = p.Paragraph('Contact Information:', styles['heading_style'])
        story.append(para)
        story.append(p.Spacer(0.0, 0.1 * u.inch))

        contacts = rmp.contacts
        contact_table = []
        contact_row = []
        table_cols = 3
        for (i, contact) in enumerate(contacts):
            contact_str = '<b>' + contact.name + '</b><br/>'
            contact_str += contact.position_title + '<br/>'
            contact_str += contact.affiliation + '<br/>'
            contact_str += 'Phone: ' + contact.phone_number + '<br/>'
            contact_str += 'Email: ' + contact.email_address
            para = p.Paragraph(contact_str, styles['body_style_9'])
            contact_row.append(para)
            if (i % table_cols) == (table_cols - 1):
                contact_table.append(contact_row)
                contact_row = []

        t = p.Table(contact_table)
        t.setStyle(
            p.TableStyle([
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 0), (-1, -1), '#f1efe4'),
                ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1.0, colors.white),
            ]))

        story.append(t)
        story.append(p.Spacer(0, 0.15 * u.inch))

        # Website link
        web_str = '<strong>LEMMA Website:</strong> '
        web_str += '<link color="#0000ff" '
        web_str += 'href="http://lemma.forestry.oregonstate.edu/">'
        web_str += 'http://lemma.forestry.oregonstate.edu</link>'
        para = p.Paragraph(web_str, styles['body_style'])
        story.append(para)

        # Page break
        story = self._make_page_break(story, self.PORTRAIT)

        # General model information
        para = p.Paragraph('General Information', styles['heading_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Report date
        current_time = datetime.now()
        now = current_time.strftime("%Y.%m.%d")
        time_str = '<strong>Report Date:</strong> ' + now
        para = p.Paragraph(time_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Model region area
        locale.setlocale(locale.LC_ALL, "")
        mr_area_ha = rmp.model_region_area
        mr_area_ac = mr_area_ha * self.ACRES_PER_HECTARE
        ha = locale.format('%d', mr_area_ha, True)
        ac = locale.format('%d', mr_area_ac, True)
        area_str = '<strong>Model Region Area:</strong> '
        area_str += str(ha) + ' hectares (' + str(ac) + ' acres)'
        para = p.Paragraph(area_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Model imagery date
        mr_imagery_str = '<strong>Model Imagery Date:</strong> '
        mr_imagery_str += str(self.model_year)
        para = p.Paragraph(mr_imagery_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Plot matching
        if self.model_type == 'sppsz':
            plot_title = ''' 
                <strong>Matching Plots to Imagery for Model Development:
                </strong>
            '''
            para = p.Paragraph(plot_title, styles['body_style'])
            story.append(para)
            story.append(p.Spacer(0, 0.1 * u.inch))
            imagery_str = """
                The current versions of the GNN maps were developed using
                data from inventory plots that span a range of dates, and
                from a yearly time-series of Landsat imagery mosaics from
                1984 to 2012 developed with the LandTrendr algorithms
                (Kennedy et al., 2010). For model development, plots were
                matched to spectral data for the same year as plot
                measurement. In addition, because as many as four plots were
                measured at a given plot location, we constrained the
                imputation for a given map year to only one plot from each
                location -- the plot nearest in date to the imagery (map)
                year. See Ohmann et al. (in press) for more detailed
                information about the GNN modeling process. 
            """
            para = p.Paragraph(imagery_str, styles['body_style'])
            story.append(para)
            story.append(p.Spacer(0, 0.10 * u.inch))

        # Mask information
        mask_title = '<strong>Nonforest Mask Information:</strong>'
        para = p.Paragraph(mask_title, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        mask_str = '''
            An important limitation of the GNN map products is the separation
            of forest and nonforest lands. The GNN modeling applies to forest
            areas only, where we have detailed field plot data. Nonforest
            areas are 'masked' as such using an ancillary map. In California,
            Oregon, Washington and parts of adjacent states, we are using
            maps of Ecological Systems developed for the Gap Analysis
            Program (GAP) as our nonforest mask. There are 'unmasked'
            versions of our GNN maps available upon request,
            in case you have an alternative map of nonforest for your area
            of interest that you would like to apply to the GNN maps.
        '''
        para = p.Paragraph(mask_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Spatial uncertainty
        nn_dist_title = \
            '<strong>Spatial Depictions of GNN Map Uncertainty:</strong>'
        para = p.Paragraph(nn_dist_title, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        nn_dist_str = '''
            In addition to the map diagnostics provided in this report, we
            develop spatial depictions of map uncertainty (available upon
            request). The value shown in the grid for a pixel is the distance
            from that pixel to the nearest-neighbor plot that was imputed to
            the pixel by GNN, and whose vegetation attributes are associated
            with the pixel. 'Distance' is Euclidean distance in
            multi-dimensional gradient space from the gradient model, where
            the axes are weighted by how much variation they explain. The
            nearest-neighbor distance is in gradient (model) space, not
            geographic space. The nearest-neighbor-distance grid can be
            interpreted as a map of potential map accuracy, although it is an
            indicator of accuracy rather than a direct measure. In general,
            the user of a GNN map would have more confidence in map
            reliability for areas where nearest-neighbor distance is short,
            where a similar plot was available (nearby) for the model, and
            less confidence (more uncertainty) where nearest-neighbor
            distance is long. Typically, high nearest-neighbor distances are
            seen in areas with lower sampling intensity of inventory plots.
        '''
        para = p.Paragraph(nn_dist_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Page break
        story = self._make_page_break(story, self.PORTRAIT)

        # Inventory plots by date
        plot_title = '<strong>Inventory Plots in Model Development</strong>'
        para = p.Paragraph(plot_title, styles['heading_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.10 * u.inch))

        # Get all the data sources from the report metadata file
        data_sources = rmp.plot_data_sources

        # Track the total number of plots
        total_plots = 0

        # Create an empty master table
        plot_table = []

        # Create the header row
        p1 = p.Paragraph('<strong>Data Source</strong>',
            styles['contact_style'])
        p2 = p.Paragraph('<strong>Description</strong>',
            styles['contact_style'])
        p3 = p.Paragraph('<strong>Plot Count by Year</strong>',
            styles['contact_style'])
        plot_table.append([p1, p2, p3])

        # Iterate over all data sources
        for ds in data_sources:

            # Iterate over all assessment years for this data source and
            # build an inner table that gives this information
            pc_table = []
            # pc_row = []

            # Hack to avoid the table row being too long.  Should only
            # impact models that use R6_ECO plots
            if len(ds.assessment_years) > 30:

                # Increment the total plot count and track the
                # number of plots in this data source
                ds_count = 0
                for ay in ds.assessment_years:
                    total_plots += ay.plot_count
                    ds_count += ay.plot_count

                # Get the minimum and maximum years
                years = [x.assessment_year for x in ds.assessment_years]
                min_year = min(years)
                max_year = max(years)

                para_str = str(min_year) + '-' + str(max_year) + ': '
                para_str += str(ds_count)
                para = p.Paragraph(para_str, styles['contact_style_right'])
                pc_table.append([[para]])

            else:
                for ay in ds.assessment_years:
                    year = ay.assessment_year
                    plot_count = str(ay.plot_count)

                    # Increment the plot count for total
                    total_plots += ay.plot_count

                    # Add the table row for this year's plot count
                    pc_row = []
                    para = p.Paragraph(year, styles['contact_style_right'])
                    pc_row.append(para)
                    para = p.Paragraph(plot_count,
                        styles['contact_style_right'])
                    pc_row.append(para)
                    pc_table.append(pc_row)

            # Create the inner table
            t = p.Table(pc_table)

            # Table style
            t.setStyle(
                p.TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.white),
                    ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))

            # Add data_source and description to the table
            p1 = p.Paragraph(ds.data_source, styles['contact_style'])
            p2 = p.Paragraph(ds.description, styles['contact_style'])

            # Append these to the master table
            plot_table.append([p1, p2, t])

        # Now append the plot count - columns 1 and 2 will be merged in the
        # table formatting upon return
        p1 = p.Paragraph('Total Plots', styles['contact_style_right_bold'])
        p2 = p.Paragraph(str(total_plots), styles['contact_style_right_bold'])
        plot_table.append(['', p1, p2])

        # Format the table into reportlab
        t = p.Table(plot_table,
            colWidths=[1.3 * u.inch, 4.2 * u.inch, 1.3 * u.inch])
        t.hAlign = 'LEFT'
        t.setStyle(
            p.TableStyle([
                ('GRID', (0, 0), (-1, -2), 1.5, colors.white),
                ('BOX', (0, -1), (-1, -1), 1.5, colors.white),
                ('LINEAFTER', (1, -1), (1, -1), 1.5, colors.white),
                ('BACKGROUND', (0, 0), (-1, -1), '#f1efe4'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),

                ('TOPPADDING', (0, 0), (1, -2), 2),
                ('BOTTOMPADDING', (0, 0), (1, -2), 2),
                ('LEFTPADDING', (0, 0), (1, -2), 6),
                ('RIGHTPADDING', (0, 0), (1, -2), 6),
                ('ALIGNMENT', (0, 0), (1, -2), 'LEFT'),

                ('TOPPADDING', (2, 0), (2, 0), 2),
                ('BOTTOMPADDING', (2, 0), (2, 0), 2),
                ('LEFTPADDING', (2, 0), (2, 0), 6),
                ('RIGHTPADDING', (2, 0), (2, 0), 6),
                ('ALIGNMENT', (2, 0), (2, 0), 'LEFT'),

                ('TOPPADDING', (2, 1), (2, -2), 0),
                ('BOTTOMPADDING', (2, 1), (2, -2), 0),
                ('LEFTPADDING', (2, 1), (2, -2), 0),
                ('RIGHTPADDING', (2, 1), (2, -2), 0),
                ('ALIGNMENT', (2, 1), (2, -2), 'LEFT'),

                ('TOPPADDING', (0, -1), (2, -1), 4),
                ('BOTTOMPADDING', (0, -1), (2, -1), 4),
                ('LEFTPADDING', (0, -1), (2, -1), 6),
                ('RIGHTPADDING', (0, -1), (2, -1), 6),
                ('ALIGNMENT', (0, -1), (2, -1), 'RIGHT'),
            ]))

        # Append this table to the main story
        story.append(t)

        # Page break
        story = self._make_page_break(story, self.PORTRAIT)

        # Print out the spatial predictor variables that are in this model
        ord_title = 'Spatial Predictor Variables in Model Development'
        para = p.Paragraph(ord_title, styles['heading_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.10 * u.inch))

        ord_var_str = """
            The list below represents the spatial predictor
            (GIS/remote sensing) variables that were used in creating
            this model.
        """
        para = p.Paragraph(ord_var_str, styles['body_style'])
        story.append(para)
        story.append(p.Spacer(0, 0.1 * u.inch))

        # Empty container for ordination_rows
        ordination_table = []

        # Create the header row
        p1 = p.Paragraph('<strong>Variable</strong>',
            styles['contact_style'])
        p2 = p.Paragraph('<strong>Description</strong>',
            styles['contact_style'])
        p3 = p.Paragraph('<strong>Data Source</strong>',
            styles['contact_style'])
        ordination_table.append([p1, p2, p3])

        # Read in the ordination variable list and, for each variable,
        # print out the variable name, description, and source into a table
        for var in rmp.ordination_variables:
            name = p.Paragraph(var.field_name, styles['contact_style'])
            desc = p.Paragraph(var.description, styles['contact_style'])
            source = p.Paragraph(var.source, styles['contact_style'])
            ordination_table.append([name, desc, source])

        # Create a reportlab table from this list
        t = p.Table(ordination_table,
            colWidths=[1.0 * u.inch, 2.3 * u.inch, 3.5 * u.inch])
        t.hAlign = 'LEFT'
        t.setStyle(
            p.TableStyle([
                ('GRID', (0, 0), (-1, -1), 1.5, colors.white),
                ('BACKGROUND', (0, 0), (-1, -1), '#f1efe4'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

        # Add to the story
        story.append(t)

        # Return the story
        return story
