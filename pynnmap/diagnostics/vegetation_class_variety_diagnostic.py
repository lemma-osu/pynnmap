import numpy as np

from models.diagnostics import diagnostic
from models.misc import utilities


class VegetationClassVarietyDiagnostic(diagnostic.Diagnostic):

    coarse_vc_remap = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 1,
        6: 2,
        7: 3,
        8: 1,
        9: 2,
        10: 3,
        11: 3,
    }

    def __init__(self, parameters):
        p = parameters
        self.stand_attr_file = p.stand_attribute_file
        self.id_field = p.plot_id_field 
        self.output_file = p.vegclass_variety_file

        # Create a list of zonal_pixel files - both independent and dependent
        self.zonal_pixel_files = [
            ('dependent', p.dependent_zonal_pixel_file),
            ('independent', p.independent_zonal_pixel_file),
        ]

        # Ensure all input files are present
        files = [
            self.stand_attr_file,
            parameters.dependent_zonal_pixel_file,
            parameters.independent_zonal_pixel_file,
        ]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping VegetationClassVarietyDiagnostic\n'
            raise e

    def run_diagnostic(self):

        # Open the stand attribute file and subset to just positive IDs
        attr_data = utilities.csv2rec(self.stand_attr_file)
        cond = np.where(getattr(attr_data, self.id_field) > 0)
        attr_data = attr_data[cond]

        # Create a simple dictionary of ID to vegetation class from the
        # attr_data
        vc_dict = dict((getattr(x, self.id_field), getattr(x, 'VEGCLASS'))
            for x in attr_data)

        # Open the output file and write the header
        out_fh = open(self.output_file, 'w')
        out_fh.write('%s,PREDICTION_TYPE\n' % self.id_field)

        # Run this for both independent and dependent predictions
        for (prd_type, zp_file) in self.zonal_pixel_files:

            # Open the zonal pixel file
            zonal_data = utilities.csv2rec(zp_file)

            # For each ID in zonal_data, retrieve the vegetation class of its
            # neighbors
            ids = getattr(attr_data, self.id_field)
            for id in ids:
                cond = np.where(getattr(zonal_data, self.id_field) == id)
                zonal_records = zonal_data[cond]
                vc_records = [vc_dict[x] for x in zonal_records.NEIGHBOR_ID]

                # Apply the logic for the variety
                outlier = self.calculate_vc_variety(vc_records)

                if outlier:
                    out_fh.write('%d,%s\n' % (id, prd_type.upper()))

        # Clean up
        out_fh.close()

    def calculate_vc_variety(self, vc_records):

        # Remap the vc_records to coarser categories
        # - Open/young (VCs 1, 2, 3, 5, 8)
        # - Closed/medium (VCs 4, 6, 9)
        # - Closed/large (VCs 7, 10, 11)

        coarse_records = [self.coarse_vc_remap[x] for x in vc_records]

        # Identify if this plot should be considered a variety plot:
        # has at least 2 '1's and 2 '3's in it
        young = [x for x in coarse_records if x == 1]
        old = [x for x in coarse_records if x == 3]

        if len(young) >= 2 and len(old) >= 2:
            return True
        else:
            return False

    def get_outlier_filename(self):
        return self.output_file
