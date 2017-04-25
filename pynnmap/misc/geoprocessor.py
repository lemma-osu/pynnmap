import os
import time

import arcpy
import numpy as np
from arcpy import env
from arcpy import sa
from matplotlib import mlab


class Geoprocessor(object):

    def __init__(self, workspace_location):
        env.workspace = workspace_location
        if env.workspace is None:
            env.workspace = '.'
        env.scratchWorkspace = 'C:/temp'
        arcpy.CheckOutExtension("Spatial")

    def get_scratch_table_name(self):
        """
        Create a name for a scratch table in this workspace

        Returns
        -------
        scratch_table_name: str
            name for a scratch table that will not conflict with
            names of other objects in the workspace
        """
        try:
            scratch_table_path = arcpy.CreateScratchName(
                '', '', 'ArcInfoTable')
        except:
            raise Exception(arcpy.GetMessages())

        # Return basename of scratch table
        return os.path.basename(scratch_table_path)

    def clip_raster(self, in_raster, boundary_raster, out_raster):
        """
        Clip raster

        Parameters
        ----------
        in_raster : str
            name of input raster to clip
        boundary_raster : str
            name of raster to use as clipping boundary
        out_raster : str
            name of clipped raster
        """
        print 'Clipping raster ' + in_raster
        try:
            scratch = sa.ExtractByMask(in_raster, boundary_raster)
            scratch.save(out_raster)
        except:
            raise Exception(arcpy.GetMessages())

    def copy_raster(self, in_raster, out_raster):
        """
        Copy raster

        Parameters
        ----------
        in_raster : str
            name of input raster to copy
        out_raster : str
            name of copied raster
        """
        print 'Copying ' + str(in_raster)
        try:
            arcpy.CopyRaster_management(in_raster, out_raster)
        except Exception:
            raise Exception(arcpy.GetMessages())

    def copy_raster_no_attributes(self, in_raster, out_raster):
        """
        Copy raster

        Parameters
        ----------
        in_raster : str
            name of input raster to copy
        out_raster : str
            name of copied raster
        """
        print 'Copying (w/o attributes) ' + in_raster
        try:
            scratch = sa.Raster(in_raster)
            scratch.save(out_raster)
            arcpy.BuildRasterAttributeTable_management(out_raster)
        except Exception:
            raise Exception(arcpy.GetMessages())

    def create_masked_raster(self, in_raster, mask_raster, out_raster):
        """
        Create a masked raster

        Parameters:
        -----------
        in_raster : str
            input raster to apply mask
        mask_raster: str
            raster to use as a mask
        out_raster: str
            name of output masked raster

        Returns:
        --------
        None
        """
        print 'Creating masked raster for ' + in_raster
        try:
            scratch = sa.Con(mask_raster, in_raster, mask_raster, "VALUE=0")
            scratch.save(out_raster)
        except:
            raise Exception(arcpy.GetMessages())

    def create_clipped_masked_raster(
            self, in_raster, clip_raster, mask_raster, out_raster):
        """
        Create a clipped and masked raster

        Parameters
        ----------
        in_raster : str
            input raster to apply mask
        clip_raster: str
            raster to use as a clip
        mask_raster: str
            raster to use as a mask
        out_raster: str
            name of output masked raster

        Returns:
        --------
        None
        """
        # Create a scratch name for the temporary grid
        temp_raster = arcpy.CreateScratchName('', '', 'RasterDataset')

        # Create the masked raster
        self.create_masked_raster(in_raster, mask_raster, temp_raster)

        # Create the clipped raster
        self.clip_raster(temp_raster, clip_raster, out_raster)

        # Sleep
        time.sleep(10)

        # Delete the original raster
        arcpy.Delete_management(temp_raster)

    def build_vat(self, raster):
        """
        Build a value attribute table (VAT) on a raster

        Parameters
        ----------
        raster : str
            name of raster to build VAT

        Returns
        -------
        None
        """
        print 'Building VAT for ' + raster
        try:
            arcpy.BuildRasterAttributeTable_management(raster)
        except:
            raise Exception(arcpy.GetMessages())

    def convert_to_integer(self, in_raster, out_raster):
        """
        Converts a floating point raster to an integer raster

        Parameters
        ----------
        in_raster : str
            name of floating point raster to convert to int
        out_raster : str
            name of output converted raster

        Returns
        -------
        None
        """
        print 'Converting ' + in_raster + ' to integer'
        try:
            # create an integer grid
            scratch = sa.Int(sa.RoundDown((sa.Raster(in_raster) * 100) + 0.5))
            scratch.save(out_raster)
        except:
            raise Exception(arcpy.GetMessages())

    def overwrite(self, func, in_raster):
        """
        Calls the function func using the in_raster name for the output
        raster, effectively overwriting it

        Parameters
        ----------
        func : str
            Function to call.  A temporary raster is created before the
            call which gets renamed to in_raster
        in_raster : str
            name of input (and output) raster

        Returns
        -------
        None
        """
        # Create a scratch name for the temporary grid
        temp_raster = arcpy.CreateScratchName('', '', 'RasterDataset')

        # Call the function
        func(in_raster, temp_raster)

        # Delete the original raster
        arcpy.Delete_management(in_raster)

        # Rename the temporary raster
        arcpy.Rename_management(temp_raster, in_raster)

    def define_projection(self, raster, projection_file):
        """
        Defines the projection on a raster

        Parameters
        ----------
        raster : str
            name of raster to define projection on
        projection_file: str
            name and path of file that specifies projection parameters

        Returns
        -------
        None
        """
        print 'Defining projection for ' + raster
        try:
            arcpy.DefineProjection_management(raster, projection_file)
        except:
            raise Exception(arcpy.GetMessages())

    def join_attributes(
            self, raster, raster_join_field, attribute_file,
            attribute_join_field, drop_fields=None):
        """
        Join attributes to a raster

        Parameters
        ----------
        raster : str
            name of raster to join attributes to
        raster_join_field : str
            field in raster to use for joining to attribute data
        attribute_file : str
            name and path of file containing attribute information
        attribute_join_field : str
            field in attribute file to use to join to raster
        drop_fields : list of str
            fields in the attribute file to drop before join to raster

        Returns
        -------
        None

        """
        # First create the ArcInfo table from attribute file (csv)
        info_table, join_fields = self.create_info_table(
            raster_join_field, attribute_file, attribute_join_field,
            drop_fields)
        # Then join attributes from the ArcInfo table to the grid
        self.join_attributes_from_info(
            raster, raster_join_field, attribute_join_field, info_table,
            join_fields)

        # Clean up
        arcpy.Delete_management(info_table)

    def create_info_table(
            self, raster_join_field, attribute_file, attribute_join_field,
            drop_fields=None):
        """
        Create ArcInfo table from attribute csv file

        Parameters
        ----------
        raster : str
            name of raster to join attributes to
        raster_join_field : str
            field in raster to use for joining to attribute data
        attribute_file : str
            name and path of file containing attribute information
        attribute_join_field : str
            field in attribute file to use to join to raster
        drop_fields : list of str
            fields in the attribute file to drop before join to raster

        Returns
        -------
        name of temp ArcInfo table, list of fields to join from info table

        """
        print 'Building info table from attribute file'

        # Crosswalk of numpy types to ESRI types for numeric data
        numpy_to_esri_type = {
            ('b', 1): 'SHORT',
            ('i', 1): 'SHORT',
            ('i', 2): 'SHORT',
            ('i', 4): 'LONG',
            ('f', 4): 'FLOAT',
            ('f', 8): 'DOUBLE',
        }

        # Read the CSV file in to a recarray
        ra = mlab.csv2rec(attribute_file)
        col_names = [str(x).upper() for x in ra.dtype.names]
        ra.dtype.names = col_names

        # If there are fields to drop, do that now and get a new recarray
        if drop_fields is not None:

            # Ensure that the drop fields are actually fields in the current
            # recarray
            drop_fields = [x for x in drop_fields if x in ra.dtype.names]

            # Create a new recarray with these fields omitted
            ra = mlab.rec_drop_fields(ra, drop_fields)
            col_names = list(ra.dtype.names)

        # Get the column types and formats
        col_types = \
            [(ra.dtype[i].kind, ra.dtype[i].itemsize) for i in
                xrange(len(ra.dtype))]
        formats = [ra.dtype[i].str for i in xrange(len(ra.dtype))]

        # Sanitize column names
        #   No field name may be longer than 16 chars
        #   No field name can start with a number
        for i in xrange(len(col_names)):
            if len(col_names[i]) > 16:
                col_names[i] = col_names[i][0:16]
            if col_names[i][0].isdigit():
                col_names[i] = col_names[i].lstrip('0123456789')

        # Reset the names for the recarray
        ra.dtype.names = col_names

        # Sanitize the data
        # Change True/False to 1/0 to be read into short type
        bit_fields = [
            (i, n) for (i, (n, t)) in enumerate(zip(col_names, col_types))
            if t[0] == 'b']
        if bit_fields:
            for rec in ra:
                for (col_num, field) in bit_fields:
                    value = getattr(rec, field)
                    if value:
                        setattr(rec, field, 1)
                    else:
                        setattr(rec, field, 0)

            # Change the bit fields to be short integer
            for (col_num, field) in bit_fields:
                formats[col_num] = '<i2'

        # Create a sanitized recarray and output back to CSV
        temp_csv = os.path.join(env.workspace, 'xxtmp.csv')
        ra2 = np.rec.fromrecords(ra, names=col_names, formats=formats)
        mlab.rec2csv(ra2, temp_csv)

        # Create a scratch name for the temporary ArcInfo table
        temp_table = arcpy.CreateScratchName('', '', 'ArcInfoTable')

        # Create the ArcInfo table and add the fields
        table_name = os.path.basename(temp_table)
        arcpy.CreateTable_management(env.workspace, table_name)
        for (n, t) in zip(col_names, col_types):
            try:
                esri_type = numpy_to_esri_type[t]
                arcpy.AddField_management(temp_table, n, esri_type)
            except KeyError:
                if t[0] == 'S':
                    arcpy.AddField_management(
                        temp_table, n, 'TEXT', '#', '#', t[1])
                else:
                    err_msg = 'Type not found for ' + str(t)
                    print err_msg
                    continue

        # Append the records from the CSV field to the temporary INFO table
        arcpy.Append_management(temp_csv, temp_table, 'NO_TEST')

        # Strip out the join field from the names if they are the same
        raster_join_field = raster_join_field.upper()
        attribute_join_field = attribute_join_field.upper()
        if raster_join_field == attribute_join_field:
            col_names.remove(attribute_join_field)

        # Create a semi-colon delimited string of the fields we want to join
        field_list = ';'.join(col_names)

        # Clean up
        os.remove(temp_csv)

        return temp_table, field_list

    def join_attributes_from_info(
            self, raster, raster_join_field, attribute_join_field, info_table,
            field_list):
        # Join the attributes to the raster
        print 'Joining attributes to ' + raster
        arcpy.BuildRasterAttributeTable_management(raster)
        arcpy.JoinField_management(
            raster, raster_join_field, info_table, attribute_join_field,
            field_list)

    def delete_info_table(self, info_table):
        """
        Deletes info table
        """
        # Clean up
        arcpy.Delete_management(info_table)

    def delete_raster(self, raster):
        """
        Checks for existence of the raster and deletes it if it's found
        """
        if arcpy.Exists(raster):
            try:
                arcpy.Delete_management(raster)
            except:
                Exception(arcpy.GetMessages())
