from models.core import prediction_run


class DependentRun(prediction_run.PredictionOutput):

    def __init__(self, prediction_run):
        super(DependentRun, self).__init__(prediction_run)

    def create_predictions(self):
        """
        Creates model predictions, zonal pixel, and nn_index files from
        dependent predictions, ie. plots are able to use themselves as
        neighbors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Aliases
        p = self.parameter_parser
        pr = self.prediction_run

        # Write the dependent nn_index file
        nn_index_file = p.dependent_nn_index_file
        pr.write_nn_index_file(self.id_field, nn_index_file)

        # Open the prediction files
        zonal_pixel_file = p.dependent_zonal_pixel_file
        predicted_file = p.dependent_predicted_file
        zonal_pixel_fh, predicted_fh = \
            self.open_prediction_files(zonal_pixel_file, predicted_file)

        # Create a generator for each ID in pr.neighbor_data
        prediction_generator = pr.calculate_predictions_at_k(
            k=p.k, id_field=self.id_field, independent=False)

        # Iterate over each prediction writing them out to the zonal pixel
        # and predicted attribute files
        for plot_prediction in prediction_generator:

            # Write this record to the zonal pixel file
            pr.write_zonal_pixel_record(plot_prediction, zonal_pixel_fh)

            # Write this record to the predicted attribute file
            pr.write_predicted_record(plot_prediction, predicted_fh)

        # Close files
        zonal_pixel_fh.close()
        predicted_fh.close()
