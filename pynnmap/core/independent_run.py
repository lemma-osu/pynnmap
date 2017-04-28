from pynnmap.core import prediction_run
from pynnmap.misc import utilities


class IndependentRun(prediction_run.PredictionOutput):

    def __init__(self, prediction_run):
        super(IndependentRun, self).__init__(prediction_run)

    def create_predictions(self, no_self_assign_field='LOC_ID'):
        """
        Creates model predictions and zonal pixel files from independent
        predictions, ie. plots are not able to use themselves (or other
        'dependent' plots) as neighbors

        Parameters
        ----------
        no_self_assign_field : str
            ID field at which no self assignment is allowed.
            Defaults to LOC_ID

        Returns
        -------
        None
        """

        # Aliases
        p = self.parameter_parser
        pr = self.prediction_run

        # Create a dictionary between id_field and no_self_assign_field
        env_file = p.environmental_matrix_file
        env_data = utilities.csv2rec(env_file)
        nsaf = no_self_assign_field
        nsa_id_dict = dict(
            (getattr(x, self.id_field), getattr(x, nsaf)) for x in env_data)

        # Open the prediction files
        zonal_pixel_file = p.independent_zonal_pixel_file
        predicted_file = p.independent_predicted_file
        zonal_pixel_fh, predicted_fh = \
            self.open_prediction_files(zonal_pixel_file, predicted_file)

        # Create a generator for each ID in pr.neighbor_data
        prediction_generator = pr.calculate_predictions_at_k(
            k=p.k, id_field=self.id_field, independent=True,
            nsa_id_dict=nsa_id_dict)

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
