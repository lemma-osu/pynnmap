from pynnmap.database import database as db
from pynnmap.misc import utilities


class WebDatabase(db.Database):

    def __init__(self, project_id, model_region, dsn='web_lemma'):
        self.project_id = project_id
        self.model_region = model_region
        self.dsn = dsn

    def get_model_region_info(self):
        """
        Get information about this model region for the accuracy
        assessment report

        Parameters
        ----------
        None

        Returns
        -------
        model_region_info: numpy.recarray
        """

        sql = """
            EXEC lemma.get_model_region_info
            @model_region = %d,
            @project_id = '%s'
        """

        sql = sql % (self.model_region, self.project_id)
        (records, descr) = self.get_data(sql)
        model_region_info = utilities.pyodbc2rec(records, descr)
        return model_region_info

    def get_people_info(self):
        """
        Get information about people associated with this project
        for the accuracy assessment report

        Parameters
        ----------
        None

        Returns
        -------
        model_region_info: numpy.recarray
        """

        sql = """
            EXEC lemma.get_people_info
            @project_id = '%s'
        """
        sql = sql % (self.project_id)
        (records, descr) = self.get_data(sql)
        people_info = utilities.pyodbc2rec(records, descr)
        return people_info
