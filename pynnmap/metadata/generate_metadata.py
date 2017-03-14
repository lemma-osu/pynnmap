import sys
from genshi import template
from models.database import plot_database as db

class GenerateMetadata(object):
    def __init__(self):
        pass
    
    def create_metadata_file(self, table_name):
        #assign dummy values to required PlotDatabase parameters since we
        #don't need these parameters to generate metadata
        model_type = 'sppsz'
        model_region = 221
        buffer = 0
        model_year = 2000
        image_source = 'N/A'
        image_version = 0.0
        dsn = 'rocky2lemma'
        
        plot_db = db.PlotDatabase(model_type, model_region, buffer,
                                  model_year,  
                                  image_source, image_version, dsn)
        
        metadata = plot_db.get_metadata_field_dictionary(table_name)
        
        ordinal_dict = {}
        for key in metadata.keys():
            ordinal_dict[metadata[key]['ORDINAL']] = key
        
        for ordinal in sorted(ordinal_dict.keys()):
            key = ordinal_dict[ordinal]
            yield metadata[key]
        
if __name__ == '__main__':
    #name of database table for which to generate metadata
    table_name = sys.argv[1]
    #name and path of output xml file containing metadata
    out_file = sys.argv[2]
    
    loader = template.TemplateLoader(['C:/code/interpreted_redesign/metadata/templates/'])
    template = loader.load('sppsz_all_template.xml')
    meta = GenerateMetadata()
    #collection = meta.create_metadata_file(table_name)
    #stream = template.generate(collection)
    stream = template.generate(collection=meta.create_metadata_file(table_name))
    out_fh = open(out_file, 'w')
    out_fh.write(stream.render('xml'))
    