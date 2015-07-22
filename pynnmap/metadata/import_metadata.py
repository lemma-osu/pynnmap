import copy
import os 
import sys
from lxml import etree
from lxml import objectify
from models.parser import xml_parser as xp

class ImportMetadata(object):
    def __init__(self, input_file, root, mosaics):
        self.input_file = input_file
        self.root = root
        self.mosaics = mosaics
    
    def import_metadata(self, destination, model_year):
        x = xp.XMLParser(self.input_file)
        y = copy.deepcopy(x)
        y.root.idinfo.timeperd.timeinfo.sngdate.caldate = model_year 
        objectify.deannotate(y.tree)
        etree.cleanup_namespaces(y.tree)
        output_file = destination + '/metadata.xml'
        y.tree.write(output_file, pretty_print=True)

def main():
    # first argument is the metadata template file
    input_file = sys.argv[1]
    
    # second argument is the parent model folder
    root = sys.argv[2]
    
    # third argument indicates whether the metadata is to be
    # imported to mosaics or original modeling files
    mosaics = sys.argv[3]
    
    meta = ImportMetadata(input_file, root, mosaics)
    
    if mosaics == '1':
        # all mosaics are stored in the root folder
        for root, dirs, files in os.walk(root):
            for name in dirs:
                if name <> 'info':
                    # extract model year from the directory name
                    year = name[-4:]
                    # set the destination directory
                    destination = root + '/' + name
                    # copy metadata and set correct model year
                    meta.import_metadata(destination, year)
    else:
        # original models are stored in subfolders named by year
        pass 

if __name__ == '__main__':
    main()
