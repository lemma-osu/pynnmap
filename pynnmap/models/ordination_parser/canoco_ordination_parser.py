from models.ordination_parser import ordination_parser


class CanocoParser(ordination_parser.OrdinationParser):

    def __init__(self):
        super(CanocoParser, self).__init__(self)

    def parse(self):
        print 'CanocoParser'
