import urllib2
import zipfile
import os


class DatasetImporter:
    """
    Import the dataset, and prepare it as a coo_matrix, i.e. a list of tuples [(row, col, val), ...]
    """

    zip_file = '/tmp/ml-100k.zip'
    data_file = '/tmp/ml-100k/u.data'

    def __init__(self):
        if not self.check_if_file_exists():
            print "Downloading the dataset"
            self.download_and_unzip()

        print "Parsing the dataset"
        self.dataset = list(self.read_lines(self.data_file))

    def download_and_unzip(self):
        response = urllib2.urlopen('http://files.grouplens.org/datasets/movielens/ml-100k.zip')

        with open(self.zip_file, 'w') as fh:
            fh.write(response.read())

        with zipfile.ZipFile(self.zip_file) as zfh:
            zfh.extractall('/tmp/')

    def check_if_file_exists(self):
        if os.path.isfile(self.data_file) and os.access(self.data_file, os.R_OK):
            return True
        else:
            return False

    @staticmethod
    def read_lines(data_file):
        with open(data_file, "r") as fh:
            lines = fh.readlines()

        for line in lines:
            chunk = line.split("\t")
            yield int(chunk[0])-1, int(chunk[1])-1, float(chunk[2])