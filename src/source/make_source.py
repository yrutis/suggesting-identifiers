# -*- coding: utf-8 -*-
import logging
import os
import zipfile
import urllib.request


def main():
    """ Downlaods Data from the internet.
    """
    print('downloading data from the internet')
    logger = logging.getLogger(__name__)
    logger.info('downloading data from the internet')

    #retrieve data folder based on location of current file
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    print(data_folder)


    if not os.path.exists(data_folder): #check if path exists
        os.mkdir(data_folder)
        print("data folder created")


    download_path = os.path.join(data_folder, 'evaldata.zip')

    if not os.path.exists(download_path):
        print('downloading data to %s ...' % download_path)
        source = 'http://groups.inf.ed.ac.uk/cup/naturalize/data/evaldata.zip'
        urllib.request.urlretrieve(source, download_path)
        print('finished downloading')

    logger.info("finished with downloading")


    unzip_path = os.path.join(data_folder, 'raw')
    if not os.path.exists(unzip_path):
        print('extracting data to %s ...' % unzip_path)
        archive = zipfile.ZipFile(download_path)
        for file in archive.namelist():
            if file.startswith('json/'):
                archive.extract(file, unzip_path)
        print('finished extracting')

    logger.info("finished with extracting")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
