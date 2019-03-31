# -*- coding: utf-8 -*-
import click
import logging
import os
import json
import zipfile
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('downloading data from the internet')
    if not os.path.exists('../../data'): #check if path exists
        os.mkdir('../../data')
        print("data folder created")

    download_path = os.path.join('../../data', 'evaldata.zip')
    if not os.path.exists(download_path):
        print('downloading data to %s ...' % download_path)
        source = 'http://groups.inf.ed.ac.uk/cup/naturalize/data/evaldata.zip'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')

    unzip_path = os.path.join('../../data', 'raw')
    if not os.path.exists(unzip_path):
        print('extracting data to %s ...' % unzip_path)
        archive = zipfile.ZipFile(download_path)
        for file in archive.namelist():
            if file.startswith('json/'):
                archive.extract(file, unzip_path)
        print('finished extracting')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
