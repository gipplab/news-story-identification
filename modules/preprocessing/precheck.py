from configparser import ConfigParser
from os.path import exists
from os import makedirs

def config_make_folders(config: ConfigParser):
    folder_paths = [
        config['GLOBAL']['InputFolder'],
        config['GLOBAL']['OutputFolder'],
        config['TASK-3']['SourceFolder'],
        config['TASK-3']['SourceDownloadedFolder'],
        config['TASK-4']['SuspFolder'],
        config['TASK-4']['SrcFolder'],
        config['TASK-4']['OutputFolder']
    ]
    for p in folder_paths:
        if not exists(p):
            makedirs(p)
    return True

def config_validation(config: ConfigParser):
    if not config.has_option('GLOBAL', 'InputFolder'):
        return False
    if not config.has_option('GLOBAL', 'OutputFolder'):
        return False
    return True

def config_check_corenlp(config: ConfigParser):
    return True

def validate(config: ConfigParser):
    validate_result = config_validation(config)
    validate_result &= config_make_folders(config) 
    validate_result &= config_check_corenlp(config)
    return validate_result

