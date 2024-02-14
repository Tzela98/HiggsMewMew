from utility import parse_txt_file
from utility import open_multiple_paths
from utility import total_lines_in_csv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    csv_path_DYJetsToLL_M_50 = ['/work/ehettwer/HiggsMewMew/data_csv/DYJetsToLL_M-50_mc_2018.csv']
    csv_path_ST_t_channel_top_5f = ['/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_top_5f_mc_2018.csv']
    csv_path_ST_t_channel_antitop_5f = ['/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_antitop_5f_mc_2018.csv']
    csv_path_ST_tW_antitop_5f_inclusiveDecays = ['/work/ehettwer/HiggsMewMew/data_csv/ST_tW_antitop_5f_inclusiveDecays_mc_2018.csv']
    csv_path_ST_tW_top_5f_inclusiveDecays = ['/work/ehettwer/HiggsMewMew/data_csv/ST_tW_top_5f_inclusiveDecays_mc_2018.csv']
    csv_path_TTToSemiLeptonic = ['/work/ehettwer/HiggsMewMew/data_csv/TTToSemiLeptonic_mc_2018.csv']
    csv_path_WWTo2L2Nu = ['/work/ehettwer/HiggsMewMew/data_csv/WWTo2L2Nu_mc_2018.csv']

    log_path_DYJetsToLL_M_50 = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/DYJetsToLL_M-50*/*.txt']
    log_path_ST_t_channel_top_5f = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/ST_t-channel_top_5f*/*.txt']
    log_path_ST_t_channel_antitop_5f = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/ST_t-channel_antitop_5f*/*.txt']
    log_path_ST_tW_antitop_5f_inclusiveDecays = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/ST_tW_antitop_5f_inclusiveDecays*/*.txt']
    log_path_ST_tW_top_5f_inclusiveDecays = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/ST_tW_top_5f_inclusiveDecays*/*.txt']
    log_path_TTToSemiLeptonic = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/TTToSemiLeptonic*/*.txt']
    log_path_WWTo2L2Nu = ['/work/ehettwer/KingMaker/data/logs/background_hmm_2018_mc/Output/WWTo2L2Nu*/*.txt']

    list_of_log_paths_DYJetsToLL_M_50 = open_multiple_paths(log_path_DYJetsToLL_M_50)
    list_of_log_paths_ST_t_channel_top_5f = open_multiple_paths(log_path_ST_t_channel_top_5f)
    list_of_log_paths_ST_t_channel_antitop_5f = open_multiple_paths(log_path_ST_t_channel_antitop_5f)
    list_of_log_paths_ST_tW_antitop_5f_inclusiveDecays = open_multiple_paths(log_path_ST_tW_antitop_5f_inclusiveDecays)
    list_of_log_paths_ST_tW_top_5f_inclusiveDecays = open_multiple_paths(log_path_ST_tW_top_5f_inclusiveDecays)
    list_of_log_paths_TTToSemiLeptonic = open_multiple_paths(log_path_TTToSemiLeptonic)
    list_of_log_paths_WWTo2L2Nu = open_multiple_paths(log_path_WWTo2L2Nu)

    print('-----------------')
    print('selection passed:')
    
    processed_events_DYJetsToLL_M_50 = total_lines_in_csv(csv_path_DYJetsToLL_M_50)
    print('number of processed events DYJetsToLL_M_50:', processed_events_DYJetsToLL_M_50)
    processed_events_ST_t_channel_top_5f = total_lines_in_csv(csv_path_ST_t_channel_top_5f)
    print('number of processed events ST_t_channel_top_5f:', processed_events_ST_t_channel_top_5f)
    processed_events_ST_t_channel_antitop_5f = total_lines_in_csv(csv_path_ST_t_channel_antitop_5f)
    print('number of processed events ST_t_channel_antitop_5f:', processed_events_ST_t_channel_antitop_5f)
    processed_events_ST_tW_antitop_5f_inclusiveDecays = total_lines_in_csv(csv_path_ST_tW_antitop_5f_inclusiveDecays)
    print('number of processed events ST_tW_antitop_5f_inclusiveDecays:', processed_events_ST_tW_antitop_5f_inclusiveDecays)
    processed_events_ST_tW_top_5f_inclusiveDecays = total_lines_in_csv(csv_path_ST_tW_top_5f_inclusiveDecays)
    print('number of processed events ST_tW_top_5f_inclusiveDecays:', processed_events_ST_tW_top_5f_inclusiveDecays)
    processed_events_TTToSemiLeptonic = total_lines_in_csv(csv_path_TTToSemiLeptonic)
    print('number of processed events TTToSemiLeptonic:', processed_events_TTToSemiLeptonic)
    processed_events_WWTo2L2Nu = total_lines_in_csv(csv_path_WWTo2L2Nu)
    print('number of processed events WWTo2L2Nu:', processed_events_WWTo2L2Nu)

    print('-----------------')
    print('before selection:')
    
    total_events_DYJetsToLL_M_50 = []
    for log_file in tqdm(list_of_log_paths_DYJetsToLL_M_50):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_DYJetsToLL_M_50.append(events)
    number_of_DYJetsToLL_M_50_events = sum(total_events_DYJetsToLL_M_50)
    print('number of DYJetsToLL_M_50 events:', number_of_DYJetsToLL_M_50_events)


    total_events_ST_t_channel_top_5f = []
    for log_file in tqdm(list_of_log_paths_ST_t_channel_top_5f):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_ST_t_channel_top_5f.append(events)
    number_of_ST_t_channel_top_5f_events = sum(total_events_ST_t_channel_top_5f)
    print('number of ST_t_channel_top_5f events:', number_of_ST_t_channel_top_5f_events)

    total_events_ST_t_channel_antitop_5f = []
    for log_file in tqdm(list_of_log_paths_ST_t_channel_antitop_5f):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_ST_t_channel_antitop_5f.append(events)
    number_of_ST_t_channel_antitop_5f_events = sum(total_events_ST_t_channel_antitop_5f)
    print('number of ST_t_channel_antitop_5f events:', number_of_ST_t_channel_antitop_5f_events)

    total_events_ST_tW_antitop_5f_inclusiveDecays = []
    for log_file in tqdm(list_of_log_paths_ST_tW_antitop_5f_inclusiveDecays):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_ST_tW_antitop_5f_inclusiveDecays.append(events)
    number_of_ST_tW_antitop_5f_inclusiveDecays_events = sum(total_events_ST_tW_antitop_5f_inclusiveDecays)
    print('number of ST_tW_antitop_5f_inclusiveDecays events:', number_of_ST_tW_antitop_5f_inclusiveDecays_events)

    total_events_ST_tW_top_5f_inclusiveDecays = []
    for log_file in tqdm(list_of_log_paths_ST_tW_top_5f_inclusiveDecays):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_ST_tW_top_5f_inclusiveDecays.append(events)
    number_of_ST_tW_top_5f_inclusiveDecays_events = sum(total_events_ST_tW_top_5f_inclusiveDecays)
    print('number of ST_tW_top_5f_inclusiveDecays events:', number_of_ST_tW_top_5f_inclusiveDecays_events)

    total_events_TTToSemiLeptonic = []
    for log_file in tqdm(list_of_log_paths_TTToSemiLeptonic):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_TTToSemiLeptonic.append(events)
    number_of_TTToSemiLeptonic_events = sum(total_events_TTToSemiLeptonic)
    print('number of TTToSemiLeptonic events:', number_of_TTToSemiLeptonic_events)

    total_events_WWTo2L2Nu = []
    for log_file in tqdm(list_of_log_paths_WWTo2L2Nu):
        events = parse_txt_file(log_file)
        if events is not None:
            total_events_WWTo2L2Nu.append(events)
    number_of_WWTo2L2Nu_events = sum(total_events_WWTo2L2Nu)
    print('number of WWTo2L2Nu events:', number_of_WWTo2L2Nu_events)

    print('-----------------')
    print('total number of events:')

    print('total number of DYJetsToLL_M_50 events:', number_of_DYJetsToLL_M_50_events)
    print('total number of ST_t_channel_top_5f events:', number_of_ST_t_channel_top_5f_events)
    print('total number of ST_t_channel_antitop_5f events:', number_of_ST_t_channel_antitop_5f_events)
    print('total number of ST_tW_antitop_5f_inclusiveDecays events:', number_of_ST_tW_antitop_5f_inclusiveDecays_events)
    print('total number of ST_tW_top_5f_inclusiveDecays events:', number_of_ST_tW_top_5f_inclusiveDecays_events)
    print('total number of TTToSemiLeptonic events:', number_of_TTToSemiLeptonic_events)
    print('total number of WWTo2L2Nu events:', number_of_WWTo2L2Nu_events)

    print('-----------------')
    print('scale factors:')

    scale_factor_DYJetsToLL_M_50 = processed_events_DYJetsToLL_M_50 / number_of_DYJetsToLL_M_50_events 
    print('scale factor DYJetsToLL_M_50:', scale_factor_DYJetsToLL_M_50)
    scale_factor_ST_t_channel_top_5f = processed_events_ST_t_channel_top_5f / number_of_ST_t_channel_top_5f_events
    print('scale factor ST_t_channel_top_5f:', scale_factor_ST_t_channel_top_5f)
    scale_factor_ST_t_channel_antitop_5f = processed_events_ST_t_channel_antitop_5f / number_of_ST_t_channel_antitop_5f_events
    print('scale factor ST_t_channel_antitop_5f:', scale_factor_ST_t_channel_antitop_5f)
    scale_factor_ST_tW_antitop_5f_inclusiveDecays = processed_events_ST_tW_antitop_5f_inclusiveDecays / number_of_ST_tW_antitop_5f_inclusiveDecays_events
    print('scale factor ST_tW_antitop_5f_inclusiveDecays:', scale_factor_ST_tW_antitop_5f_inclusiveDecays)
    scale_factor_ST_tW_top_5f_inclusiveDecays = processed_events_ST_tW_top_5f_inclusiveDecays / number_of_ST_tW_top_5f_inclusiveDecays_events
    print('scale factor ST_tW_top_5f_inclusiveDecays:', scale_factor_ST_tW_top_5f_inclusiveDecays)
    scale_factor_TTToSemiLeptonic = processed_events_TTToSemiLeptonic / number_of_TTToSemiLeptonic_events
    print('scale factor TTToSemiLeptonic:', scale_factor_TTToSemiLeptonic)
    scale_factor_WWTo2L2Nu = processed_events_WWTo2L2Nu / number_of_WWTo2L2Nu_events
    print('scale factor WWTo2L2Nu:', scale_factor_WWTo2L2Nu)


    data_dict = ['DYJetsToLL_M_50', 'TTToSemiLeptonic', 'ST_t_channel_top_5f', 'ST_t_channel_antitop_5f', 
                 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'WWTo2L2Nu']
    total_events = np.array([number_of_DYJetsToLL_M_50_events, number_of_TTToSemiLeptonic_events, number_of_ST_t_channel_top_5f_events, number_of_ST_t_channel_antitop_5f_events, 
                                number_of_ST_tW_antitop_5f_inclusiveDecays_events, number_of_ST_tW_top_5f_inclusiveDecays_events, number_of_WWTo2L2Nu_events])
    scale_factors = np.array([scale_factor_DYJetsToLL_M_50, scale_factor_TTToSemiLeptonic, scale_factor_ST_t_channel_top_5f, scale_factor_ST_t_channel_antitop_5f, 
                     scale_factor_ST_tW_antitop_5f_inclusiveDecays, scale_factor_ST_tW_top_5f_inclusiveDecays, scale_factor_WWTo2L2Nu])
    cross_sections = np.array([6225.4, 358.57, 136.02, 80.95, 35.9, 35.9, 12.178])
    weights = np.multiply(scale_factors, cross_sections)

    df_mc_2018 = pd.DataFrame(data=[total_events, scale_factors, cross_sections, weights], columns=data_dict, index=['total_events', 'scale factors', 'cross sections', 'weights'])
    df_mc_2018.to_csv('data_csv/mc_2018.csv')

    print(df_mc_2018.head(n=5))    
    