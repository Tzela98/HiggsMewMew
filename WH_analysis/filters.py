import numpy as np
import pandas as pd


def filter_for_jet_mass(df, threshold_mass):
    return df[df.mjj > threshold_mass]


def filter_pseudo_rapidity_separation(df, threshold_rapidity):
    return df[np.abs(df.jeta_1 - df.jeta_2) > threshold_rapidity]


def jet_selection(df, leading_jet_pt = 35, sub_leading_jet_pt = 25):
    if 'jpt_1' in df.columns and 'jpt_2' in df.columns:
        condition_1 = df['jpt_1'] > leading_jet_pt
        condition_2 = df['jpt_2'] > sub_leading_jet_pt

        filtered_df = df[condition_1 & condition_2]
        return filtered_df
    
    else: raise ValueError("Columns 'jpt_1' and 'jpt_2' are not in the DataFrame.")


def trigger_var_selection(df):
    condition = df['trg_single_mu24'] == 1
    return df[condition]


def selection_pipeline_trg(df, leading_jet_pt: int, sub_leading_jet_pt: int, threshold_mass: int, threshold_rapidity: float):
    return trigger_var_selection(filter_pseudo_rapidity_separation(filter_for_jet_mass(jet_selection(df, leading_jet_pt, sub_leading_jet_pt), threshold_mass), threshold_rapidity))


def selection_pipeline(df, leading_jet_pt: int, sub_leading_jet_pt: int, threshold_mass: int, threshold_rapidity: float):
    return filter_pseudo_rapidity_separation(filter_for_jet_mass(jet_selection(df, leading_jet_pt, sub_leading_jet_pt), threshold_mass), threshold_rapidity)

