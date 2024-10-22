import numpy as np
import ROOT
from array import array

bins = np.loadtxt('bins_ML.txt')
signal = np.loadtxt('signal_ML.txt')
background = np.loadtxt('background_ML.txt')
#data = np.loadtxt('histogram_data.txt')
#all_backgrounds = np.loadtxt('all_backgrounds.txt')

h_signal = ROOT.TH1D("h_signal", "", len(bins)-1, array('d', bins))
h_background = h_signal.Clone('h_background')
#h_data = h_signal.Clone('h_data')
#h_all_backgrounds = h_signal.Clone('h_all_backgrounds')

for i in range(len(bins)-1):
    h_signal.SetBinContent(i+1, signal[i])
    h_background.SetBinContent(i+1, background[i])
    #h_data.SetBinContent(i+1, data[i])
    #h_all_backgrounds.SetBinContent(i+1, all_backgrounds[i])

tf = ROOT.TFile('histograms_cut_05to1_8bins.root', 'recreate')
h_signal.Write()
h_background.Write()
#h_data.Write()
#h_all_backgrounds.Write()
tf.Close()
