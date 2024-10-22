import numpy as np
import ROOT
from array import array

bins = np.loadtxt('bins_ML.txt')
signal = np.loadtxt('signal_ML.txt')
background = np.loadtxt('background_ML.txt')
#data = np.loadtxt('histogram_data.txt')

h_signal = ROOT.TH1D("h_signal", "", len(bins)-1, array('d', bins))
h_background = h_signal.Clone('h_background')
#h_data = h_signal.Clone('h_data')

for i in range(len(bins)-1):
    h_signal.SetBinContent(i+1, signal[i])
    h_background.SetBinContent(i+1, background[i])
    #h_data.SetBinContent(i+1, data[i])

tf = ROOT.TFile('histograms_vbf_ML_dimuon05to1.root', 'recreate')
h_signal.Write()
h_background.Write()
#h_data.Write()
tf.Close()
