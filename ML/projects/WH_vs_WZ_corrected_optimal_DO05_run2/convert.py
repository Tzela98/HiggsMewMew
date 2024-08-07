import numpy as np
import ROOT
from array import array

bins = np.loadtxt('bins.txt')
signal = np.loadtxt('signal.txt')
background = np.loadtxt('background.txt')

h_signal = ROOT.TH1D("h_signal", "", len(bins)-1, array('d', bins))
h_background = h_signal.Clone('h_background')

for i in range(len(bins)-1):
    h_signal.SetBinContent(i+1, signal[i])
    h_background.SetBinContent(i+1, background[i])

tf = ROOT.TFile('histograms.root', 'recreate')
h_signal.Write()
h_background.Write()
tf.Close()
