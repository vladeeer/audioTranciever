import numpy as np

def calcLpFir(normCutoff, nTaps):
   assert nTaps % 2 != 0
   n = np.arange(nTaps)

   taps = np.sinc(normCutoff * (n - (nTaps - 1) / 2))

   window = np.hamming(nTaps)
   taps *= window

   taps /= np.sum(taps)
   return taps

def lpFilter(samples, normCutoff, nTaps):
   taps = calcLpFir(normCutoff, nTaps)
   filteredSamples = np.convolve(samples, taps, mode='same')
   return filteredSamples