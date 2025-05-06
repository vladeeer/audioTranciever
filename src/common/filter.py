import numpy as np

def calcLpFirTaps(normCutoff, nTaps):
   assert nTaps % 2 != 0
   n = (np.arange(nTaps) - (nTaps - 1) / 2).astype(np.float32)

   taps = np.sinc(normCutoff * n)

   window = np.hamming(nTaps)
   taps *= window

   taps /= np.sum(taps)

   return taps

def lpFilter(samples, normCutoff, nTaps):
   taps = calcLpFirTaps(normCutoff, nTaps)
   filteredSamples = np.convolve(samples, taps, mode='same')
   return filteredSamples

def calcHpFirTaps(normCutoff, nTaps):
   assert nTaps % 2 != 0
   n = (np.arange(nTaps) - (nTaps - 1) / 2).astype(np.float32)

   lpTaps = np.sinc(normCutoff * n) * normCutoff
   taps = np.zeros(nTaps, dtype=np.float32)
   taps[np.argmin(np.abs(n))] = 1
   taps = taps - lpTaps

   window = np.hamming(nTaps)
   taps *= window

   return taps

def hpFilter(samples, normCutoff, nTaps):
   groupDelay = (nTaps - 1) // 2
   taps = calcHpFirTaps(normCutoff, nTaps)
   filteredSamples = np.convolve(samples, taps, mode='same')
   filteredSamples = filteredSamples[groupDelay:]
   return filteredSamples.astype(np.float32)