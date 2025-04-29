import numpy as np
import matplotlib.pyplot as plt

from ..common.filter import lpFilter
from ..common.params import nullValue, pilotValue

def getRand(n):
   return np.abs(n + 4) + np.abs((n - 100) ** 4)

class PilotGen:
   def __init__(self, nDataSubcarriers, nPilotSubcarriers, nNullSubcarriers, cpLen, sampleRate):
      self.nDataSubcarriers = nDataSubcarriers
      self.nPilotSubcarriers = nPilotSubcarriers
      self.nNullSubcarriers = nNullSubcarriers
      self.nDataAndPilotSubcarriers = nDataSubcarriers + nPilotSubcarriers
      self.dftSize = self.nDataAndPilotSubcarriers + nNullSubcarriers
      self.cpLen = cpLen
      self.sampleRate = sampleRate

      self.upsampleFactor = 3
      self.nTaps = 8191

      self.genFdIq()
      self.genBbSymbol()

   def genFdIq(self):
      mult = (np.abs(pilotValue)**2 * self.nPilotSubcarriers + self.nDataSubcarriers) / (np.abs(pilotValue)**2 * self.nDataAndPilotSubcarriers)
      subcIdxs = np.arange(self.nDataAndPilotSubcarriers)
      self.fdSymb = pilotValue * np.exp(1j * 2 * np.pi * getRand(subcIdxs) / self.nDataAndPilotSubcarriers) * mult
      self.fdSymb = np.insert(self.fdSymb, len(self.fdSymb) // 2, nullValue)
   
   def genBbSymbol(self):
      tdSymb = np.fft.ifft(np.fft.ifftshift(self.fdSymb))
      self.bbSymb = np.zeros(self.dftSize + self.cpLen, dtype=np.complex64)
      self.bbSymb[0 : self.cpLen] = tdSymb[-self.cpLen:]
      self.bbSymb[self.cpLen : self.cpLen + self.dftSize] = tdSymb[:]

   def genTdSymbol(self):
      upsampled = np.zeros(len(self.bbSymb) * self.upsampleFactor, dtype=np.complex64)
      upsampled[::self.upsampleFactor] = self.bbSymb
      upsampled = lpFilter(upsampled, 1/self.upsampleFactor, self.nTaps)

      timeVec = np.arange(len(upsampled)) / self.sampleRate
      exponent = np.sqrt(2) * np.exp(1j * 2 * np.pi * self.centerFreq * timeVec)
      self.symb = np.real(upsampled * exponent)
