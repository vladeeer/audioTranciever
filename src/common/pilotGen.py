import numpy as np
import matplotlib.pyplot as plt

from ..common.params import nullValue, pilotValue

def getRand(n):
   return np.abs(n + 4) + np.abs((n - 100) ** 4)

class PilotGen:
   def __init__(self, nDataSubcarriers, nPilotSubcarriers, nNullSubcarriers, cpLen):
      self.nDataSubcarriers = nDataSubcarriers
      self.nPilotSubcarriers = nPilotSubcarriers
      self.nNullSubcarriers = nNullSubcarriers
      self.nDataAndPilotSubcarriers = nDataSubcarriers + nPilotSubcarriers
      self.dftSize = self.nDataAndPilotSubcarriers + nNullSubcarriers
      self.cpLen = cpLen

      self.genFdIq()
      self.genSymbol()

   def genFdIq(self):
      mult = (np.abs(pilotValue)**2 * self.nPilotSubcarriers + self.nDataSubcarriers) / (np.abs(pilotValue)**2 * self.nDataAndPilotSubcarriers)
      subcIdxs = np.arange(self.nDataAndPilotSubcarriers)
      self.fdSymb = pilotValue * np.exp(1j * 2 * np.pi * getRand(subcIdxs) / self.nDataAndPilotSubcarriers) * mult
      self.fdSymb = np.insert(self.fdSymb, len(self.fdSymb) // 2, nullValue)
   
   def genSymbol(self):
      tdSymb = np.fft.ifft(np.fft.ifftshift(self.fdSymb))
      self.symbol = np.zeros(self.dftSize + self.cpLen, dtype=np.complex64)
      self.symbol[0 : self.cpLen] = tdSymb[-self.cpLen:]
      self.symbol[self.cpLen : self.cpLen + self.dftSize] = tdSymb[:]