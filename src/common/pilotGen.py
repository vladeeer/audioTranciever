import numpy as np
import matplotlib.pyplot as plt

from ..common.params import nullValue, pilotValue

def getRand(n):
   return np.abs(n + 4) + np.abs((n - 100) ** 4)

class PilotGen:
   def __init__(self, nSubcarriers, nNullSubcarriers, cpLen):
      self.nSubcarriers = nSubcarriers
      self.nNullSubcarriers = nNullSubcarriers
      self.dftSize = self.nSubcarriers + nNullSubcarriers
      self.cpLen = cpLen

      self.genFdIq()
      self.genSymbol()

   def genFdIq(self):
      mult = 1 / (np.abs(pilotValue))
      subcIdxs = np.arange(self.nSubcarriers)
      self.fdSymb = pilotValue * np.exp(1j * 2 * np.pi * getRand(subcIdxs) / self.nSubcarriers) * mult
      self.fdSymbWithNull = np.insert(self.fdSymb, len(self.fdSymb) // 2, nullValue)
   
   def genSymbol(self):
      tdSymb = np.fft.ifft(np.fft.ifftshift(self.fdSymbWithNull))
      self.symbol = np.zeros(self.dftSize + self.cpLen, dtype=np.complex64)
      self.symbol[0 : self.cpLen] = tdSymb[-self.cpLen:]
      self.symbol[self.cpLen : self.cpLen + self.dftSize] = tdSymb[:]