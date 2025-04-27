import numpy as np
import matplotlib.pyplot as plt

from ..common.params import nullValue

class PilotGen:
   def __init__(self, nSubcarriers, cpLen):
      self.nSubcarriers = nSubcarriers
      self.cpLen = cpLen

      self.genFdIq()
      self.genSymbol()

   def genFdIq(self):
      subcIdxs = np.arange(self.nSubcarriers)
      self.fdIq = np.sqrt(2)*np.exp(1j * 2 * np.pi * subcIdxs / self.nSubcarriers)
   
   def genSymbol(self):
      fdSymb = np.insert(self.fdIq, len(self.fdIq) // 2, nullValue)
      tdSymb = np.fft.ifft(np.fft.ifftshift(fdSymb))
      self.symbol = np.zeros(self.nSubcarriers + self.cpLen, dtype=np.complex64)
      self.symbol[0 : self.cpLen] = tdSymb[-self.cpLen:]
      self.symbol[self.cpLen : self.cpLen + self.dftSize] = tdSymb[:]