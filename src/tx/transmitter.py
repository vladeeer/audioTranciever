import numpy as np

class Transmitter():
   def __init__(self, modulationScheme):
      self.symbolsPerFrame = 10
      
      self.modulationScheme = modulationScheme
      match self.modulationScheme:
         # case 0: # 4 pilots, 1 audio sample per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 20
         # case 1: # 8 pilots, 2 audio samples per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 40
         # case 2: # 16 pilots, 4 audio samples per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 80
         # case 3: # 4 pilots, 2 audio samples per symbol
         #    self.modulation = "QPSK"
         #    self.nSubcarriers = 20
         # case 4: # 8 pilots, 4 audio samples per symbol
         #    self.modulation = "QPSK"
         #    self.nSubcarriers = 40
         case 5: # 16 pilots, 8 audio samples per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 81 # 64 data + 16 pilots + 1 null
            self.audioSamplesPerSymbol = 8

      self.samplingRate = 44100
      self.subcarrierSpacing = 544.444444444
      self.dftSize = self.nSubcarriers

   def pad(self):
      suffixLen = (self.audioSamplesPerSymbol - len(self.tdSamples) % self.audioSamplesPerSymbol) % self.audioSamplesPerSymbol
      self.tdSamples = np.pad(self.tdSamples, (0, suffixLen))

   def transmit(self, samples):
      self.tdSamples = samples

      self.pad()


      return self.tdSamples

