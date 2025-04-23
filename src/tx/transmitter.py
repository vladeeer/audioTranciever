import numpy as np
import matplotlib.pyplot as plt

pilotValue = np.complex64(1, 0)

qpskMap = np.array([
   -0.7071 + 0.7071j, #0x00
    0.7071 + 0.7071j, #0x01
   -0.7071 - 0.7071j, #0x10
    0.7071 - 0.7071j, #0x11
], dtype=np.complex64)

def calcLpFir(normCutoff, nTaps):
   n = np.arange(nTaps)

   taps = np.sinc(2 * np.pi * normCutoff * (n - (nTaps - 1) / 2))

   window = np.hamming(nTaps)
   taps *= window

   taps /= np.sum(taps)
   return taps

def lpFilter(samples, normCutoff, nTaps):
   taps = calcLpFir(normCutoff, nTaps)
   filteredSamples = np.convolve(samples, taps, mode='same')
   return filteredSamples

class Transmitter():
   def __init__(self, modulationScheme):
      match modulationScheme:
         # case 0: # 4 pilots, 1 audio sample per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 20
         # case 1: # 8 pilots, 2 audio samples per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 40
         # case 2: # 16 pilots, 4 audio samples per symbol
         #    self.modulation = "BPSK"
         #    self.nSubcarriers = 80
         case 3: # 4 pilots, 2 audio samples per symbol
            self.modulation = "QPSK"
            self.nDataSubcarriers = 16
            self.nPilotSubcarriers = 4
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 2
         case 4: # 8 pilots, 4 audio samples per symbol
            self.modulation = "QPSK"
            self.nDataSubcarriers = 32
            self.nPilotSubcarriers = 8
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 4
         case 5: # 16 pilots, 8 audio samples per symbol
            self.modulation = "QPSK"
            self.nDataSubcarriers = 64
            self.nPilotSubcarriers = 16
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 8
            
      match self.modulation:
         case "QPSK":
            self.modulationMap = qpskMap
            self.bitsPerElement = 2

      #self.symbolsPerFrame = 10 #???
      self.nSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers + self.nNullSubcarriers
      self.dftSize = self.nSubcarriers
      self.cpLen = 20
      self.bw = 14700
      self.centerFreq = 11000
      self.sampleRate = 44100 # = 14700 * 3

   def pad(self, samples):
      suffixLen = (self.audioSamplesPerSymbol - len(samples) % self.audioSamplesPerSymbol) % self.audioSamplesPerSymbol
      return np.pad(samples, (0, suffixLen))

   def mapSamplesToQPSK(self, samples):
      iqSamplesPerSample = np.dtype(np.int16).itemsize * 8 // self.bitsPerElement
      iqSamples = np.zeros(iqSamplesPerSample * len(samples), dtype=np.complex64)
      for sampleIdx, sample in enumerate(samples):
         for elementIdx in range(iqSamplesPerSample):
            bits = sample & 0x0003
            sample = sample >> self.bitsPerElement
            iqSamples[sampleIdx * iqSamplesPerSample + elementIdx] = self.modulationMap[bits]
         #print(f'{iqSamples[sampleIdx * iqSamplesPerSample]}{iqSamples[sampleIdx * iqSamplesPerSample + 1]}{iqSamples[sampleIdx * iqSamplesPerSample + 2]}{iqSamples[sampleIdx * iqSamplesPerSample + 3]}{iqSamples[sampleIdx * iqSamplesPerSample + 4]}{iqSamples[sampleIdx * iqSamplesPerSample + 5]}{iqSamples[sampleIdx * iqSamplesPerSample + 6]}{iqSamples[sampleIdx * iqSamplesPerSample + 7]}')

      return iqSamples
   
   def mapToSymbIFFTcp(self, iqDataSamples):
      assert len(iqDataSamples) % self.nDataSubcarriers == 0
      nSymbols = len(iqDataSamples) // self.nDataSubcarriers
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      iqIdx = 0
      pilotIdx = 0
      tdIqSamples = np.zeros((self.dftSize + self.cpLen) * nSymbols, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         symb = np.zeros(nDataAndPilotSubcarriers, dtype=np.complex64)
         for subcIdx in range(nDataAndPilotSubcarriers):
            if subcIdx == pilotIdx:
               symb[subcIdx] = pilotValue
               pilotIdx = (pilotIdx + 5) % nDataAndPilotSubcarriers # Every fith is a pilot
            else:
               symb[subcIdx] = iqDataSamples[iqIdx] # add NULLLLLLLLLLLLLLLLLLLLLLLLLLL
               iqIdx += 1
         pilotIdx = (pilotIdx + 1) % 5 # pilots are shifted by one for each symbol
         symb = np.insert(symb, len(symb) // 2, np.complex64(0, 0)) # insert null subcarrier
         #symb = np.pad(symb, (20, 20), constant_values=(np.complex64(0, 0), np.complex64(0, 0)))
         #plt.plot(np.abs(symb))
         #plt.savefig("symb.png")
         #plt.close()
         tdSymb = np.fft.ifft(symb)
         tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : symbIdx * (self.dftSize + self.cpLen) + self.cpLen] = tdSymb[-self.cpLen:]
         tdIqSamples[symbIdx * (self.dftSize + self.cpLen) + self.cpLen : (symbIdx + 1) * (self.dftSize + self.cpLen)] = tdSymb[:]
         #plt.plot(np.abs(tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : (symbIdx + 1) * (self.dftSize + self.cpLen)]))
         #plt.savefig(f"tdSymb/tdSymb{symbIdx}.png")
         #plt.close()
      return tdIqSamples

   def resample(self, samples):
      # upsample x5, filter, downsample x2
      upsampleFactor = 5
      downsampleFactor = 2
      nTaps = 128
      assert upsampleFactor > downsampleFactor
      assert nTaps <= len(samples)

      if upsampleFactor > 1:
         upsampled = np.zeros(len(samples) * upsampleFactor, dtype=np.complex64)
         upsampled[::upsampleFactor] = samples
         upsampled = lpFilter(upsampled, 1/upsampleFactor, nTaps)
      else:
         upsampled = samples
      
      downsampled = upsampled[::downsampleFactor]
      return downsampled

   def dac(self, tdIqSamples):
      timeVec = np.arange(len(tdIqSamples)) / self.sampleRate
      #carrierCos = np.cos(2 * np.pi * self.centerFreq * timeVec)
      #carrierSin = np.sin(2 * np.pi * self.centerFreq * timeVec)

      exponent = np.sqrt(2) * np.exp(1j * 2 * np.pi * self.centerFreq * timeVec)

      modulatedSamples = np.real(tdIqSamples * exponent)
      #modulatedSamples = tdIqSamples.real * carrierCos + tdIqSamples.imag * carrierSin

      normFactor = np.iinfo(np.int16).max / np.max(np.abs(modulatedSamples))
      modulatedSamples *= normFactor
      print(np.average(np.abs(modulatedSamples)))
      return modulatedSamples.astype(np.int16)

   def transmit(self, samples):
      print("Adding padding")
      paddedSamples = self.pad(samples)

      print("Mapping samples")
      iqDataSamples = self.mapSamplesToQPSK(paddedSamples)

      print("Calculalating OFDM IQ samples")
      tdIqSamples = self.mapToSymbIFFTcp(iqDataSamples)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamples)

      print("Modulating baseband")
      tdSamples = self.dac(tdIqSamples)

      return tdSamples

