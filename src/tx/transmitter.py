import numpy as np
import matplotlib.pyplot as plt

from ..common.params import pilotValue, nullValue, getModeParams, getModulationParams, getConsts
from ..common.filter import lpFilter
from ..common.pilotGen import PilotGen

class Transmitter():
   def __init__(self, mode):
      getConsts(self)
      getModeParams(self, mode)
      getModulationParams(self, self.modulation)

   def pad(self, samples):
      suffixLen = (self.audioSamplesPerSymbol - len(samples) % self.audioSamplesPerSymbol) % (self.audioSamplesPerSymbol * self.nMixedSymbolsPerFrame)
      return np.pad(samples, (0, suffixLen))

   def mapSamplesToQPSK(self, samples):
      iqSamplesPerSample = np.dtype(np.int16).itemsize * 8 // self.bitsPerElement
      iqSamples = np.zeros(iqSamplesPerSample * len(samples), dtype=np.complex64)
      for sampleIdx, sample in enumerate(samples):
         for elementIdx in range(iqSamplesPerSample):
            bits = sample & self.modulationBitMask
            sample = sample >> self.bitsPerElement
            iqSamples[sampleIdx * iqSamplesPerSample + elementIdx] = self.modulationMap[bits]
         #print(f'{iqSamples[sampleIdx * iqSamplesPerSample]}{iqSamples[sampleIdx * iqSamplesPerSample + 1]}{iqSamples[sampleIdx * iqSamplesPerSample + 2]}{iqSamples[sampleIdx * iqSamplesPerSample + 3]}{iqSamples[sampleIdx * iqSamplesPerSample + 4]}{iqSamples[sampleIdx * iqSamplesPerSample + 5]}{iqSamples[sampleIdx * iqSamplesPerSample + 6]}{iqSamples[sampleIdx * iqSamplesPerSample + 7]}')

      return iqSamples
   
   def mapToSymbIFFTcp(self, iqDataSamples):
      assert len(iqDataSamples) % self.nDataSubcarriers == 0
      nMixedSymbols = len(iqDataSamples) // self.nDataSubcarriers
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      iqIdx = 0
      pilotIdx = 0
      tdIqSamples = np.zeros((self.dftSize + self.cpLen) * nMixedSymbols, dtype=np.complex64)
      for symbIdx in range(nMixedSymbols):
         symb = np.zeros(nDataAndPilotSubcarriers, dtype=np.complex64)
         for subcIdx in range(nDataAndPilotSubcarriers):
            if subcIdx == pilotIdx:
               symb[subcIdx] = pilotValue
               pilotIdx = (pilotIdx + 5) % nDataAndPilotSubcarriers # Every fith is a pilot
            else:
               symb[subcIdx] = iqDataSamples[iqIdx]
               iqIdx += 1
         pilotIdx = (pilotIdx + 1) % 5 # pilots are shifted by one for each symbol
         symb = np.insert(symb, len(symb) // 2, nullValue) # insert null subcarrier
         #symb = np.pad(symb, (20, 20), constant_values=(np.complex64(0, 0), np.complex64(0, 0)))
         #plt.plot(np.abs(symb))
         #plt.savefig("symb.png")
         #plt.close()
         tdSymb = np.fft.ifft(np.fft.ifftshift(symb))
         tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : symbIdx * (self.dftSize + self.cpLen) + self.cpLen] = tdSymb[-self.cpLen:]
         tdIqSamples[symbIdx * (self.dftSize + self.cpLen) + self.cpLen : (symbIdx + 1) * (self.dftSize + self.cpLen)] = tdSymb[:]
         #plt.plot(np.abs(tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : (symbIdx + 1) * (self.dftSize + self.cpLen)]))
         #plt.savefig(f"tdSymb/tdSymb{symbIdx}.png")
         #plt.close()

      return (tdIqSamples, nMixedSymbols)

   def addPilotSymb(self, tdIqSamples, nFrames):
      symbLen = self.dftSize + self.cpLen
      frameLen = symbLen * self.nSymbolsPerFrame
      pilotlessFrameLen = symbLen * self.nMixedSymbolsPerFrame
      tdIqSamplesWithPilotSymb = np.zeros(len(tdIqSamples) + nFrames * symbLen, dtype=np.complex64)

      pilotGen = PilotGen(self.nDataSubcarriers, self.nPilotSubcarriers, self.nNullSubcarriers, self.cpLen, self.sampleRate)
      for frameIdx in range(nFrames): #frameIdx*pilotlessFrameLen
         tdIqSamplesWithPilotSymb[frameIdx * frameLen : frameIdx * frameLen + symbLen * 5] = \
            tdIqSamples[frameIdx * pilotlessFrameLen : frameIdx * pilotlessFrameLen + symbLen * 5]
         tdIqSamplesWithPilotSymb[frameIdx * frameLen + symbLen * 5 : frameIdx*frameLen + symbLen * 6] = \
            pilotGen.bbSymb[:]
         tdIqSamplesWithPilotSymb[frameIdx * frameLen + symbLen * 6 : frameIdx * frameLen + symbLen * 11] = \
            tdIqSamples[frameIdx * pilotlessFrameLen + symbLen * 5 : frameIdx * pilotlessFrameLen + symbLen * 10]
         
      return tdIqSamplesWithPilotSymb

   def resample(self, samples):
      upsampleFactor = 3
      downsampleFactor = 1
      nTaps = 8191
      assert upsampleFactor > downsampleFactor
      assert nTaps <= len(samples)

      upsampled = np.zeros(len(samples) * upsampleFactor, dtype=np.complex64)
      upsampled[::upsampleFactor] = samples
      upsampled = lpFilter(upsampled, 1/upsampleFactor, nTaps)
      
      downsampled = upsampled[::downsampleFactor]
      return downsampled

   def iqDac(self, tdIqSamples):
      timeVec = np.arange(len(tdIqSamples)) / self.sampleRate
      exponent = np.sqrt(2) * np.exp(1j * 2 * np.pi * self.centerFreq * timeVec)
      modulatedSamples = np.real(tdIqSamples * exponent)

      #percentile = np.percentile(np.abs(modulatedSamples), 99.999)
      #normFactor = np.iinfo(np.int16).max / percentile
      normFactor = np.iinfo(np.int16).max / np.max(np.abs(modulatedSamples))
      modulatedSamples *= normFactor
      #print(np.average(np.abs(modulatedSamples)))
      return modulatedSamples.astype(np.int16)

   def transmit(self, samples):
      print("------------------------------------")
      print("Adding padding")
      paddedSamples = self.pad(samples)

      print("Mapping samples")
      iqDataSamples = self.mapSamplesToQPSK(paddedSamples)

      print("Calculalating OFDM IQ samples")
      tdIqSamples, nMixedSymbols = self.mapToSymbIFFTcp(iqDataSamples)
      nFrames = nMixedSymbols // self.nMixedSymbolsPerFrame

      print("Adding pilot symbols")
      tdIqSamplesWithPilotSymb = self.addPilotSymb(tdIqSamples, nFrames)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamplesWithPilotSymb)

      print("Modulating baseband")
      tdSamples = self.iqDac(tdIqSamples)

      print(f"Transmitted {nFrames} frames")

      return tdSamples

