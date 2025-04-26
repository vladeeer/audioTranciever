import numpy as np
import matplotlib.pyplot as plt

from ..common.params import pilotValue, nullValue, getModeParams, getModulationParams, getConsts
from ..common.filter import lpFilter
from ..common.utils import *

class Receiver():
   def __init__(self, mode):
      getConsts(self)
      getModeParams(self, mode)
      getModulationParams(self, self.modulation)

   def iqDac(self, modulatedSamples):
      modulatedSamples = modulatedSamples.astype(np.complex64)

      timeVec = np.arange(len(modulatedSamples)) / self.sampleRate
      exponent = np.sqrt(2) * np.exp(-1j * 2 * np.pi * self.centerFreq * timeVec)
      demodulatedSamples = modulatedSamples * exponent

      #print(np.average(np.abs(modulatedSamples)))
      #return np.real(demodulatedSamples).astype(np.int16)
      return demodulatedSamples

   def resample(self, samples):
      upsampleFactor = 1
      downsampleFactor = 3
      nTaps = 1023
      assert upsampleFactor < downsampleFactor
      assert nTaps <= len(samples)

      upsampled = np.zeros(len(samples) * upsampleFactor, dtype=np.complex64)
      upsampled[::upsampleFactor] = samples
      upsampled = lpFilter(upsampled, 1/downsampleFactor, nTaps)
      
      downsampled = upsampled[::downsampleFactor]
      return downsampled

   def detachCp(self, samples, nSymbols):
      tdIqSamples = np.ndarray(nSymbols * self.dftSize, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         tdIqSamples[symbIdx*self.dftSize : (symbIdx + 1)*self.dftSize] \
            = samples[symbIdx*(self.dftSize + self.cpLen) + self.cpLen : (symbIdx + 1)*(self.dftSize + self.cpLen)]

      return tdIqSamples
   
   def fft(self, samples, nSymbols):
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      fdIqSamples = np.ndarray(nSymbols * nDataAndPilotSubcarriers, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         tdSymb = samples[symbIdx*self.dftSize : (symbIdx + 1)*self.dftSize]
         fdSymb = np.fft.fftshift(np.fft.fft(tdSymb))
         fdSymb = np.delete(fdSymb, len(fdSymb) // 2)
         fdIqSamples[symbIdx*nDataAndPilotSubcarriers : (symbIdx + 1)*nDataAndPilotSubcarriers] = fdSymb
         #plt.plot(np.abs(fdSymb))
         #plt.savefig(f"symbs/symb{symbIdx}.png")
         #plt.close()
      return fdIqSamples

   def detachPilots(self, iqSamples, nSymbols):
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      assert len(iqSamples) % nDataAndPilotSubcarriers == 0
      nextPilot = 0
      iqIdx = 0
      pilotIdx = 0

      dataIqSamples = np.zeros(nSymbols * self.nDataSubcarriers, dtype=np.complex64)
      pilotIqSamples = np.zeros(nSymbols * self.nPilotSubcarriers, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         for subcIdx in range(nDataAndPilotSubcarriers):
            if subcIdx == nextPilot:
               pilotIqSamples[pilotIdx] = iqSamples[symbIdx * nDataAndPilotSubcarriers + subcIdx]
               pilotIdx += 1
               nextPilot = (nextPilot + 5) % nDataAndPilotSubcarriers # Every fith is a pilot
            else:
               dataIqSamples[iqIdx] = iqSamples[symbIdx * nDataAndPilotSubcarriers + subcIdx]
               iqIdx += 1
         nextPilot = (nextPilot + 1) % 5 # pilots are shifted by one for each symbol
      return (dataIqSamples, pilotIqSamples)

   def combinePilots(self, pilotIqSamples, nFrames):
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      combinedPilots = np.zeros(len(pilotIqSamples) // 2, dtype=np.complex64)
      for frameIdx in range(nFrames):
         combinedFramePilots = np.zeros(nDataAndPilotSubcarriers, dtype=np.complex64)
         subcIdx = 0
         for symbIdx in range(self.symbolsPerFrame // 2):
            for pilotIdx in range(self.nPilotSubcarriers):
               combinedFramePilots[subcIdx] = \
                  pilotIqSamples[frameIdx*self.symbolsPerFrame + symbIdx*self.nPilotSubcarriers + pilotIdx]
               subcIdx += 5
            subcIdx = (subcIdx + 1) % 5
         for symbIdx in range(self.symbolsPerFrame // 2, self.symbolsPerFrame):
            for pilotIdx in range(self.nPilotSubcarriers):
               combinedFramePilots[subcIdx - self.symbolsPerFrame // 2] = \
                  (combinedFramePilots[subcIdx] + pilotIqSamples[frameIdx*self.symbolsPerFrame + symbIdx*self.nPilotSubcarriers + pilotIdx]) * 0.5
               subcIdx += 5
            subcIdx = (subcIdx + 1) % 5
         combinedPilots[frameIdx*nDataAndPilotSubcarriers : (frameIdx + 1)*nDataAndPilotSubcarriers] = combinedFramePilots[:]
      return combinedPilots

   def demapIqSample(self, iqSample, pilot):
      iqSample = np.sqrt(2) * iqSample / pilot
      minD = np.abs(iqSample - self.modulationMap[0])
      val = 0
      for mapIdx in range(1, len(self.modulationMap)):
         d = np.abs(iqSample - self.modulationMap[mapIdx])
         if d < minD:
            minD = d
            val = mapIdx
      #print(f'{iqSample} {pilot} {val}')
      return val

   def demapBytes(self, dataIqSamples, combinedPilots, nFrames):
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      demapedBytes = np.zeros(len(dataIqSamples), dtype=np.int16)
      dataIdx = 0
      for frameIdx in range(nFrames):
         pilotIdx = 0
         for symboldIdx in range(self.symbolsPerFrame):
            for subcIdx in range(nDataAndPilotSubcarriers):
               if subcIdx == pilotIdx:
                  pilotIdx += 5
               else:
                  demapedSample = self.demapIqSample(dataIqSamples[dataIdx], combinedPilots[frameIdx*nDataAndPilotSubcarriers + subcIdx])
                  demapedBytes[dataIdx] = demapedSample
                  dataIdx += 1
            pilotIdx = (pilotIdx + 1) % 5
      return demapedBytes

   def combineBytes(self, demappedBytes, nFrames):
      iqSamplesPerSample = np.dtype(np.int16).itemsize * 8 // self.bitsPerElement
      print(iqSamplesPerSample)
      nSamples = nFrames * self.symbolsPerFrame * self.nDataSubcarriers // iqSamplesPerSample
      print(nSamples)
      print(len(demappedBytes) // iqSamplesPerSample)
      samples = np.zeros(nSamples, dtype=np.int16)
      for sampleIdx in range(nSamples):
         sample = np.int16(0)
         for elementIdx in range(iqSamplesPerSample):
            element = demappedBytes[sampleIdx*iqSamplesPerSample + elementIdx] << elementIdx * self.bitsPerElement
            sample = sample | element
            #print(demappedBytes[sampleIdx*iqSamplesPerSample + elementIdx])
         samples[sampleIdx] = sample
         #print(f'sample: {sample}')
      return samples




#    def pad(self, samples):
#       suffixLen = (self.audioSamplesPerSymbol - len(samples) % self.audioSamplesPerSymbol) % self.audioSamplesPerSymbol
#       return np.pad(samples, (0, suffixLen))

#    def mapSamplesToQPSK(self, samples):
#       iqSamplesPerSample = np.dtype(np.int16).itemsize * 8 // self.bitsPerElement
#       iqSamples = np.zeros(iqSamplesPerSample * len(samples), dtype=np.complex64)
#       for sampleIdx, sample in enumerate(samples):
#          for elementIdx in range(iqSamplesPerSample):
#             bits = sample & self.modulationBitMask
#             sample = sample >> self.bitsPerElement
#             iqSamples[sampleIdx * iqSamplesPerSample + elementIdx] = self.modulationMap[bits]
#          #print(f'{iqSamples[sampleIdx * iqSamplesPerSample]}{iqSamples[sampleIdx * iqSamplesPerSample + 1]}{iqSamples[sampleIdx * iqSamplesPerSample + 2]}{iqSamples[sampleIdx * iqSamplesPerSample + 3]}{iqSamples[sampleIdx * iqSamplesPerSample + 4]}{iqSamples[sampleIdx * iqSamplesPerSample + 5]}{iqSamples[sampleIdx * iqSamplesPerSample + 6]}{iqSamples[sampleIdx * iqSamplesPerSample + 7]}')

#       return iqSamples
   
#    def mapToSymbIFFTcp(self, iqDataSamples):
#       assert len(iqDataSamples) % self.nDataSubcarriers == 0
#       nSymbols = len(iqDataSamples) // self.nDataSubcarriers
#       nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
#       iqIdx = 0
#       pilotIdx = 0
#       tdIqSamples = np.zeros((self.dftSize + self.cpLen) * nSymbols, dtype=np.complex64)
#       for symbIdx in range(nSymbols):
#          symb = np.zeros(nDataAndPilotSubcarriers, dtype=np.complex64)
#          for subcIdx in range(nDataAndPilotSubcarriers):
#             if subcIdx == pilotIdx:
#                symb[subcIdx] = pilotValue
#                pilotIdx = (pilotIdx + 5) % nDataAndPilotSubcarriers # Every fith is a pilot
#             else:
#                symb[subcIdx] = iqDataSamples[iqIdx] # add NULLLLLLLLLLLLLLLLLLLLLLLLLLL
#                iqIdx += 1
#          pilotIdx = (pilotIdx + 1) % 5 # pilots are shifted by one for each symbol
#          symb = np.insert(symb, len(symb) // 2, nullValue) # insert null subcarrier
#          #symb = np.pad(symb, (20, 20), constant_values=(np.complex64(0, 0), np.complex64(0, 0)))
#          #plt.plot(np.abs(symb))
#          #plt.savefig("symb.png")
#          #plt.close()
#          tdSymb = np.fft.ifft(np.fft.ifftshift(symb))
#          tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : symbIdx * (self.dftSize + self.cpLen) + self.cpLen] = tdSymb[-self.cpLen:]
#          tdIqSamples[symbIdx * (self.dftSize + self.cpLen) + self.cpLen : (symbIdx + 1) * (self.dftSize + self.cpLen)] = tdSymb[:]
#          #plt.plot(np.abs(tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : (symbIdx + 1) * (self.dftSize + self.cpLen)]))
#          #plt.savefig(f"tdSymb/tdSymb{symbIdx}.png")
#          #plt.close()
#       return tdIqSamples

   def receive(self, tdSamples, nFrames):
      nSymbols = nFrames * self.symbolsPerFrame

      print("Demodulating baseband")
      tdIqSamples = self.iqDac(tdSamples)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamples)

      print("Detaching CP")
      tdIqSamples = self.detachCp(tdIqSamples, nSymbols)

      print("Performing FFT")
      fdIqSamples = self.fft(tdIqSamples, nSymbols)

      print("Separating pilots from data")
      dataIqSamples, pilotIqSamples = self.detachPilots(fdIqSamples, nSymbols)

      plotStarmap(dataIqSamples, "dataStarmap.png")
      plotStarmap(pilotIqSamples, "pilotStarmap.png")

      print("Combining pilots")
      combinedPilots = self.combinePilots(pilotIqSamples, nFrames)

      plotStarmap(combinedPilots, "combinedPilotStarmap.png")

      print("Demapping bytes")
      demappedBytes = self.demapBytes(dataIqSamples, combinedPilots, nFrames)

      print("Combining bytes")
      samples = self.combineBytes(demappedBytes, nFrames)


      # print("Adding padding")
      # paddedSamples = self.pad(samples)

      # print("Mapping samples")
      # iqDataSamples = self.mapSamplesToQPSK(paddedSamples)

      # print("Calculalating OFDM IQ samples")
      # tdIqSamples = self.mapToSymbIFFTcp(iqDataSamples)

      # print("Resampling")
      # tdIqSamples = self.resample(tdIqSamples)

      # print("Modulating baseband")
      # tdSamples = self.iqDac(tdIqSamples)

      return samples

