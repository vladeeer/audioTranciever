import numpy as np
import matplotlib.pyplot as plt

from ..common.params import pilotValue, nullValue, getModeParams, getModulationParams, getConsts
from ..common.filter import lpFilter
from ..common.pilotGen import PilotGen
from ..common.utils import *

class Receiver():
   def __init__(self, mode):
      getConsts(self)
      getModeParams(self, mode)
      getModulationParams(self, self.modulation)

   def iqAdc(self, modulatedSamples):
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
      nTaps = 8191
      assert upsampleFactor < downsampleFactor
      assert nTaps <= len(samples)

      upsampled = np.zeros(len(samples) * upsampleFactor, dtype=np.complex64)
      upsampled[::upsampleFactor] = samples
      upsampled = lpFilter(upsampled, 1/downsampleFactor, nTaps)
      
      downsampled = upsampled[::downsampleFactor]
      return downsampled

   def sync(self, tdIqSamples, nFrames):
      pilotGen = PilotGen(self.nDataSubcarriers, self.nPilotSubcarriers, self.nNullSubcarriers, self.cpLen, self.sampleRate)
      symbLen = self.dftSize + self.cpLen
      frameLen = symbLen * self.nSymbolsPerFrame

      s = tdIqSamples[:frameLen]
      corr = np.correlate(np.abs(s), np.abs(pilotGen.bbSymb), mode='valid')
      initialSampleOffset = (np.argmax(corr) + symbLen * 6) % frameLen
      tdIqSamples = tdIqSamples[initialSampleOffset:]
      print(f'initialSampleOffset: {initialSampleOffset}')
      plt.plot(corr)
      plt.savefig("Autocor.png")
      plt.close()

      desync_warning_shown = None
      syncedTdIqSamples = np.zeros(len(tdIqSamples), dtype=np.complex64)
      syncedTdIqSamples[:frameLen] = tdIqSamples[:frameLen]
      for frameIdx in range(1, nFrames):
         s = tdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen]
         corr = np.correlate(np.abs(s), np.abs(pilotGen.bbSymb), mode='valid')
         sampleOffset = np.argmax(corr) - symbLen * 5

         if (desync_warning_shown is None and sampleOffset != 0):
            print(f'[WARNING] Desync by {sampleOffset} at frame {frameIdx}')
            desync_warning_shown = True

         syncedTdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen] = \
            tdIqSamples[frameIdx * frameLen + sampleOffset : (frameIdx + 1) * frameLen + sampleOffset]

      return syncedTdIqSamples

   def detachCp(self, samples, nFrames):
      nSymbols = nFrames * self.nSymbolsPerFrame
      tdIqSamples = np.ndarray(nSymbols * self.dftSize, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         tdIqSamples[symbIdx*self.dftSize : (symbIdx + 1)*self.dftSize] \
            = samples[symbIdx*(self.dftSize + self.cpLen) + self.cpLen : (symbIdx + 1)*(self.dftSize + self.cpLen)]

      return tdIqSamples
   
   def fft(self, samples, nFrames):
      nSymbols = nFrames * self.nSymbolsPerFrame
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

   def detachPilots(self, iqSamples, nFrames):
      nMixedSymbols = nFrames * self.nMixedSymbolsPerFrame
      nSymbols = nFrames * self.nSymbolsPerFrame
      nDataAndPilotSubcarriers = self.nDataSubcarriers + self.nPilotSubcarriers
      assert len(iqSamples) % nDataAndPilotSubcarriers == 0
      nextPilot = 0
      iqIdx = 0
      pilotIdx = 0

      dataIqSamples = np.zeros(nMixedSymbols * self.nDataSubcarriers, dtype=np.complex64)
      pilotIqSamples = np.zeros(nMixedSymbols * self.nPilotSubcarriers, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         if symbIdx % self.nSymbolsPerFrame != 5:
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
         for symbIdx in range(self.nMixedSymbolsPerFrame // 2):
            for pilotIdx in range(self.nPilotSubcarriers):
               combinedFramePilots[subcIdx] = \
                  pilotIqSamples[frameIdx*self.nMixedSymbolsPerFrame + symbIdx*self.nPilotSubcarriers + pilotIdx]
               subcIdx += 5
            subcIdx = (subcIdx + 1) % 5
         for symbIdx in range(self.nMixedSymbolsPerFrame // 2, self.nMixedSymbolsPerFrame):
            for pilotIdx in range(self.nPilotSubcarriers):
               combinedFramePilots[subcIdx - self.nMixedSymbolsPerFrame // 2] = \
                  (combinedFramePilots[subcIdx] + pilotIqSamples[frameIdx*self.nMixedSymbolsPerFrame + symbIdx*self.nPilotSubcarriers + pilotIdx]) * 0.5
               subcIdx += 5
            subcIdx = (subcIdx + 1) % 5
         combinedPilots[frameIdx*nDataAndPilotSubcarriers : (frameIdx + 1)*nDataAndPilotSubcarriers] = combinedFramePilots[:]
      return combinedPilots

   def equalizeSample(self, iqSample, pilot):
      epsilon = 0.000000001
      iqSample = np.sqrt(2) * iqSample / (pilot + np.complex64(epsilon, 0))
      self.iqData.append(iqSample)
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
         for symboldIdx in range(self.nMixedSymbolsPerFrame):
            for subcIdx in range(nDataAndPilotSubcarriers):
               if subcIdx == pilotIdx:
                  pilotIdx += 5
               else:
                  demapedSample = self.equalizeSample(dataIqSamples[dataIdx], combinedPilots[frameIdx*nDataAndPilotSubcarriers + subcIdx])
                  demapedBytes[dataIdx] = demapedSample
                  dataIdx += 1
            pilotIdx = (pilotIdx + 1) % 5
      return demapedBytes

   def combineBytes(self, demappedBytes, nFrames):
      iqSamplesPerSample = np.dtype(np.int16).itemsize * 8 // self.bitsPerElement
      #print(iqSamplesPerSample)
      nSamples = nFrames * self.nMixedSymbolsPerFrame * self.nDataSubcarriers // iqSamplesPerSample
      #print(nSamples)
      #print(len(demappedBytes) // iqSamplesPerSample)
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

   def receive(self, tdSamples, nFrames):
      #noise = 100 * np.random.randn(len(tdSamples))
      #tdSamples = tdSamples + noise
      #delay = 12
      #delayedSamples = np.pad(tdSamples, (delay, 0))
      #tdSamples = np.pad(tdSamples, (0, delay))
      #tdSamples = tdSamples + 0.1 * delayedSamples
      tdSamples = np.pad(tdSamples, (16, 0))
      print("------------------------------------")

      print("Demodulating baseband")
      tdIqSamples = self.iqAdc(tdSamples)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamples)

      #tdIqSamples = tdIqSamples[(self.dftSize + self.cpLen)*11 - 10:]

      print("Sync Frames")
      tdIqSamples = self.sync(tdIqSamples, nFrames)

      print("Detaching CP")
      tdIqSamples = self.detachCp(tdIqSamples, nFrames)

      print("Performing FFT")
      fdIqSamples = self.fft(tdIqSamples, nFrames)

      print("Separating pilots from data")
      dataIqSamples, pilotIqSamples = self.detachPilots(fdIqSamples, nFrames)

      plotStarmap(dataIqSamples, "dataStarmap.png")
      plotStarmap(pilotIqSamples, "pilotStarmap.png")

      print("Combining pilots")
      combinedPilots = self.combinePilots(pilotIqSamples, nFrames)

      plotStarmap(combinedPilots, "combinedPilotStarmap.png")

      self.iqData = []
      print("Demapping bytes")
      demappedBytes = self.demapBytes(dataIqSamples, combinedPilots, nFrames)
      plotStarmap(self.iqData, "pilotedDataStarmap.png")

      print("Combining bytes")
      samples = self.combineBytes(demappedBytes, nFrames)

      return samples

