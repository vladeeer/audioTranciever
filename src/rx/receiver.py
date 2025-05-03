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
      pilotGen = PilotGen(self.nDataSubcarriers, self.nPilotSubcarriers, self.nNullSubcarriers, self.cpLen)
      symbLen = self.dftSize + self.cpLen
      frameLen = symbLen * self.nSymbolsPerFrame

      s = tdIqSamples[:frameLen]
      corr = np.abs(np.correlate(s, pilotGen.symbol, mode='valid'))
      initialSampleOffset = (np.argmax(corr) + symbLen * 6) % frameLen
      tdIqSamples = tdIqSamples[initialSampleOffset:]

      lag1 = np.argmax(corr)
      corr[lag1] = 0
      lag2 = np.argmax(corr)
      corr[lag2] = 0
      lag3 = np.argmax(corr)

      print(f'lag1: {lag1}, lag2: {lag2}, lag3: {lag3}')

      print(f'initialSampleOffset: {initialSampleOffset}')
      #plt.rcParams['agg.path.chunksize'] = 1000
      #plt.figure(figsize=(500, 0.5))
      plt.plot(corr)
      plt.savefig("Autocor.png", dpi=300)
      plt.close()

      desync_warning_shown = None
      syncedTdIqSamples = np.zeros(len(tdIqSamples), dtype=np.complex64)
      syncedTdIqSamples[:frameLen] = tdIqSamples[:frameLen]
      for frameIdx in range(1, nFrames):
         s = tdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen]
         corr = np.correlate(np.abs(s), np.abs(pilotGen.symbol), mode='valid')
         sampleOffset = np.argmax(corr) - symbLen * 5

         if (desync_warning_shown is None and sampleOffset != 0):
            print(f'[WARNING] Desync by {sampleOffset} at frame {frameIdx}')
            desync_warning_shown = None

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
      fdIqSamples = np.ndarray(nSymbols * self.nSubcarriers, dtype=np.complex64)
      for symbIdx in range(nSymbols):
         tdSymb = samples[symbIdx*self.dftSize : (symbIdx + 1)*self.dftSize]
         fdSymb = np.fft.fftshift(np.fft.fft(tdSymb))
         fdSymb = np.delete(fdSymb, len(fdSymb) // 2)
         fdIqSamples[symbIdx * self.nSubcarriers : (symbIdx + 1) * self.nSubcarriers] = fdSymb
         #plt.plot(np.abs(fdSymb))
         #plt.savefig(f"symbs/symb{symbIdx}.png")
         #plt.close()
      return fdIqSamples

   def detachPilots(self, iqSamples, nFrames):
      assert len(iqSamples) % self.nSubcarriers == 0
      symbLen = self.nSubcarriers
      pilotFrameLen = symbLen * self.nPilotSymmbolsPerFrame
      dataFrameLen = symbLen * self.nDataSymbolsPerFrame
      frameLen = symbLen * self.nSymbolsPerFrame

      pilotFdIqSamples = np.zeros(nFrames * pilotFrameLen, dtype=np.complex64)
      dataFdIqSamples = np.zeros(nFrames * dataFrameLen, dtype=np.complex64)

      pilotSymbIdx = 0
      dataSymbIdx = 0
      for frameIdx in range(nFrames):
         for symbIdx in range(self.nSymbolsPerFrame):
            if symbIdx in self.pilotSymbInd:
               pilotFdIqSamples[pilotSymbIdx * symbLen : (pilotSymbIdx + 1) * symbLen] = \
                  iqSamples[frameIdx * frameLen + symbIdx * symbLen : frameIdx * frameLen + (symbIdx + 1) * symbLen]
               pilotSymbIdx += 1
            elif symbIdx in self.dataSymbInd:
               dataFdIqSamples[dataSymbIdx * symbLen : (dataSymbIdx + 1) * symbLen] = \
                  iqSamples[frameIdx * frameLen + symbIdx * symbLen : frameIdx * frameLen + (symbIdx + 1) * symbLen]
               dataSymbIdx += 1

      return (dataFdIqSamples, pilotFdIqSamples)

   def estimateChannel(self, pilotFdIqSamples, nFrames):
      symbLen = self.nSubcarriers
      pilotFrameLen = symbLen * self.nPilotSymmbolsPerFrame

      pilotGen = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen)

      combinedPilots = np.zeros(nFrames * symbLen, dtype=np.complex64)
      for frameIdx in range(nFrames):
         estimation = pilotFdIqSamples[frameIdx * pilotFrameLen : frameIdx * pilotFrameLen + symbLen]
         estimation /= pilotGen.fdSymb[:]
         combinedPilots[frameIdx * self.nSubcarriers : (frameIdx + 1) * self.nSubcarriers] = estimation

         plt.figure(figsize=(12, 6))
         plt.subplot(2, 1, 1)
         plt.plot(np.abs(estimation))
         plt.xlim(0, len(estimation))
         plt.title('Amplitude (Magnitude)')
         plt.xlabel('Index')
         plt.ylabel('Amplitude')

         plt.subplot(2, 1, 2)
         plt.plot(np.angle(estimation))
         plt.ylim(-np.pi, np.pi)
         plt.xlim(0, len(estimation))
         plt.title('Phase (Angle)')
         plt.xlabel('Index')
         plt.ylabel('Phase (radians)')

         plt.tight_layout()
         plt.savefig(f"channelEstimate/Frame_{frameIdx}.png")
         plt.close()
      return combinedPilots

   def equalizeSample(self, iqSample, channelEstimate):
      epsilon = 0.000000001
      iqSample = np.sqrt(2) * iqSample / (channelEstimate + np.complex64(epsilon, 0))
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

   def demapBytes(self, dataIqSamples, channelEstimates, nFrames):
      demapedBytes = np.zeros(len(dataIqSamples), dtype=np.int16)
      dataIdx = 0
      for frameIdx in range(nFrames):
         for symboldIdx in range(self.nDataSymbolsPerFrame):
            for subcIdx in range(self.nSubcarriers):
               demapedSample = self.equalizeSample(dataIqSamples[dataIdx], channelEstimates[frameIdx * self.nSubcarriers + subcIdx])
               demapedBytes[dataIdx] = demapedSample
               dataIdx += 1
      return demapedBytes

   def combineBytes(self, demappedBytes, nFrames):
      iqSamplesPerSample = 16 // self.bitsPerElement
      #print(iqSamplesPerSample)
      nSamples = nFrames * self.nDataSymbolsPerFrame * self.nSubcarriers // iqSamplesPerSample
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

   def descramble(self, samples):
      seq = 0xB182
      samples = samples.view(np.uint16)
      samples = (samples << 3) | (samples >> (16 - 3)) # Apply cyclic shift before and after xor
      samples = samples ^ seq
      samples = (samples >> 5) | (samples << (16 - 5))
      return samples

   def receive(self, tdSamples, nFrames):
      print("------------------------------------")
      sinr_dB = 20
      signalPower = np.mean((tdSamples.astype(np.int32))**2)
      noisePower = signalPower / (10 ** (sinr_dB / 10))
      print(f'SINR: {sinr_dB} dB, signalPower: {signalPower}, noisePower: {noisePower}')
      noise = np.sqrt(noisePower) * np.random.randn(len(tdSamples))
      #tdSamples = tdSamples + noise
      #delay = 12
      #delayedSamples = np.pad(tdSamples, (delay, 0))
      #tdSamples = np.pad(tdSamples, (0, delay))
      #tdSamples = tdSamples + 0.1 * delayedSamples
      #tdSamples = np.pad(tdSamples, (61, 0))
      print("------------------------------------")

      print("Demodulating baseband")
      tdIqSamples = self.iqAdc(tdSamples)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamples)

      #tdIqSamples = tdIqSamples[(self.dftSize + self.cpLen)*11 - 10:]

      print("Sync Frames")
      #tdIqSamples = self.sync(tdIqSamples, nFrames)

      print("Detaching CP")
      tdIqSamples = self.detachCp(tdIqSamples, nFrames)

      print("Performing FFT")
      fdIqSamples = self.fft(tdIqSamples, nFrames)

      print("Separating pilots from data")
      dataFdIqSamples, pilotFdIqSamples = self.detachPilots(fdIqSamples, nFrames)

      plotStarmap(dataFdIqSamples, "dataStarmap.png")
      plotStarmap(pilotFdIqSamples, "pilotStarmap.png")

      print("Estimating channel response")
      channelEstimates = self.estimateChannel(pilotFdIqSamples, nFrames)

      plotStarmap(channelEstimates, "combinedPilotStarmap.png")

      self.iqData = []
      print("Demapping bytes")
      demappedBytes = self.demapBytes(dataFdIqSamples, channelEstimates, nFrames)
      plotStarmap(self.iqData, "pilotedDataStarmap.png")

      print("Combining bytes")
      samples = self.combineBytes(demappedBytes, nFrames)

      print("Descrambling")
      samples = self.descramble(samples)

      return samples

