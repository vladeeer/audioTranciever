import numpy as np
import matplotlib.pyplot as plt

from ..common.params import pilotValue, nullValue, getModeParams, getModulationParams, getConsts
from ..common.filter import lpFilter, hpFilter
from ..common.pilotGen import PilotGen
from ..common.utils import *

class Receiver():
   def __init__(self, mode):
      getConsts(self)
      getModeParams(self, mode)
      getModulationParams(self, self.modulation)

   def filterHp(self, samples):
      nTaps = 32767
      normFreq = (self.centerFreq - self.bw / 2) / (self.sampleRate * 0.5)
      samples = hpFilter(samples, normFreq, nTaps)
      return samples

   def iqAdc(self, modulatedSamples):
      modulatedSamples = modulatedSamples.astype(np.complex64)

      timeVec = np.arange(len(modulatedSamples)) / self.sampleRate
      exponent = np.exp(-1j * 2 * np.pi * self.centerFreq * timeVec)
      demodulatedSamples = modulatedSamples * exponent

      #print(np.average(np.abs(modulatedSamples)))
      #return np.real(demodulatedSamples).astype(np.int16)
      return demodulatedSamples

   def resample(self, samples):
      upsampleFactor = 1
      downsampleFactor = self.resamplingFactor
      nTaps = 8191
      assert upsampleFactor < downsampleFactor
      assert nTaps <= len(samples)

      upsampled = np.zeros(len(samples) * upsampleFactor, dtype=np.complex64)
      upsampled[::upsampleFactor] = samples
      upsampled = lpFilter(upsampled, 1/downsampleFactor, nTaps)
      
      downsampled = upsampled[::downsampleFactor]
      return downsampled

   def sync(self, tdIqSamples, nFrames):
      pilotGen0 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 0)
      pilotGen1 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 1)
      symbLen = self.dftSize + self.cpLen
      frameLen = symbLen * self.nSymbolsPerFrame
      offset = 0
      maxInd0 = 0
      maxInd1 = 0
      while maxInd1 - maxInd0 != (self.pilotSymbInd[1] - self.pilotSymbInd[0]) * symbLen:
         s = tdIqSamples[:frameLen]
         corr0 = np.abs(np.correlate(s, pilotGen0.symbol, mode='valid'))
         corr1 = np.abs(np.correlate(s, pilotGen1.symbol, mode='valid'))
         maxInd0 = np.argmax(corr0)
         maxInd1 = np.argmax(corr1)
         tdIqSamples = tdIqSamples[symbLen:]

         if offset % 1 == 0:
            plt.subplot(2, 1, 1)
            plt.plot(corr0)
            plt.subplot(2, 1, 2)
            plt.plot(corr1)
            plt.tight_layout()
            plt.savefig(f"initialSyncCorr/offset_{offset}_symbols.png", dpi=300)
            plt.close()
         offset += 1

         if len(tdIqSamples) < frameLen:
            print('[ERROR] could not sync')

      initialSampleOffset = (np.argmax(corr0) + symbLen * 11 - 2) % frameLen 
      tdIqSamples = tdIqSamples[initialSampleOffset:]

      lag1 = np.argmax(corr0)
      corr0[lag1] = 0
      lag2 = np.argmax(corr0)
      corr0[lag2] = 0
      lag3 = np.argmax(corr0)

      print(f'lag1: {lag1}, lag2: {lag2}, lag3: {lag3}')

      print(f'initialSampleOffset: {initialSampleOffset}')
      #plt.rcParams['agg.path.chunksize'] = 1000
      #plt.figure(figsize=(500, 0.5))
      plt.subplot(2, 1, 1)
      plt.plot(corr0)
      plt.subplot(2, 1, 2)
      plt.plot(corr1)
      plt.tight_layout()
      plt.savefig("Autocor.png", dpi=300)
      plt.close()

      desync_warning_shown = None
      syncedTdIqSamples = np.zeros(len(tdIqSamples), dtype=np.complex64)
      syncedTdIqSamples[:frameLen] = tdIqSamples[:frameLen]
      for frameIdx in range(1, nFrames):
         s = tdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen]
         corr0 = np.correlate(np.abs(s), np.abs(pilotGen0.symbol), mode='valid')
         sampleOffset = 0  #np.argmax(corr0) - symbLen * 5

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

      pilotGen0 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 0)
      pilotGen1 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 1)

      combinedPilots = np.zeros(nFrames * symbLen, dtype=np.complex64)
      for frameIdx in range(nFrames):
         estimation0 = pilotFdIqSamples[frameIdx * pilotFrameLen : frameIdx * pilotFrameLen + symbLen]
         estimation0 /= pilotGen0.fdSymb[:]
         phase0 = np.angle(estimation0)
         amp0 = np.abs(estimation0)
         estimation1 = pilotFdIqSamples[frameIdx * pilotFrameLen + symbLen : frameIdx * pilotFrameLen + 2 * symbLen]
         estimation1 /= pilotGen1.fdSymb[:]
         phase1 = np.angle(estimation1)
         amp1 = np.abs(estimation1)

         amp = 0.5 * (amp0 + amp1)
         phase = 0.5 * (phase0 + phase1)
         estimation = estimation0 #np.array(amp * (np.cos(phase) + 1j * np.sin(phase)), dtype = np.complex64)

         combinedPilots[frameIdx * self.nSubcarriers : (frameIdx + 1) * self.nSubcarriers] = estimation

         if frameIdx < 10:
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
      iqSample = iqSample / (channelEstimate + np.complex64(epsilon, 0))
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
      assert len(samples) % self.audioSamplesPerSymbol == 0
      nSymbols = len(samples) // self.audioSamplesPerSymbol

      n = np.arange(self.audioSamplesPerSymbol)
      seq = (534534512311 * n + 1984).astype(np.uint16)
      for symbIdx in range(nSymbols):
         symbolSamples = samples[self.audioSamplesPerSymbol * symbIdx : self.audioSamplesPerSymbol * (symbIdx + 1)].view(np.uint16)
         symbolSamples = (symbolSamples << 3) | (symbolSamples >> (16 - 3)) # Apply cyclic shift before and after xor
         symbolSamples = symbolSamples ^ seq
         symbolSamples = (symbolSamples >> 5) | (symbolSamples << (16 - 5))

         samples[self.audioSamplesPerSymbol * symbIdx : self.audioSamplesPerSymbol * (symbIdx + 1)] = symbolSamples
      return samples.view(np.int16)

   def receive(self, tdSamples, nFrames):
      print("------------------------------------")
      # sinr_dB = 12
      # signalPower = np.mean((tdSamples.astype(np.int32))**2)
      # noisePower = signalPower / (10 ** (sinr_dB / 10))
      # print(f'SINR: {sinr_dB} dB, signalPower: {signalPower}, noisePower: {noisePower}')
      # noise = np.sqrt(noisePower) * np.random.randn(len(tdSamples))
      # tdSamples = tdSamples + noise
      # delay = 100
      # delayedSamples = np.pad(tdSamples, (delay, 0))
      # tdSamples = np.pad(tdSamples, (0, delay))
      # tdSamples = tdSamples + 0.7 * delayedSamples
      # tdSamples = np.pad(tdSamples, (62, 0))
      # print("------------------------------------")

      print("Filtering")
      #tdSamples = self.filterHp(tdSamples)

      print("Demodulating to baseband")
      tdIqSamples = self.iqAdc(tdSamples)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamples)

      print("Sync Frames")
      tdIqSamples = self.sync(tdIqSamples, nFrames)

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

