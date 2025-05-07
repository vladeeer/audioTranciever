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
      nTaps = 1023
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
      step = symbLen // 2
      offset = 0
      maxInd0 = 0
      maxInd1 = 0
      
      while np.abs((maxInd1 - maxInd0) - (self.pilotSymbInd[1] - self.pilotSymbInd[0]) * symbLen) > 2:
         s = tdIqSamples[:frameLen]
         corr0 = np.abs(np.correlate(s, pilotGen0.symbol, mode='valid'))
         corr1 = np.abs(np.correlate(s, pilotGen1.symbol, mode='valid'))
         maxInd0 = np.argmax(corr0)
         maxInd1 = np.argmax(corr1)
         tdIqSamples = tdIqSamples[step:]

         plt.subplot(2, 1, 1)
         plt.plot(corr0)
         plt.subplot(2, 1, 2)
         plt.plot(corr1)
         plt.tight_layout()
         plt.savefig(f"initialSyncCorr/offset_{offset}_samples.png", dpi=300)
         plt.close()
         offset += step
         

         if len(tdIqSamples) < frameLen:
            print('[ERROR] could not sync')

      print(f'Sync detected at sample offset: {offset}')

      initialSampleOffset = (np.argmax(corr0) + symbLen * 11 - self.nSyncBufferSamples) % frameLen 
      tdIqSamples = tdIqSamples[initialSampleOffset:]

      print(f'initialSampleOffset: {initialSampleOffset + offset}')
      #plt.rcParams['agg.path.chunksize'] = 1000
      #plt.figure(figsize=(500, 0.5))
      plt.subplot(2, 1, 1)
      plt.plot(corr0)
      plt.subplot(2, 1, 2)
      plt.plot(corr1)
      plt.tight_layout()
      plt.savefig("Autocor.png", dpi=300)
      plt.close()

      syncedTdIqSamples = np.zeros(len(tdIqSamples), dtype=np.complex64)
      syncedTdIqSamples[:frameLen] = tdIqSamples[:frameLen]
      for frameIdx in range(1, nFrames):
         # First frame is previous!
         s = tdIqSamples[frameLen : frameLen * 2]

         corr0 = np.abs(np.correlate(s, pilotGen0.symbol, mode='valid'))
         corr1 = np.abs(np.correlate(s, pilotGen1.symbol, mode='valid'))
         maxInd0 = np.argmax(corr0)
         maxInd1 = np.argmax(corr1)

         sampleOffset = 0
         infoString = f'frame: {frameIdx}, maxInd0: {maxInd0-self.nSyncBufferSamples}({symbLen*3-1}), maxInd1: {maxInd1-self.nSyncBufferSamples}({symbLen*10-1})'
         if maxInd1 - maxInd0 != (self.pilotSymbInd[1] - self.pilotSymbInd[0]) * symbLen:
            print(f'freerun         | {infoString}')
            syncedTdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen] = tdIqSamples[frameLen : frameLen * 2]
         else:
            if (maxInd0-self.nSyncBufferSamples == symbLen * 3 - 1) and (maxInd1-self.nSyncBufferSamples == symbLen * 10 - 1):
               print(f'no need to sync | {infoString}')
               syncedTdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen] = tdIqSamples[frameLen : frameLen * 2]
            else:
               print(f'resync          | {infoString}')
               sampleOffset = maxInd0 - (symbLen * 3 + self.nSyncBufferSamples - 1)
               syncedTdIqSamples[frameIdx * frameLen : (frameIdx + 1) * frameLen] = tdIqSamples[frameLen + sampleOffset : frameLen * 2 + sampleOffset]

         tdIqSamples = tdIqSamples[frameLen + sampleOffset:]

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
      pilotFrameLen = symbLen * self.nPilotSymbolsPerFrame
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

   def interpolateEstimation(self, phase0, phase1):
      idxs = np.arange(self.nSymbolsPerFrame)
      phase = np.interp(idxs, self.pilotSymbInd, (phase0, phase1))
      phase = np.delete(phase, self.pilotSymbInd)
      return phase

   def estimateChannel(self, pilotFdIqSamples, nFrames):
      symbLen = self.nSubcarriers
      pilotFrameLen = symbLen * self.nPilotSymbolsPerFrame
      dataFrameLen = symbLen * self.nDataSymbolsPerFrame

      pilotGen0 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 0)
      pilotGen1 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 1)

      estimatedFrames = np.zeros(nFrames * dataFrameLen, dtype=np.complex64)
      for frameIdx in range(nFrames):
         estimation0 = pilotFdIqSamples[frameIdx * pilotFrameLen : frameIdx * pilotFrameLen + symbLen]
         estimation1 = pilotFdIqSamples[frameIdx * pilotFrameLen + symbLen : frameIdx * pilotFrameLen + 2 * symbLen]
         estimation0 /= pilotGen0.fdSymb[:]
         estimation1 /= pilotGen1.fdSymb[:]

         frameEstimation = np.zeros((symbLen, self.nDataSymbolsPerFrame), dtype=np.complex64)
         for subcIdx in range(self.nSubcarriers):
            subcEstimation0 = estimation0[subcIdx]
            subcEstimation1 = estimation1[subcIdx]
            subcEstimation = self.interpolateEstimation(subcEstimation0, subcEstimation1)
            frameEstimation[subcIdx][:] = subcEstimation[:]
         
         frameEstimation = np.transpose(frameEstimation).flatten()

         estimatedFrames[frameIdx * dataFrameLen : (frameIdx + 1) * dataFrameLen] = frameEstimation

         if frameIdx % 5 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(np.abs(frameEstimation[symbLen:symbLen*2]))
            plt.xlim(0, len(frameEstimation[symbLen:symbLen*2]))
            plt.title('Amplitude (Magnitude)')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.plot(np.angle(frameEstimation[symbLen:symbLen*2]))
            plt.ylim(-np.pi, np.pi)
            plt.xlim(0, len(frameEstimation[symbLen:symbLen*2]))
            plt.title('Phase (Angle)')
            plt.xlabel('Index')
            plt.ylabel('Phase (radians)')

            plt.tight_layout()
            plt.savefig(f"channelEstimate/Frame_{frameIdx}.png")
            plt.close()
      return estimatedFrames

   def equalizeDemodSample(self, iqSample, channelEstimate):
      epsilon = 0.00001
      iqSample = iqSample / (channelEstimate + iqSample * epsilon)
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
         for symbIdx in range(self.nDataSymbolsPerFrame):
            for subcIdx in range(self.nSubcarriers):
               dataIqSample = dataIqSamples[frameIdx * self.nSubcarriers * self.nDataSymbolsPerFrame + symbIdx * self.nSubcarriers + subcIdx]
               channelEstimate = channelEstimates[frameIdx * self.nSubcarriers * self.nDataSymbolsPerFrame + symbIdx * self.nSubcarriers + subcIdx]
               demapedSample = self.equalizeDemodSample(dataIqSample, channelEstimate)
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

   def deinterleave(self, samples):
      audioSamplesPerFrame = self.audioSamplesPerSymbol * self.nDataSymbolsPerFrame
      assert len(samples) % audioSamplesPerFrame == 0
      nFrames = len(samples) // audioSamplesPerFrame
      bytesPerSample = 2
      bitsPerSample = bytesPerSample * 8

      masksForByte = (1 << np.arange(7, -1, -1, dtype=np.uint8))
      masksForFrame = np.tile(masksForByte, audioSamplesPerFrame * bytesPerSample)
      for frameIdx in range(nFrames):
         frameBits = np.ndarray(self.nDataSymbolsPerFrame * self.audioSamplesPerSymbol * bitsPerSample, dtype=np.uint8)
 
         frameBytes = samples[frameIdx * audioSamplesPerFrame : (frameIdx + 1) * audioSamplesPerFrame].view(np.uint8)
         frameBytes = np.repeat(frameBytes, 8)
         frameBits = (frameBytes & masksForFrame) > 0
         #print(f'{frameBits.shape} {Bits.shape} {horizBytes.shape} {masksForHoriz.shape}')
         frameBits = frameBits.reshape(self.audioSamplesPerSymbol * bitsPerSample, self.nDataSymbolsPerFrame)
         
         frameBytes = np.packbits(frameBits.transpose().flatten())
         samples[frameIdx * audioSamplesPerFrame : (frameIdx + 1) * audioSamplesPerFrame] = frameBytes.view(np.int16)

      return samples

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

      print("Deinterleaving")
      samples = self.deinterleave(samples)

      return samples

