import numpy as np
import matplotlib.pyplot as plt

from ..common.params import nullValue, getModeParams, getModulationParams, getConsts
from ..common.filter import lpFilter
from ..common.pilotGen import PilotGen
from ..common.coder import Coder

class Transmitter():
   def __init__(self, mode):
      getConsts(self)
      getModeParams(self, mode)
      getModulationParams(self, self.modulation)

   def pad(self, samples):
      samplesPerFrame = self.audioSamplesPerSymbol * self.nDataSymbolsPerFrame
      suffixLen = (samplesPerFrame - (len(samples) % samplesPerFrame)) % samplesPerFrame
      return np.pad(samples, (0, suffixLen))

   def encode(self, samples):
      if self.codeRateInverse > 1:
         coder = Coder(self.codeRateInverse)
         return coder.encode(samples)
      else:
         return samples

   def interleave(self, samples):
      audioSamplesPerFrame = self.audioSamplesPerSymbol * self.nDataSymbolsPerFrame
      assert len(samples) % audioSamplesPerFrame == 0
      nFrames = len(samples) // audioSamplesPerFrame
      bytesPerSample = 2
      bitsPerSample = bytesPerSample * 8

      masksForByte = (1 << np.arange(7, -1, -1, dtype=np.uint8))
      masksForSymbol = np.tile(masksForByte, self.audioSamplesPerSymbol * bytesPerSample)
      for frameIdx in range(nFrames):
         frameBits = np.ndarray((self.nDataSymbolsPerFrame, self.audioSamplesPerSymbol * bitsPerSample), dtype=np.uint8)
         for symbIdx in range(self.nDataSymbolsPerFrame):
            symbolBytes = samples[frameIdx * audioSamplesPerFrame + symbIdx * self.audioSamplesPerSymbol : \
                                  frameIdx * audioSamplesPerFrame + (symbIdx + 1) * self.audioSamplesPerSymbol].view(np.uint8)
            symbolBytes = np.repeat(symbolBytes, 8)
            symbolBits = (symbolBytes & masksForSymbol) > 0
            frameBits[symbIdx][:] = symbolBits[:]

         frameBytes = np.packbits(frameBits.transpose().flatten())
         samples[frameIdx * audioSamplesPerFrame : (frameIdx + 1) * audioSamplesPerFrame] = frameBytes.view(np.int16)

      return samples

   def scramble(self, samples):
      assert len(samples) % self.audioSamplesPerSymbol == 0
      nSymbols = len(samples) // self.audioSamplesPerSymbol

      n = np.arange(self.audioSamplesPerSymbol)
      seq = (534534512311 * n + 1984).astype(np.uint16)
      for symbIdx in range(nSymbols):
         symbolSamples = samples[self.audioSamplesPerSymbol * symbIdx : self.audioSamplesPerSymbol * (symbIdx + 1)].view(np.uint16)
         symbolSamples = (symbolSamples << 5) | (symbolSamples >> (16 - 5)) # Apply cyclic shift before and after xor
         symbolSamples = symbolSamples ^ seq
         symbolSamples = (symbolSamples >> 3) | (symbolSamples << (16 - 3))
         samples[self.audioSamplesPerSymbol * symbIdx : self.audioSamplesPerSymbol * (symbIdx + 1)] = symbolSamples

      return samples

   def mapSamplesToQPSK(self, samples):
      iqSamplesPerSample = 16 // self.bitsPerElement
      iqSamples = np.zeros(iqSamplesPerSample * len(samples), dtype=np.complex64)
      for sampleIdx, sample in enumerate(samples):
         for elementIdx in range(iqSamplesPerSample):
            bits = sample & self.modulationBitMask
            sample = sample >> self.bitsPerElement
            iqSamples[sampleIdx * iqSamplesPerSample + elementIdx] = self.modulationMap[bits]
         #print(f'{iqSamples[sampleIdx * iqSamplesPerSample]}{iqSamples[sampleIdx * iqSamplesPerSample + 1]}{iqSamples[sampleIdx * iqSamplesPerSample + 2]}{iqSamples[sampleIdx * iqSamplesPerSample + 3]}{iqSamples[sampleIdx * iqSamplesPerSample + 4]}{iqSamples[sampleIdx * iqSamplesPerSample + 5]}{iqSamples[sampleIdx * iqSamplesPerSample + 6]}{iqSamples[sampleIdx * iqSamplesPerSample + 7]}')

      return iqSamples
   
   def mapToSymbIFFTcp(self, iqDataSamples):
      assert len(iqDataSamples) % self.nSubcarriers == 0
      nDataSymbols = len(iqDataSamples) // self.nSubcarriers
      symbLen = self.dftSize + self.cpLen

      tdIqSamples = np.zeros((self.dftSize + self.cpLen) * nDataSymbols, dtype=np.complex64)
      iqIdx = 0
      for symbIdx in range(nDataSymbols):
         symb = np.zeros(self.nSubcarriers, dtype=np.complex64)
         for subcIdx in range(self.nSubcarriers):
            symb[subcIdx] = iqDataSamples[iqIdx]
            iqIdx += 1
         symb = np.insert(symb, len(symb) // 2, nullValue) # insert null subcarrier
         #symb = np.pad(symb, (20, 20), constant_values=(np.complex64(0, 0), np.complex64(0, 0)))
         #plt.plot(np.abs(symb))
         #plt.savefig("symb.png")
         #plt.close()
         tdSymb = np.fft.ifft(np.fft.ifftshift(symb))
         tdIqSamples[symbIdx * symbLen : symbIdx * symbLen + self.cpLen] = tdSymb[-self.cpLen:]
         tdIqSamples[symbIdx * symbLen + self.cpLen : (symbIdx + 1) * symbLen] = tdSymb[:]
         #plt.plot(np.abs(tdIqSamples[symbIdx * (self.dftSize + self.cpLen) : (symbIdx + 1) * (self.dftSize + self.cpLen)]))
         #plt.savefig(f"tdSymb/tdSymb{symbIdx}.png")
         #plt.close()

      return (tdIqSamples, nDataSymbols)

   def addPilotSymb(self, tdIqSamples, nFrames):
      symbLen = self.dftSize + self.cpLen
      frameLen = symbLen * self.nSymbolsPerFrame
      pilotlessFrameLen = symbLen * self.nDataSymbolsPerFrame

      pilotGen0 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 0)
      pilotGen1 = PilotGen(self.nSubcarriers, self.nNullSubcarriers, self.cpLen, 1)

      tdIqSamplesWithPilotSymb = np.zeros(len(tdIqSamples) + nFrames * symbLen * self.nPilotSymbolsPerFrame, dtype=np.complex64)

      for frameIdx in range(nFrames):
         dataSymbIdx = 0
         for symbIdx in range(self.nSymbolsPerFrame):
            if symbIdx == self.pilotSymbInd[0]:
               tdIqSamplesWithPilotSymb[frameIdx * frameLen + symbIdx * symbLen : frameIdx * frameLen + (symbIdx + 1) * symbLen] = \
                  pilotGen0.symbol[:]
            elif symbIdx == self.pilotSymbInd[1]:
               tdIqSamplesWithPilotSymb[frameIdx * frameLen + symbIdx * symbLen : frameIdx * frameLen + (symbIdx + 1) * symbLen] = \
                  pilotGen1.symbol[:]
            elif symbIdx in self.dataSymbInd:
               tdIqSamplesWithPilotSymb[frameIdx * frameLen + symbIdx * symbLen : frameIdx * frameLen + (symbIdx + 1) * symbLen] = \
                  tdIqSamples[frameIdx * pilotlessFrameLen + dataSymbIdx * symbLen : frameIdx * pilotlessFrameLen + (dataSymbIdx + 1) * symbLen]
               dataSymbIdx += 1
         
      return tdIqSamplesWithPilotSymb

   def resample(self, samples):
      upsampleFactor = self.resamplingFactor
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

      print("Coding")
      codedSamples = self.encode(paddedSamples)

      print("Interleaving")
      interleavedSamples = self.interleave(codedSamples)

      print("Scrambling")
      scrambledSamples = self.scramble(interleavedSamples)

      print("Mapping samples")
      iqDataSamples = self.mapSamplesToQPSK(scrambledSamples)

      print("Calculalating OFDM IQ samples")
      tdIqSamples, nMixedSymbols = self.mapToSymbIFFTcp(iqDataSamples)
      nFrames = nMixedSymbols // self.nDataSymbolsPerFrame

      print("Adding pilot symbols")
      tdIqSamplesWithPilotSymb = self.addPilotSymb(tdIqSamples, nFrames)

      print("Resampling")
      tdIqSamples = self.resample(tdIqSamplesWithPilotSymb)

      print("Modulating baseband")
      tdSamples = self.iqDac(tdIqSamples)

      print(f"Transmitted {nFrames} frames")

      return tdSamples

