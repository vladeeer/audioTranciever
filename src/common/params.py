import numpy as np

pilotValue = np.complex64(np.sqrt(2), 0)
nullValue = np.complex64(0, 0)

bpskMap = np.array([
     1 + 0j, #0x0
    -1 + 0j, #0x1
], dtype=np.complex64)

qpskMap = np.array([
   -0.7071 + 0.7071j, #0x00
    0.7071 + 0.7071j, #0x01
   -0.7071 - 0.7071j, #0x10
    0.7071 - 0.7071j, #0x11
], dtype=np.complex64)

def getModeParams(self, mode):
    match mode:
        case 0: # BPSK, 4 pilots, 1 audio sample per symbol
            self.modulation = "BPSK"
            self.nSubcarriers = 16
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 1
        case 1: # BPSK, 8 pilots, 2 audio samples per symbol
            self.modulation = "BPSK"
            self.nSubcarriers = 32
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 2
        case 2: # BPSK, 16 pilots, 4 audio samples per symbol
            self.modulation = "BPSK"
            self.nSubcarriers = 64
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 4
        case 3: # BPSK, 32 pilots, 8 audio samples per symbol
            self.modulation = "BPSK"
            self.nSubcarriers = 128
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 8
        case 4: # QPSK, 2 pilots, 1 audio sample per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 8
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 1
        case 5: # QPSK, 4 pilots, 2 audio samples per symbol
            self.modulation = "QPSK"
            self.nDataSubcarriers = 16
            self.nSubcarriers = 1
            self.audioSamplesPerSymbol = 2
        case 6: # QPSK, 8 pilots, 4 audio samples per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 32
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 4
        case 7: # QPSK, 16 pilots, 8 audio samples per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 64
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 8
        case 8: # QPSK, 32 pilots, 16 audio samples per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 128
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 16
        case 9: # QPSK, 32 pilots, 16 audio samples per symbol
            self.modulation = "QPSK"
            self.nSubcarriers = 256
            self.nNullSubcarriers = 1
            self.audioSamplesPerSymbol = 32

    self.dftSize = self.nSubcarriers + self.nNullSubcarriers
    self.nSymbolsPerFrame = self.nDataSymbolsPerFrame + self.nPilotSymmbolsPerFrame

def getModulationParams(self, modulation):
    match modulation:
        case "BPSK":
            self.modulationMap = bpskMap
            self.bitsPerElement = 1
            self.modulationBitMask = 0b0001
        case "QPSK":
            self.modulationMap = qpskMap
            self.bitsPerElement = 2
            self.modulationBitMask = 0b0011

def getConsts(self):
    self.bw = 14700
    self.centerFreq = 9000
    self.sampleRate = 44100 # = 14700 * 3
    self.cpLen = 256 # approx 20 ms
    self.nDataSymbolsPerFrame = 12
    self.nPilotSymmbolsPerFrame = 2
    self.dataSymbInd = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13])
    self.pilotSymbInd = np.array([3, 10])