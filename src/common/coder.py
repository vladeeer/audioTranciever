#!/usr/bin/env python3

import numpy as np

genPolynomialsList = [[0b1101, 0b1111],                  \
                      [0b1011, 0b1101, 0b1111],          \
                      [0b1011, 0b1101, 0b1101, 0b1111]]

def maskBits(n):
   s = np.uint8(1)
   for i in range(n-1):
      s = (s << 1) + 1
   return s

class Coder:
   def __init__(self, codeRateInverse):
      self.nGenerators = codeRateInverse
      self.genPolynomials = genPolynomialsList[codeRateInverse - 2]
      self.constraintLength = 4

   def encode(self, samples):
      bits = np.unpackbits(samples.view(np.uint8))
      
      mask = maskBits(self.constraintLength)
      state = 0
      encodedBits = np.empty(self.nGenerators * len(bits), dtype=np.uint8)
      for i in range(len(bits)):
         state = ((state << 1) | bits[i]) & mask
         for gIdx in range(self.nGenerators):
            encodedBits[self.nGenerators * i + gIdx] = bin(state & self.genPolynomials[gIdx]).count('1') % 2

      encodedBytes = np.packbits(encodedBits)
      return encodedBytes.view(np.int16)

   def decode(self, samples):
      receivedBits = np.unpackbits(samples.view(np.uint8))

      windowLen = 256
      nStates = 2 ** (self.constraintLength - 1)
      nBits = len(receivedBits) // self.nGenerators

      pathMetrics = np.full(nStates, np.iinfo(np.uint32).max, dtype=np.uint32)
      pathMetrics[0] = 0.0

      prevStates = np.zeros((nBits, nStates), dtype=np.int16)
      prevInputs = np.zeros((nBits, nStates), dtype=np.uint8)

      nextState = np.zeros((nStates, 2), dtype=np.uint8)
      outputBits = np.zeros((nStates, 2, self.nGenerators), dtype=np.uint8)
      mask = maskBits(self.constraintLength)
      for state in range(nStates):
         for bit in [0, 1]:
               s = ((state << 1) | bit) & mask
               sNext = s & (nStates - 1)
               nextState[state, bit] = sNext
               for gIdx in range(self.nGenerators):
                  outputBits[state, bit, gIdx] = bin(s & self.genPolynomials[gIdx]).count('1') % 2

      decodedBits = np.zeros(nBits, dtype=np.uint8)

      for t in range(nBits):
         r = receivedBits[self.nGenerators * t : self.nGenerators * (t + 1)]
         rTiled = np.tile(r, (nStates, 2, 1))
         
         expected = outputBits
         metrics = np.sum((rTiled != expected).astype(np.uint8), axis=2)

         newPathMetrics = np.full(nStates, np.iinfo(np.uint32).max, dtype=np.uint32)
         for state in range(nStates):
               for bit in [0, 1]:
                  totalMetrics = pathMetrics[state] + metrics[state, bit]
                  sNext = nextState[state, bit]
                  if totalMetrics < newPathMetrics[sNext]:
                     newPathMetrics[sNext] = totalMetrics
                     prevStates[t, sNext] = state
                     prevInputs[t, sNext] = bit

         pathMetrics = newPathMetrics

         if (t + 1) % windowLen == 0 or t == nBits - 1:
            bestState = int(np.argmin(pathMetrics))
            for invT in reversed(range(t - (min(windowLen, t + 1)) + 1, t + 1)):
                  decodedBits[invT] = prevInputs[invT, bestState]
                  bestState = int(prevStates[invT, bestState])

      decoded_bytes = np.packbits(decodedBits)
      return decoded_bytes.view(np.int16)

def add_noise(samples, p):
   bits = np.unpackbits(samples.view(np.uint8))

   noise = np.random.rand(len(bits)) < p
   bits = np.bitwise_xor(bits, noise.astype(np.uint8))

   noisyBytes = np.packbits(bits)
   return noisyBytes.view(np.int16)

def main():
   np.random.seed(42)
   num_samples = 88200
   original_samples = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16)

   coder = Coder(codeRateInverse=2)
   encoded = coder.encode(original_samples)
   noisy = add_noise(encoded, p=0.05)
   recovered_samples = coder.decode(noisy)

   original_bits = np.unpackbits(original_samples.view(np.uint8))
   recovered_bits = np.unpackbits(recovered_samples.view(np.uint8))

   total_bits = len(original_bits)
   bit_errors = np.count_nonzero(original_bits != recovered_bits)
   ber = bit_errors / total_bits

   print(f"Total bits transmitted: {total_bits}")
   print(f"Bit errors: {bit_errors}")
   print(f"BER: {ber:.6f}")

   matches = np.sum(recovered_samples == original_samples)
   print(f"Sample-wise matches: {matches} out of {num_samples}")
   print(f"Sample-wise accuracy: {matches/num_samples*100:.2f}%")

if __name__ == "__main__":
   main()