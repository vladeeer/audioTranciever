#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..common.wav import Wav
from ..common.utils import plotSignedSamples
from ..tx.transmitter import Transmitter

#python3 -m src.tx -t samples -i Hellcat44100.wav -o tx.wav
def main():
   parser = argparse.ArgumentParser(description="OFDM audio signal transmitter.")
   parser.add_argument("-t" ,"--io-type", choices=["samples", "file"], required=True,
                       help="Compare samples or whole files")
   parser.add_argument("-i" ,"--input-file", type=str, required=True,
                       help="Path to file used as input for transmission")
   parser.add_argument("-o" ,"--output-file", type=str, required=True,
                       help="Path to file used as output of transmission")
   parser.add_argument("-m" ,"--mode", type=int, default=8, #choices=range(0, 9),
                       help="Selects number of subcarriers, modulation type (BPSK or QPSK) and number of pilots")
   
   args = parser.parse_args()
   input_file_path = args.input_file
   output_file_path = args.output_file

   if args.io_type == "samples":
      inputSamples = Wav(path = input_file_path).samples
      outputSamples = Wav(path = output_file_path).samples
   elif args.io_type == "file":
      inputSamples = np.fromfile(input_file_path, dtype=np.int16)
      outputSamples = np.fromfile(output_file_path, dtype=np.int16)

   print(f'len(inputSamples): {len(inputSamples)}, len(outputSamples): {len(outputSamples)}')
   if len(inputSamples) <= len(outputSamples):
      transmitter = Transmitter(args.mode)
      inputSamples = transmitter.pad(inputSamples)
   else:
      corr = np.correlate(inputSamples.astype(np.float64), outputSamples.astype(np.float64), mode='valid')
      sampleOffset = np.argmax(corr)
      inputSamples = inputSamples[sampleOffset : sampleOffset + len(outputSamples)]
      print(f'sampleOffset: {sampleOffset}')
      plt.plot(corr)
      plt.savefig("testCorr.png", dpi=300)
      plt.close()
   assert inputSamples.shape == outputSamples.shape

   diff = (inputSamples ^ outputSamples).view(np.uint8)
   diffBits = np.unpackbits(diff)
   nDiffBits = np.sum(diffBits)
   nBits = len(inputSamples) * 16
   ber = nDiffBits / nBits

   print("------------------------------------")
   print(f'BER: {ber}')


   #plotSignedSamples(1000, 300, tdSamples, "txTdSignal.png", 44100)

if __name__ == "__main__":
   main()
