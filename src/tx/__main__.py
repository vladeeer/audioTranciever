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
   parser.add_argument("-t" ,"--input-type", choices=["samples", "file"], required=True,
                       help="Tx the whole file or just sample bytes of a .wav file")
   parser.add_argument("-i" ,"--input-file", type=str, required=True,
                       help="Path to input file")
   parser.add_argument("-o" ,"--output-file", type=str, required=True,
                       help="Path to output .wav file")
   parser.add_argument("-m" ,"--mode", type=int, choices=range(0, 9), default=8,
                       help="Selects number of subcarriers, modulation type (BPSK or QPSK) and number of pilots")
   
   args = parser.parse_args()
   input_file_path = args.input_file
   output_file_path = args.output_file

   if args.input_type == "samples":
      tdSamples = Wav(path = input_file_path).samples
   elif args.input_type == "file":
      tdSamples = np.fromfile(input_file_path, dtype=np.int16)

   transmitter = Transmitter(args.mode)
   tdSamples = transmitter.transmit(tdSamples)

   Wav(tdSamples).write(output_file_path)

   plotSignedSamples(1000, 300, tdSamples, "txTdSignal.png", 44100)

if __name__ == "__main__":
   main()
