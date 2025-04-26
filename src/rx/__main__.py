#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..common.wav import Wav
from ..common.utils import plotSignedSamples
from ..rx.receiver import Receiver

#python3 -m src.rx -t samples -m 4 -i tx.wav -o rx.wav -n 2205
def main():
   parser = argparse.ArgumentParser(description="OFDM audio signal receiver.")
   parser.add_argument("-t" ,"--output-type", choices=["samples", "file"], required=True,
                       help="Rx the whole file or just sample bytes of a .wav file")
   parser.add_argument("-i" ,"--input-file", type=str, required=True,
                       help="Path to input .wav file")
   parser.add_argument("-o" ,"--output-file", type=str, required=True,
                       help="Path to output file")
   parser.add_argument("-m" ,"--mode", type=int, choices=range(0, 9), default=8,
                       help="Selects number of subcarriers, modulation type (BPSK or QPSK) and number of pilots")
   parser.add_argument("-n" ,"--num-frames", type=int, required=True,
                       help="Number of frames to listen for")
   
   args = parser.parse_args()
   input_file_path = args.input_file
   output_file_path = args.output_file

   tdSamples = Wav(path = input_file_path).samples

   receiver = Receiver(args.mode)
   tdSamples = receiver.receive(tdSamples, args.num_frames)

   if args.output_type == "samples":
      Wav(tdSamples).write(output_file_path)
   elif args.output_type == "file":
      tdSamples.tofile(output_file_path)

   plotSignedSamples(1000, 300, tdSamples, "rxTdSignal.png", 44100)

if __name__ == "__main__":
   main()
