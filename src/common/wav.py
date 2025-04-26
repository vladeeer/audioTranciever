import wave
import numpy as np

class Wav:
   def __init__(self, samples = None, path = None):
      if samples is not None:
         self.nChannels = 1
         self.sampleSize = 2
         self.framerate = 44100
         self.nFrames = len(samples)
         self.samples = samples

      elif path:
         self.read(path)

      else:
         self.nChannels = None
         self.sampleSize = None
         self.framerate = None
         self.nFrames = None
         self.samples = None

   def read(self, path):
      with wave.open(path, "rb") as file:
         # Read header data
         self.nChannels = file.getnchannels()
         self.sampleSize = file.getsampwidth()
         self.framerate = file.getframerate()
         self.nFrames = file.getnframes()

         print("------------------------------------")
         print(f"Read {path}")
         print(f"Channels: {self.nChannels}")
         print(f"Sample size: {self.sampleSize} bytes")
         print(f"Frame rate: {self.framerate} Hz")
         print(f"Frames: {self.nFrames}")

         if self.sampleSize != 2:
            raise ValueError(f"Unsupported sample size: {self.sampleSize}. Only 16 bit samples are supported.")

         frames = file.readframes(self.nFrames)
         self.samples = np.frombuffer(frames, dtype=np.int16)

         # Use only first channel if 2 are present
         if self.nChannels == 2:
            self.samples = self.samples.reshape(-1, 2)[:, 0]

   def write(self, path):
      with wave.open(path, "wb") as file:
         print("------------------------------------")
         print(f"Write {path}")
         print(f"Channels: {self.nChannels}")
         print(f"Sample size: {self.sampleSize} bytes")
         print(f"Frame rate: {self.framerate} Hz")
         print(f"Frames: {self.nFrames}")

         file.setnchannels(1)
         file.setsampwidth(self.sampleSize)
         file.setframerate(self.framerate)

         file.writeframes(self.samples.tobytes())