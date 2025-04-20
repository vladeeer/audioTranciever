import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plotSignedSamples(w, h, samples, path, sampleRate = None):
   DPI = 80
   k = int(len(samples) / w / 16)

   def format_db(x, pos=None):
      if pos == 0:
         return ""
      if x == 0:
         return "-inf"
      
      db = 20 * np.log10(abs(x) / float(256 ** 2 / 2))
      return f"{int(db)} dB"
   
   def format_time(x, pos=None):
      duration = len(samples) / sampleRate
      progress = int(x / len(samples) * duration * k)
      mins, secs = divmod(progress, 60)
      out = "%d:%02d" % (mins, secs)
      return out

   plt.figure(1, figsize=(float(w)/DPI, float(h)/DPI), dpi=DPI)
   plt.subplots_adjust(wspace=0, hspace=0)
   axes = plt.subplot(1, 1, 1)
   axes.set_facecolor("k")
   axes.plot(samples[::k], "g")
   axes.set_xlim(0,len(samples) / k)
   axes.yaxis.set_major_formatter(ticker.FuncFormatter(format_db))

   if sampleRate:
      axes.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
   else:
      axes.xaxis.set_major_formatter(ticker.NullFormatter())

   plt.grid(True, axis="y", color="w")
   plt.savefig(path)
   plt.close()

