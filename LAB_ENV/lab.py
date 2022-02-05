# -*- coding: utf-8 -*-
"""
Inicialização para o trabalho de laboratório de Sinais e Sistemas

Created on Oct 2021

@author: pmqa
"""

def timevar(duration):
	"""Produce a time variable in [-duration/2,duration/2) with the sampling rate defined in 'samplingrate'.
	The value duration/2 is not produced, so as to return an even, highly composite number of samples
	if samplingrate is highly composite.
	"""
	tlim = duration / 2
	t = arange(-round(tlim * samplingrate), round(tlim * samplingrate), 1) / samplingrate    # This gives exact values for integer values of t
	return t

def u(t):
	"""Unit step function.
	"""
	return float_(t>=0)          # float_ is the numpy standard function for conversion to float (equivalent to float64)

def delta(t):
	y = zeros(shape(t))
	y[argmin(abs(t))] = samplingrate
	return y

def tplot(x):
	duration = (size(x) - 1) / samplingrate
	t = timevar(duration)						# Time variable local to this function
	plot(t,x)

#nos antigos lab2,lab3:
#def tplot(x, mode='extended'):
#	"""Plot a continuous-time signal with the correct time scale.
#	If mode == 'extended' (the default), zeros are appended at both ends.
#	If mode != 'extended', only the signal x is plotted, with no appended zeros.
#	"""
#	sizex = size(x)
#	if mode == 'extended':
#		x = append(zeros((sizex-1)/10), x)		# Append zeros in the beginning...
#		x = append(x, zeros((sizex-1)/10))		# ...and in the end
#	duration = size(x) / samplingrate
#	t = timevar(duration)						# Time variable local to this function
#	plot(t,x)

def play(x):
	"""Play an audio signal. The amplitude is clipped to [-1,1].
	"""
	x = clip(x, -1, 1)									# If |x|>1 this would give a strong distortion
	wavwrite('play.wav', samplingrate, (32767*x).astype(int16))
	PlaySound('play.wav',0)								# Set the second argument to 1 to returm immediately to the calling program
	os.remove('play.wav')

def seqsin(*arg):
	"""Generate a sequence of sinusoids with specified frequencies and durations.
	The arguments come in pairs: frequency and duration of each sinusoid.
	A frequency of zero corresponds to a period of silence.
	"""
	if mod(len(arg),2) != 0:
		print('seqsin: Number of arguments must be even')
		return

	t1 = .001 * samplingrate							# Duration, in samples, of the initial taper
	t2 = .01 * samplingrate							# Duration, in samples, of the final taper
	y = zeros(0)									# Initialize output array
	pos = 0										# Initialize position in which we are in the output array
	for nseg in range(int(len(arg) / 2)):
		freq = arg[2*nseg]
		duration = arg[2*nseg+1] * samplingrate			# Duration of this segment, in samples
		i = arange(t1)
		y = append(y, .5 * (1 + cos(pi / t1 * (i - t1))) * sin(2 * pi * freq / samplingrate * i))
		i = arange(t1, duration-t2)
		y = append(y, sin(2 * pi * freq / samplingrate * i))
		i = arange(duration-t2, duration)
		y = append(y, .5 * (1 + cos(pi / t2 * (i - duration + t2 + 1))) * sin(2 * pi * freq / samplingrate * i))
		pos += duration
	return y

def sistema1(x):
	sx = size(x)
	delay = int(floor(.4 * samplingrate))		# Delay, measured in samples
	y = zeros(sx+delay)					# Reserve enough space for the delayed signal
	y[0:sx] = x
	y[delay : sx+delay] += x * .7			
	y = append(zeros(delay), y)
	return y

from sistema2 import *

from sistema3 import *

def FourierTransform(x):
	"""Compute the Fourier transform of the signal in x.
	We use the FFT, and then adjust the output amplitude to correspond to the Fourier transform,
	taking into account the value of samplingrate.
	
	If x is not real, an error message is printed.
	
	If the size of x is not highly factorizable, this function may take quite a bit of time!
	"""
	xf = fft.fftshift(fft.fft(fft.ifftshift(x))) / samplingrate
	return xf

def fplot(xf):
	"""Plot the argument, xf, which is assumed to be in the frequency domain.
	"""
	if not all(imag(xf) == 0):
		print('fplot: The input data are not real. Cannot plot.')
		return
	sizexf = size(xf)
	f = arange(-sizexf/2, sizexf/2-.9999, 1) * 2 * pi * samplingrate / sizexf
	plot(f,xf)
 
def convolution(x1, x2, mode=''):
	"""Compute the linear convolution of two signals.
	If mode != 'full' (the default), the two signals must have the same duration, and the result is cropped to that duration.
	If mode == 'full', the two signals may be of any durations, and the result is not cropped.
	"""

	if mode != 'full' and size(x1) != size(x2):
		print("convolution: The two signals must be of the same duration, or then you must provide the third argument with a value of 'full'.\n")
		return
	
	sizex1 = size(x1)
	sizex2 = size(x2)
	sizey = sizex1 + sizex2 - 1
	
	fftsize = int(2 ** ceil(log2(sizey)))			# First power of 2 that is not smaller than the size of the output
	
	x1 = append(x1, zeros(fftsize - sizex1))	# Extend the input arrays to the fft size
	x2 = append(x2, zeros(fftsize - sizex2))
	
	x1f = fft.fft(x1)						# Compute the Fourier transforms of the inputs
	x2f = fft.fft(x2)
	
#	pdb.set_trace()
	
	y = real(fft.ifft(x1f * x2f))			# Compute the output
	
	if mode == 'full':
		y = y[0:sizey] / samplingrate		# Set the output to the correct size for the full convolution, and to the correct amplitude
	else:
		y = y[round((sizex1-1)/2) : round((sizex1-1)/2) + sizex1] / samplingrate		# Truncate the output to the original size and set it to the correct amplitude
	
	return y

def sistema4(x):									# Lowpass filter with cutoff at approx. 500 Hz
	n = 25									# The length of the filter's response is 2*n+1
	sigma = 6									# Standard deviation of the Gaussian
	h = exp(-linspace(-n, n, 2*n+1)**2 / (2 * sigma**2))	# Gaussian frequency response
	h = h / sum(h) * samplingrate					# Set DC gain = 1
	y = convolution(x, h, 'full')
	return y

def sistema5(x):									# Highpass filter with cutoff at approx. 250 Hz
	n = 25									# The length of the filter's response is 2*n+1
	sigma = 6									# Standard deviation of the Gaussian
	h = exp(-linspace(-n, n, 2*n+1)**2 / (2 * sigma**2))	# Gaussian frequency response
	h = - h / sum(h) * samplingrate					# Set DC gain = 1
	h[n] = h[n] + samplingrate						# Add delta(t) to the impulse response
	y = convolution(x, h, 'full')
	return y

def sistema6(x):							# Passband filter (800-1000Hz, for samplingrate=16000)
	sizex = size(x)
	y = zeros(2 * sizex)					# We compute the response in an interval longer than the size of x
	sizey = size(y)
	y[round(sizey / 4) : round(3 * sizey / 4)] = x			# Center the input signal in y
	yf = fft.fft(fft.ifftshift(y))
	yf[0 : round(sizey / 20)] = 0					# Set the stopband to zero
	yf[round(sizey / 16) : round(15 * sizey / 16)] = 0		# Set the stopband to zero
	yf[round(19 * sizey / 20) : sizey] = 0			# Set the stopband to zero
	y = real(fft.fftshift(fft.ifft(yf)))
	y = y[round(sizey / 4) : round(3 * sizey / 4)]			# Truncate the output to the time interval spanned by the input
	return y

def sample(xc, T):
	"""Sample the signal in xc with sampling period T, which must be a multiple of 1/samplingrate.
	"""
	sampleratio = max(round(T * samplingrate), 1)
	if abs(sampleratio - T * samplingrate) > 1e-10:
		T = sampleratio / samplingrate
		print('sample: T was not a multiple of 1/samplingrate; using T=%7.5f .\n' % T)
	duration = len(xc) / samplingrate
	nsamples = round(2 * floor(duration / (2 * T)))
	xd = zeros(nsamples)
	for n in arange(nsamples):
		xd[n] = xc[round(duration / 2 * samplingrate - nsamples / 2 * sampleratio + n * sampleratio)]
	return xd

def dplot(x, mode='extended'):
	"""Plot the argument as a discrete-time signal.
	If mode == 'extended' (the default), zeros are appended at both ends.
	If mode != 'extended', only the signal x is plotted, with no appended zeros.
	"""
	nsamples = len(x)
	if mode == 'extended':
		x = append(zeros(round(nsamples/10)), x)
		x = append(x, zeros(round(nsamples/10)))

		nsamples = len(x)
	n = arange(nsamples) - nsamples / 2	# Discrete time variable
	stem(n, x)
	
def reconstruct(yd, T):
	"""Reconstruct a continuous time signal from the samples in yd, with sampling period T, which must be a multiple of 1/samplingrate.
	We implement the reconstruction filter in the frequency domain, and therefore use circular convolution, which causes wraparound
	effects in the time domain. To reduce these effects, we compute the response with the double of the final duration, and then
	truncate it to the final duration.
	"""
#	pdb.set_trace()
	sampleratio = max(round(T * samplingrate), 1)
	if abs(sampleratio - T * samplingrate) > 1e-10:
		T = sampleratio / samplingrate
		print('reconstruct: T was not a multiple of 1/samplingrate; using T=%7.5f .\n' % T)
	nsamplesd = len(yd)				# Number of samples of yd
	nsamplesc = nsamplesd * sampleratio	# Number of samples of yc
	zerod = int(floor(nsamplesd / 2))			# Sample that corresponds to t=0 in yd
	zeroc = int(floor(nsamplesc / 2))			# Sample that corresponds to t=0 in yc
	yc = zeros(nsamplesc)
	for n in arange(nsamplesd):
		yc[(n - zerod) * sampleratio + zeroc] = yd[n] * samplingrate		# Put a delta impulse for each discrete sample
	
	# Construct reconstruction filter
#	taperfraction = .001							# Fraction of the bandwidth that is tapered
	bandwidth = round(nsamplesc / (2 * sampleratio))		# Bandwidth of the reconstruction filter in samples
#	taperwidth = round(bandwidth * taperfraction)
	hf = zeros(nsamplesc)
	hf[zeroc-bandwidth : zeroc+bandwidth+1] = 1			# Passband
#	hf[zeroc+bandwidth+1-taperwidth : zeroc+bandwidth+1] = .5 * (1 + cos(linspace(0, pi, taperwidth)))	# Taper the high end of the passband
#	hf[zeroc-bandwidth : zeroc-bandwidth+taperwidth] = .5 * (1 + cos(linspace(-pi, 0, taperwidth)))		# Taper the low end of the passband
	hf = hf * T									# Adjust the gain
	
	# Filter the sequence of deltas
	yc = real(fft.ifft(fft.fft(yc) * fft.ifftshift(hf)))
	
	return yc

import pdb				
import os
from numpy import *
from matplotlib.pyplot import plot, stem
from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread
from soundapi import *

x = wavread('fala.wav')
fala = array(x[1]) / 32767		# x[0] contains the sampling rate of the wav file, which is 16000 for this file
del x

samplingrate = 16000

t1 = timevar(4)
p = .5 * (sign(sin(50*2*pi*t1-1e-15))+sin(50*2*pi*t1))
p[0] = 0;
del t1

samplingrate = 20000 				# for the following definition
# Generate signal to be sampled
xcf = 0j * timevar(4)								# Signal'spectrum, initialized with complex zeros
xcf[39700:39999] = -samplingrate * 1j					# Set the signal's spectrum (will become +-1j)
xcf[40001:40300] = samplingrate * 1j
xc1 = real(fft.ifftshift(fft.ifft(fft.fftshift(xcf))))		# Compute the inverse Fourier transform
del xcf

samplingrate = 16000				# for all the experiments except the section on sampling

print("\n\nSinais e Sistemas - trabalho de laboratório: inicialização concluída.\n")
