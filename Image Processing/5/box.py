import numpy as np
import matplotlib.pyplot as plt

N = 100000
T = 1.0 / 10000.0
x = np.linspace(0.0, N*T, N)
y = np.zeros(x.shape)
for i in range(x.shape[0]):
    if -1 < x[i] < 1:
        y[i] = 1.0

plt.plot(x, y)
plt.xlim(-3, 3)
plt.title(r'Box function')
plt.savefig('rectangular.png')
plt.close()

yf = np.fft.fft(y)
yf = np.fft.fftshift(yf)
xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)

fig, ax = plt.subplots()
ax.plot(xf, np.abs(yf))
plt.xlim(-10,10)
plt.title('FFT Magnitude Spectrum')
plt.grid()
plt.savefig('fft_rectangular_abs.png', bbox_inches='tight')
plt.close()


fig, ax = plt.subplots()
ax.plot(xf, np.real(yf) )
plt.xlim(-10,10)
plt.title('FFT (rectangular function) Real')
plt.grid()
plt.savefig('fft_rectangular_real.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
ax.plot(xf, np.imag(yf) )
plt.xlim(-10,10)
plt.title('FFT (rectangular function) Imaginary')
plt.grid()
plt.savefig('fourrier_transform_rectangular_imag.png', bbox_inches='tight')
plt.close()
