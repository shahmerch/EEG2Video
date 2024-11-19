clc;
%obtaining FFT data from loaded voltage signal
fft_data = power_fftscope(out.ScopeData) ;
%changing maxFrequency parameter in fit_data structure
fft_data.maxFrequency = 2000;
%run analysis with new defined maxFrequency value
fft_data = power_fftscope(fft_data) ;
%extracting frequency and corresponding magnitude from fft_data structure
freq = fft_data.freq;
mag = fft_data.mag;
%eliminating the DC component

mag(1) = 0;
%Plotting the harmonic spectrum 
bar(freq, mag);
xlim ([0 2000]);
title ('FFT spectrum of Y_(Ao)');
xlabel ('Frequency (Hz)'); ylabel ('Magnitude (V)');
%Calculating THD as a percentage of peak fundamental frequency component
THD_percentage = 100*sqrt(2)/mag(2) * sqrt(sum((mag(3: end)/sqrt(2)).^2))