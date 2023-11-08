% Clear the workspace and close all figures.
clc;
clear;
close all;

% Load EEG signal and set the sampling frequency (Fs).
load NewEEGSignal.mat;
EEG_signal = NewEEGSignal;
fs = 256;

% Create a time vector.
t = 0:1/fs:(length(EEG_signal)/fs) - 1/fs;

% Question 2-1 (a) - Time domain plot of EEG signal
figure;
subplot(3, 1, 1);
plot(t, EEG_signal, 'r');
title('EEG Signal in Time');
xlabel('Time (sec)');
ylabel('Voltage');


% Fourier Transform of EEG signal
fx = fft(EEG_signal');
m = numel(fx);
fx = abs(fx(1:floor(m/2)));
f = linspace(0, fs/2, floor(m/2));

% Frequency domain plot of EEG signal
subplot(3, 1, 2);
plot(f, fx, 'r');
title('Frequency Content of EEG Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% Short Time Fourier Transform of EEG signal
L = 128;
wind = hamming(L);
Noverlap = L/2;
nfft = floor(m/2);

% Spectrogram of EEG signal
subplot(3, 1, 3);
spectrogram(EEG_signal, wind, Noverlap, nfft, fs, 'yaxis');
title('Spectrogram of EEG Signal (Hamming Window, L=128, Novelap=64, nfft=128)');
colormap('jet');
colorbar;
% Frequency Content by FFT, Periodogram, and Welch
figure;
subplot(3, 1, 1);
plot(f, fx / (floor(m/2)), 'k');
title('FFT of EEG Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;
grid minor;

[Pxx1, f1] = periodogram(EEG_signal, [], nfft, fs);
subplot(3, 1, 2);
plot(f1, Pxx1, 'b');
title('Periodogram of EEG Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;
grid minor;

[Pxx2, f2] = pwelch(EEG_signal, [], [], nfft, fs);
subplot(3, 1, 3);
plot(f2, Pxx2, 'b');
title('Welch Method of EEG Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;
grid minor;

% Question 2-2 (b) - Low-pass filtering and downsampling
[b, a] = butter(3, 64 / (fs/2), 'low');
filtered_EEGsig = filtfilt(b, a, EEG_signal);

M = 2;  % Downsampling factor
EEG_ds = decimate(filtered_EEGsig, M);
new_fs = fs / M;
new_t = 0:1/new_fs:(length(EEG_ds)/new_fs) - 1/new_fs;

% Plot downsampled EEG signal in time domain
figure;
subplot(3, 1, 1);
plot(new_t, EEG_ds, 'r');
title('Downsampled EEG Signal in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

% Fourier Transform of the downsampled EEG signal
fx_new = fft(EEG_ds');
m_new = numel(fx_new);
fx_new = abs(fx_new(1:floor(m_new/2)));
f_new = linspace(0, new_fs/2, floor(m_new/2));

% Frequency domain plot of downsampled EEG signal
subplot(3, 1, 2);
plot(f_new, fx_new, 'r');
title('Frequency Content of Downsampled EEG Signal (fs = 128 Hz)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% Short Time Fourier Transform of downsampled EEG signal
subplot(3, 1, 3);
L = 64;
wind = hamming(L);
Noverlap = L/2;
nfft = floor(m_new/2);
spectrogram(EEG_ds, wind, Noverlap, nfft, new_fs, 'yaxis');
title('Spectrogram of Downsampled EEG Signal (Hamming Window, L=64, Novelap=32, nfft=128, fs = 128 Hz)');

% Overlay original and downsampled signals in time and frequency domain
figure;
subplot(2, 1, 1);
plot(t, EEG_signal, 'b');
hold on;
plot(new_t, EEG_ds, 'r');
xlabel('Time (s)');
ylabel('Amplitude');
title('Comparison of Original EEG Signal (Blue) and Downsampled EEG Signal (Red)');
legend('Original EEG Signal', 'Downsampled EEG Signal');

subplot(2, 1, 2);
plot(f, fx / (floor(m/2)), 'b');
hold on;
plot(f_new, fx_new / (floor(m_new/2)), 'r');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency Domain Comparison (Original vs. Downsampled EEG Signals)');
legend('Original EEG Signal', 'Downsampled EEG Signal');
%%

Ms = [1, 2, 3, 4, 6, 8, 10];  % Different downsampling factors

for i = 1:length(Ms)
    M = Ms(i);
    EEG_ds = decimate(EEG_signal, M);
    new_fs = fs / M;
    new_t = 0:1/new_fs:(length(EEG_ds)/new_fs) - 1/new_fs;

    % Plot downsampled EEG signal in time domain
    figure;
    subplot(2, 1, 1);
    plot(t, EEG_signal, 'b');
    hold on;
    plot(new_t, EEG_ds, 'r');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Comparison of Original EEG Signal (Blue) and Downsampled EEG Signal (Red) - M = ' num2str(M)]);
    legend('Original EEG Signal', 'Downsampled EEG Signal');

    % Perform FFT for both signals
    fx = fft(EEG_signal');
    m = numel(fx);
    fx = abs(fx(1:floor(m/2)));
    f = linspace(0, fs/2, floor(m/2));

    fx_new = fft(EEG_ds');
    m_new = numel(fx_new);
    fx_new = abs(fx_new(1:floor(m_new/2)));
    f_new = linspace(0, new_fs/2, floor(m_new/2));

    % Plot frequency domain comparison
    subplot(2, 1, 2);
    plot(f, fx / (floor(m/2)), 'b');
    hold on;
    plot(f_new, fx_new / (floor(m_new/2)), 'r');
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    title('Frequency Domain Comparison (Original vs. Downsampled EEG Signals)');
    legend('Original EEG Signal', 'Downsampled EEG Signal');
    
    % Perform additional analyses and plotting for each value of M
    % Add your code here for additional analyses or plots as needed.
end
%%
% Question 2-3 (p) - Zero padding
L = length(EEG_signal);
M = 2;  % Downsampling factor
EEG_ds = decimate(filtered_EEGsig, M);
% Compute the N-point Discrete Fourier Transform (DFT) and normalize by L
fft_EEGSignal = fft(EEG_signal, L) / L;

% Take the absolute value and consider only the first half of the spectrum
fft_EEGSignal = abs(fft_EEGSignal(1:floor(L/2)));
f = linspace(0, fs/2, floor(L/2));

% Create a new figure and subplot for the original EEG signal DFT
figure;
subplot(5, 1, 1);
plot(f, fft_EEGSignal, 'b');
title('N-Point DFT Amplitude of Original EEG Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
%%qestion  Question 2-4 (t)

figure;

L = length(EEG_signal);
fft_EEGSignal = fft(EEG_signal,L)/L;
fft_EEGSignal = abs(fft_EEGSignal(1:floor(L/2)));
f = linspace(0,fs/2,floor(L/2));

subplot(5,1,1);plot(f,fft_EEGSignal);title('N point DFT Amlitude of original EEGSignal');xlabel('f(Hz)');ylabel('Amplitude');

N = length(EEG_ds);
fft_EEG_ds1 = fft(EEG_ds,N)/N;
fft_EEG_ds1 = abs(fft_EEG_ds1(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,2);plot(f,fft_EEG_ds1);title('N point DFT Amlitude of Downsampled EEGSignal');
xlabel('f(Hz)');ylabel('Amplitude');

N = N/2; % N=N/2
fft_EEG_ds2 = fft(EEG_ds(1:N),N)/N;
fft_EEG_ds2 = abs(fft_EEG_ds2(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,3);plot(f,fft_EEG_ds2);title('N/2 point DFT Amlitude of Downsampled EEGSignal')
xlabel('f(Hz)');ylabel('Amplitude');

N = N/2; % N=N/4
fft_EEG_ds3 = fft(EEG_ds(1:N),N)/N;
fft_EEG_ds3 = abs(fft_EEG_ds3(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,4);plot(f,fft_EEG_ds3);title('N/4 point DFT Amlitude of Downsampled EEGSignal');
xlabel('f(Hz)');ylabel('Amplitude');

N = N/2; % N=N/8
fft_EEG_ds4 = fft(EEG_ds(1:N),N)/N;
fft_EEG_ds4 = abs(fft_EEG_ds4(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,5);plot(f,fft_EEG_ds4);title('N/8 point DFT Amlitude of Downsampled EEGSignal');
xlabel('f(Hz)');ylabel('Amplitude');

%% zero padding 

figure;

L = length(EEG_signal);
fft_EEGSignal = fft(EEG_signal,L)/L;
fft_EEGSignal = abs(fft_EEGSignal(1:floor(L/2)));
f = linspace(0,fs/2,floor(L/2));

subplot(5,1,1);plot(f,fft_EEGSignal);title('N point DFT Amlitude of original EEGSignal');xlabel('f(Hz)');ylabel('Amplitude');

N = length(EEG_ds);
fft_EEG_ds1 = fft(EEG_ds,N)/N;
fft_EEG_ds1 = abs(fft_EEG_ds1(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,2);plot(f,fft_EEG_ds1);title('N point DFT Amlitude of Downsampled EEGSignal');
xlabel('f(Hz)');ylabel('Amplitude');


EEG_ds2 = zeros(1,N);
N1 = N/2;
EEG_ds2(1:N1) = EEG_ds(1:N1);
fft_EEG_ds2 = fft(EEG_ds2,N)/N;
fft_EEG_ds2 = abs(fft_EEG_ds2(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,3);plot(f,fft_EEG_ds2);title('N point DFT Amlitude of Downsampled EEGSignal(0:N/2) with zero padding')
xlabel('f(Hz)');ylabel('Amplitude');

EEG_ds3 = zeros(1,N);
N2 = N/4;
EEG_ds3(1:N2) = EEG_ds(1:N2);
fft_EEG_ds3 = fft(EEG_ds3,N)/N;
fft_EEG_ds3 = abs(fft_EEG_ds3(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,4);plot(f,fft_EEG_ds3);title('N point DFT Amlitude of Downsampled EEGSignal(0:N/4) with zero padding');
xlabel('f(Hz)');ylabel('Amplitude');

EEG_ds4 = zeros(1,N);
N3 = N/8;
EEG_ds4(1:N3) = EEG_ds(1:N3);
fft_EEG_ds4 = fft(EEG_ds4,N)/N;
fft_EEG_ds4 = abs(fft_EEG_ds4(1:floor(N/2)));
f = linspace(0,fs/2,floor(N/2));
subplot(5,1,5);plot(f,2*fft_EEG_ds4);title('N point DFT Amlitude of Downsampled EEGSignal(0:N/8) with zero padding');
xlabel('f(Hz)');ylabel('Amplitude');

