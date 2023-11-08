close all 
clear
%% 
% Parameters
fs = 1000;           % Sampling frequency (Hz)
T = 2;               % Signal duration (s)
t = 0:1/fs:2;
f0 = 100;  % Initial frequency (Hz)
beta = 100;% Frequency increase rate
f1 = 500

%%
% Generate the signal
f_t = (f0 + (f1-f0)*t.^2);
signal = chirp(t,100,2,500,'quadratic');
%%
% a) Plot the signal
figure;
plot(t, signal);
title('Signal');
xlabel('Time (s)');
ylabel('Amplitude');
%%
% b) Generate and compare different windows
L = 128;  % Window length
rect_window = rectwin(L);
triangular_window = triang(L);
gauss_window = gausswin(L);
hamming_window = hamming(L);
%%
figure;
wvtool(rect_window);
wvtool(triangular_window);
wvtool(gauss_window);
wvtool(hamming_window);
wvtool (rect_window,triangular_window,gauss_window,hamming_window) % plot windows
%%
% c) Calculate and plot the time-frequency spectrum using STFT
nfft = L;      % Number of DFT points
noverlap = 0;  % Number of overlapping points

figure;
for i = 1:4
    subplot(2, 2, i);
    switch i
        case 1
            window = rect_window;
            window_name = 'Rectangular Window';
        case 2
            window = triangular_window;
            window_name = 'Triangular Window';
        case 3
            window = gauss_window;
            window_name = 'Gaussian Window';
        case 4
            window = hamming_window;
            window_name = 'Hamming Window';
    end

    [S, F, T, P] = spectrogram(signal, window, noverlap, nfft, fs);

    imagesc(T, F, 10*log10(abs(P)));
    title(['Spectrogram using ', window_name]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end

figure;
for i = 1:4
    subplot(2, 2, i);
    switch i
        case 1
            window = rect_window;
            window_name = 'Rectangular Window';
        case 2
            window = triangular_window;
            window_name = 'Triangular Window';
        case 3
            window = gauss_window;
            window_name = 'Gaussian Window';
        case 4
            window = hamming_window;
            window_name = 'Hamming Window';
    end

    [s, f, t] = stft(signal, fs, window = window);
    mesh(t, f, abs(s).^2);
    view(2), axis tight;
    title(['STFT Mesh using ', window_name]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end

figure;
for i = 1:4
    subplot(2, 2, i);
    switch i
        case 1
            window = rect_window;
            window_name = 'Rectangular Window';
        case 2
            window = triangular_window;
            window_name = 'Triangular Window';
        case 3
            window = gauss_window;
            window_name = 'Gaussian Window';
        case 4
            window = hamming_window;
            window_name = 'Hamming Window';
    end

    stft(signal, fs, window = window);
    title(['STFT using ', window_name]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end

figure;
for i = 1:4
    subplot(2, 2, i);
    switch i
        case 1
            window = rect_window;
            window_name = 'Rectangular Window';
        case 2
            window = triangular_window;
            window_name = 'Triangular Window';
        case 3
            window = gauss_window;
            window_name = 'Gaussian Window';
        case 4
            window = hamming_window;
            window_name = 'Hamming Window';
    end

    [S, F, T, P] = spectrogram(signal, window, noverlap, nfft, fs);

    mesh(T, F, abs(P).^2);
    title(['Spectrogram Mesh using ', window_name]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end

figure;
for i = 1:4
    subplot(2, 2, i);
    switch i
        case 1
            window = rect_window;
            window_name = 'Rectangular Window';
        case 2
            window = triangular_window;
            window_name = 'Triangular Window';
        case 3
            window = gauss_window;
            window_name = 'Gaussian Window';
        case 4
            window = hamming_window;
            window_name = 'Hamming Window';
    end

    [S, F, T, P] = spectrogram(signal, window, noverlap, nfft, fs);

    waterplot(S, F, T);
    title(['Waterfall Plot using ', window_name]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end



%
%%

% d) Change the number of overlapping points
figure;
Noverlaps = [0, 64,127];
for i = 1:numel(Noverlaps)
    subplot(3, 1, i);
    nooverlap = Noverlaps(i);

    [S, F, T, P] = spectrogram(signal, triangular_window , nooverlap, nfft, fs);

    imagesc(T, F, 10*log10(abs(P)));
    title(['Noverlap = ', num2str(nooverlap),'with triangular window']);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;
end
%%
L = [32 128 512];
Noverlap = L-1;
fs = 1000;
nfft = L;
figure;

for i = 1:3
    
    % spectogram using triangular window
    win2 = triang(L(i));
    subplot(3,1,i);
    spectrogram(signal,win2,Noverlap(i),nfft(i),fs,'yaxis');
    str = sprintf('Spectogram of quadratic shirp with triangular window & L= %d, Novelap= %d, nfft= %d',L(i),Noverlap(i),L(i));
    title(str);
    axis xy;
    colormap('jet');
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
   

end


%%
% f) Change the number of DFT points
figure;
L = 128;
nffts = [L, 2*L, 4*L];
for i = 1:numel(nffts)
    subplot(3, 1, i);
    nfft = nffts(i);
    nooverlap = L/2;
    [S, F, T, P] = spectrogram(signal, hamming_window, nooverlap, nfft, fs);

    imagesc(T, F, 10*log10(abs(P)));
    title(['nfft = ', num2str(nfft)]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis xy;
    colormap('jet');
    colorbar;

end

%%
% Set the sampling frequency (Fs).
fs = 1000;

% Define parameters for the STFT analysis.
wind_len = 128; % Window length
overlap = 100; % Overlap
nfft = 4 * wind_len; % Number of DFT points

% Create a figure for visualization.
figure;

% Subplot 1: Spectrogram using a rectangular window.
win1 = rectwin(wind_len);
subplot(2, 1, 1);
spectrogram(signal, win1, overlap, nfft, fs, 'yaxis');
colormap('jet');
colorbar;
title('STFT of Quadratic Chirp with Rectangular Window (Using spectrogram function)');

% Subplot 2: Spectrogram using custom MySpectrogram function.
[S, f, t] = MySpectrogram(signal, wind_len, nfft, fs, overlap);
subplot(2, 1, 2);
surf(t, f, 10 * log10(abs(S)), 'EdgeColor', 'none');
axis xy;
axis tight;
colormap('jet');
colorbar;
view(0, 90);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(['STFT with MySpectrogram, Window Length = ', num2str(wind_len), ...
    ', Overlap = ', num2str(overlap), ', nfft = ', num2str(nfft)]);

% Custom function MySpectrogram for STFT computation.
function [S, Freq, Time] = MySpectrogram(Y, wind_len, nfft, fs, overlap)
    % Calculate the time increment (sampling interval).
    dt = 1 / fs;

    % Get the length of the input signal.
    sig_len = length(Y);

    % Calculate the number of windows (frames) in the signal.
    K = floor((sig_len - overlap) / (wind_len - overlap));

    % Create a rectangular window of specified length.
    window = rectwin(wind_len);

    % Initialize the spectrogram matrix.
    S = zeros(nfft, K);

    % Calculate the offset based on the overlap.
    offset = fix((1 - overlap / wind_len) * wind_len);

    % Iterate over the signal and compute the STFT for each window.
    for i = 1:K
        start = (i - 1) * offset;
        y_tmp = Y(1 + start : start + wind_len);

        % Apply the window function to the frame.
        y_tmp = y_tmp .* window';

        % Compute the FFT for the frame.
        y_f = fft(y_tmp, nfft);

        % Store the FFT result in the spectrogram matrix.
        S(:, i) = y_f;

        % Calculate the corresponding time for this frame.
        Time(i) = (start + (wind_len / 2)) * dt;
    end

    % Determine the number of positive frequencies based on the FFT length.
    if mod(nfft, 2) % If N is odd.
        Nf = (nfft + 1) / 2;
    else % If N is even.
        Nf = nfft / 2 + 1;
    end

    % Retain only the positive frequencies and adjust frequency values.
    S = S(1:Nf, :);
    Freq = (0:Nf - 1) * fs / nfft;
end

function waterplot(s,f,t)
% Waterfall plot of spectrogram
    waterfall(f,t,abs(s)'.^2)
    set(gca,XDir="reverse",View=[30 50])
    xlabel("Frequency/\pi")
    ylabel("Samples")
end




    


    
