# EEG Signal Processing Project

This project is about applying various signal processing techniques to an EEG signal, such as time domain analysis, frequency domain analysis, low-pass filtering, downsampling, zero-padding, and the Discrete Fourier Transform (DFT). The project also explores the effects of different window functions, overlapping points, window length, and the number of DFT points on the time-frequency representation of the signal using the Short-Time Fourier Transform (STFT) method.

## Data

The data used in this project is an EEG signal with a sampling frequency of 256 Hz. The EEG signal is processed through different techniques and compared with the original signal.

## Methods

The project report contains the following sections:

- **Time Domain Analysis**: This section examines the raw EEG signal in the time domain and its frequency content with spectrogram.
- **Frequency Domain Analysis**: This section performs a Fourier Transform of the EEG signal to analyze its frequency components.
- **Short Time Fourier Transform (STFT)**: This section applies the STFT to the EEG signal using a Hamming window with specific parameters and displays the spectrogram.
- **Low-Pass Filtering and Downsampling**: This section applies low-pass filtering to the EEG signal and downsample it by a specified factor. The resulting signal is named EEG_ds.
- **Spectrogram of Downsampled EEG Signal**: This section generates the spectrogram of the downsampled EEG signal with different parameters.
- **Analysis with Different Downsampling Factors**: This section explores the effects of various downsampling factors (1, 2, 3, 4, 6, 8, 10) on the EEG signal and compares the original EEG signal with the downsampled versions.
- **Zero Padding and DFT**: This section discusses the application of zero padding to the downsampled EEG signal and analyzes the DFT using different fractions of N.
## Results

The project report presents the results of each section in the form of figures and tables, along with descriptions and discussions. The figures show the time-domain plot, frequency content, spectrogram, and DFT amplitude of the EEG signal and its processed versions. The tables summarize the effects of different parameters on the time and frequency resolution of the spectrogram.
! [] (signal.png)
! [] (spectrogram water plots for 4 windows.jpg)
! [] (Spectrogram.jpg)
! [] (windows.jpg)
! [] (STFT 4 windows heatmat.jpg)
! [] (EEG signal tme freq spectrom.png)
! [] (EEG freq content.png)
! [] (m = 2.png)
! [] (EEG signal tme freq spectrom.png)
! [] (N-point DFT.png)
! [] (zerro_padding.png)



## Conclusion

The project report concludes with a summary of the main findings and insights from the signal processing techniques. The report also discusses the implications and limitations of the methods and suggests possible directions for future work.

## References

The project report cites the following sources:
-  R. G. Lyons, Understanding Digital Signal Processing, 3rd ed. Upper Saddle River, NJ: Prentice Hall, 2011.
-  S. K. Mitra, Digital Signal Processing: A Computer-Based Approach, 4th ed. New York, NY: McGraw-Hill, 2011.
