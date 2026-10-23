# OFDM Basics

Orthogonal frequency-division multiplexing (OFDM) splits a wideband channel into many narrow subcarriers that are spaced so they are mathematically orthogonal. Each subcarrier carries a low-rate stream of symbols, usually drawn from a QAM or PSK constellation, so the symbol duration on each subcarrier is long compared with the channel's delay spread.

## IFFT and FFT

The transmitter maps a block of constellation symbols onto the subcarriers and applies an inverse fast Fourier transform (IFFT) to produce the time-domain OFDM symbol. The receiver applies a fast Fourier transform (FFT) to recover the symbols on each subcarrier. Using the FFT makes OFDM cheap to implement even with thousands of subcarriers.

## Cyclic prefix

Before transmission, the last part of each OFDM symbol is copied to its front. This copy is the cyclic prefix. If the cyclic prefix is longer than the channel's delay spread, the channel's linear convolution behaves like a circular convolution, inter-symbol interference is removed, and each subcarrier sees a single complex gain. The receiver can then correct the channel with a one-tap equalizer per subcarrier, dividing each received value by the estimated channel gain. The cost is overhead: the cyclic prefix carries no new data.

## Where OFDM is used

OFDM is the physical-layer waveform of Wi-Fi (IEEE 802.11a/g/n/ac/ax), the LTE downlink and 5G New Radio, which uses cyclic-prefix OFDM (CP-OFDM). It is also used in DVB-T digital television.

## Drawbacks

OFDM signals have a high peak-to-average power ratio (PAPR), which forces power amplifiers to operate with back-off and lowers efficiency. OFDM is also sensitive to carrier frequency offset and phase noise, because these break the orthogonality between subcarriers and cause inter-carrier interference.
