# Integrated Sensing and Communication (ISAC)

Integrated sensing and communication (ISAC) uses one waveform, one spectrum allocation and largely shared hardware for two jobs at once: transmitting data and sensing the environment like a radar. Instead of building separate radar and communication systems, an ISAC transmitter sends a communication signal and also processes its echoes to detect objects and estimate their range and velocity.

## OFDM as a sensing waveform

OFDM is a popular ISAC waveform because its structure is easy to process for radar. After removing the known transmitted symbols from the received echo, the delay of a target appears as a linear phase change across subcarriers, so an IFFT across subcarriers gives a range profile. The Doppler shift of a moving target appears as a phase change from one OFDM symbol to the next, so an FFT across symbols gives the target's velocity.

## Resolution

Range resolution depends on bandwidth: the resolution is c / (2B), where c is the speed of light and B is the signal bandwidth. With 1 GHz of bandwidth the range resolution is 15 cm. Velocity resolution improves with a longer observation time, that is, more OFDM symbols.

## Why it matters

ISAC is widely studied as a key feature of 6G networks, where base stations and devices could provide sensing as a service, for example for traffic monitoring or gesture recognition. The IEEE 802.11bf task group is standardising WLAN sensing for Wi-Fi. A central design question is the trade-off between sensing accuracy and data rate, since resources spent on sensing are not available for data.
