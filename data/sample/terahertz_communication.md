# Terahertz Communication

The terahertz (THz) band is usually defined as frequencies from 0.1 THz to 10 THz, between millimetre waves and infrared light. Its main attraction is bandwidth: contiguous channels tens of gigahertz wide are available, which makes data rates above 100 Gbit/s possible over a single link. For this reason THz links are considered for 6G backhaul, data centres and short-range kiosk downloads.

## Propagation challenges

THz signals suffer high free-space path loss, which grows with frequency, and additional molecular absorption, mostly from water vapour, at specific absorption peaks. Practical links therefore use the lower-loss transmission windows between those peaks, operate over short distances, and rely on highly directional antennas to concentrate energy into narrow beams. Narrow beams in turn make beam alignment and tracking important.

## Photonic generation

One way to generate THz carriers is photonic: two lasers with slightly different optical wavelengths are combined and shone onto a fast photodiode, often a uni-travelling-carrier photodiode (UTC-PD). The photodiode current oscillates at the difference frequency of the two lasers, which falls in the THz range. Tuning one laser tunes the THz carrier, and data can be modulated onto the optical signal before conversion. Photonic approaches benefit from mature fibre-optic components and low phase noise.

## Receivers

THz receivers often use Schottky barrier diode mixers or envelope detectors. Because the hardware has strong nonlinearities and impairments at these frequencies, researchers increasingly apply digital signal processing and machine learning based equalizers to recover the data.
