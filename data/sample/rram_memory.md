# Resistive RAM (RRAM)

Resistive random-access memory (RRAM) stores a bit as the resistance of a thin insulating layer sandwiched between two electrodes, a metal-insulator-metal (MIM) stack. Hafnium oxide (HfO2) is a common switching layer because it is already used in CMOS manufacturing; titanium nitride (TiN) is a common electrode material.

## How switching works

A fresh device first needs a forming step: a relatively high voltage creates a conductive filament made of oxygen vacancies through the oxide. After forming, the device switches between two states. The SET operation reconnects the filament and puts the cell in the low-resistance state (LRS). The RESET operation, in bipolar devices done with the opposite voltage polarity, ruptures the filament near one electrode and returns the cell to the high-resistance state (HRS). Reading uses a small voltage that does not disturb the state.

## Crossbar arrays

RRAM cells can be placed at the intersections of perpendicular word lines and bit lines, forming a dense crossbar array. A known problem in passive crossbars is sneak-path current: when reading one cell, current also leaks through neighbouring low-resistance cells and corrupts the reading. Adding a selector device or transistor in series with each cell (1T1R or 1S1R) suppresses sneak paths.

## Fabrication

Because RRAM can be built at low temperature, it fits into the back end of line (BEOL) of a CMOS process. Thin, uniform HfO2 layers are typically deposited with atomic layer deposition (ALD), while metal electrodes are often deposited by physical vapour deposition (PVD), such as sputtering. Variability between cycles and between devices remains one of the main research challenges.
