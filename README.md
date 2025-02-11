# Simple ANC

This project implements an active noise cancellation system in Python. It leverages adaptive filtering, frequency-domain processing, and a hybrid approach to minimize undesirable noise in real time.

## Features

- **Adaptive Noise Cancellation:** Uses the Normalized Least Mean Square (NLMS) algorithm to filter noise based on previous signal values.
- **Frequency Domain Processing:** Applies FFT-based techniques to reduce noise in specific frequency ranges.
- **Hybrid Mode:** Combines the benefits of both adaptive filtering and frequency-domain processing.
- **Real-time Audio Processing:** Processes audio input with low latency for immediate noise cancellation.
- **Performance Monitoring:** Tracks metrics such as latency and clipping events to help diagnose and fine-tune system performance.
- **Visualizations:** (Optional) Visualize signal processing results and system statistics.

## Code Structure

- **anc.py**  
  Contains the core implementation, including:
  - `ANCAudioProcessor`: Provides DSP operations and adaptive filtering functions.
  - `ANCSystem`: Manages the overall noise cancellation system, including configuration, audio streaming, and performance monitoring.

## Prerequisites

Before running the ANC system, make sure you have the following installed:

- Python 3.6 or higher
- [NumPy](https://numpy.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/) (for visualization support)

You can install the required packages using pip:
```bash
pip install numpy sounddevice scipy matplotlib
```


## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/anc-system.git
   cd anc-system
   ```

2. **Configure and Run**

   The system is configured with default parameters in the code. You can override these parameters via command-line arguments or by modifying the configuration dictionary in the `ANCSystem` class.

   To view available options, run:

   ```bash
   python3 anc.py --help
   ```

3. **Executing the System**

   Simply run the main file to start the noise cancellation process:

   ```bash
   python3 anc.py
   ```

## Configuration Options

The default configuration is as shown in the code:

- **sample_rate:** 44100 Hz
- **block_size:** 1024 samples
- **channels:** 1 (mono)
- **mode:** 'hybrid' (can be set to 'adaptive', 'frequency', or 'simple')
- **filter_order:** 4
- **low_cutoff:** 30 Hz
- **high_cutoff:** 400 Hz
- **adaptation_rate:** 0.1
- **visualization:** True

You can modify these settings directly in the code or pass overrides through command-line arguments.

## Performance Monitoring

The system monitors:
- **Latency:** The time (in milliseconds) between audio input and noise-canceled output.
- **Clipping Events:** Occurrences where the input signal exceeds the maximum limits of the system (which may lead to audio distortion).

If you encounter warnings like "Input signal clipping detected", consider lowering the input volume or adjusting the microphone sensitivity.
