#!/usr/bin/env python3
"""
Advanced Active Noise Cancellation (ANC) System
--------------------------------------------
A sophisticated implementation of real-time noise cancellation using adaptive filtering
and multiple noise reduction strategies.

Features:
- Multiple noise cancellation modes
- Adaptive filtering with various algorithms
- Real-time visualization of signal processing
- Device selection and configuration
- Performance monitoring and optimization
"""

import numpy as np
import sounddevice as sd
from scipy import signal
import threading
import time
import sys
import argparse
import matplotlib.pyplot as plt
from collections import deque
import json
import os

class ANCAudioProcessor:
    """Audio processing component handling the core DSP operations."""
    
    def __init__(self, mode='adaptive'):
        self.mode = mode
        self.reset_state()
        
    def reset_state(self):
        """Reset internal state variables."""
        self.prev_input = np.zeros(1024)
        self.prev_output = 0
        self.error_history = deque(maxlen=1000)
        self.weights = np.zeros(256)
        
    def nlms_filter(self, input_signal, step_size=0.1, eps=1e-6):
        """Normalized Least Mean Square adaptive filter."""
        output = np.dot(self.weights, self.prev_input[:len(self.weights)])
        error = input_signal - output
        
        # Update weights using NLMS
        norm = np.dot(self.prev_input[:len(self.weights)], self.prev_input[:len(self.weights)])
        if norm > eps:
            self.weights += step_size * error * self.prev_input[:len(self.weights)] / norm
            
        # Update signal history
        self.prev_input = np.roll(self.prev_input, 1)
        self.prev_input[0] = input_signal
        self.error_history.append(error)
        
        return output

    def process_block(self, input_block, mode='adaptive'):
        """Process a block of audio data using the selected algorithm."""
        if mode == 'adaptive':
            return self._process_adaptive(input_block)
        elif mode == 'frequency':
            return self._process_frequency_domain(input_block)
        elif mode == 'hybrid':
            return self._process_hybrid(input_block)
        else:
            return self._process_simple(input_block)
    
    def _process_adaptive(self, input_block):
        """Adaptive filtering with NLMS algorithm."""
        output_block = np.zeros_like(input_block)
        for i in range(len(input_block)):
            output_block[i] = self.nlms_filter(input_block[i])
        return -output_block  # Phase inversion
    
    def _process_frequency_domain(self, input_block):
        """Frequency domain processing using FFT."""
        # Apply FFT
        spectrum = np.fft.rfft(input_block)
        
        # Apply frequency-dependent processing
        freq_response = np.ones_like(spectrum)
        freq_response[0:int(len(freq_response)*0.1)] *= 0.8  # Reduce low frequencies
        processed_spectrum = spectrum * freq_response
        
        # Inverse FFT
        output_block = np.fft.irfft(processed_spectrum)
        return -output_block[:len(input_block)]
    
    def _process_hybrid(self, input_block):
        """Hybrid approach combining adaptive and frequency domain processing."""
        adaptive_output = self._process_adaptive(input_block)
        freq_output = self._process_frequency_domain(input_block)
        return 0.6 * adaptive_output + 0.4 * freq_output
    
    def _process_simple(self, input_block):
        """Simple phase inversion with smoothing."""
        return -input_block * 0.5

class ANCSystem:
    """Advanced Active Noise Cancellation System."""
    
    DEFAULT_CONFIG = {
        'sample_rate': 44100,
        'block_size': 1024,
        'channels': 1,
        'mode': 'hybrid',
        'filter_order': 4,
        'low_cutoff': 30,
        'high_cutoff': 400,
        'adaptation_rate': 0.1,
        'visualization': True
    }
    
    def __init__(self, **kwargs):
        """Initialize the ANC system with optional configuration."""
        # Load configuration
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        
        # Initialize system state
        self.is_running = False
        self.stream = None
        self._init_audio_processing()
        self._init_visualization()
        
        # Performance monitoring
        self.latency = 0
        self.clipping_count = 0
        self._last_callback_time = None
        
        # Statistics
        self.stats = {
            'avg_latency': 0,
            'max_latency': 0,
            'clipping_events': 0,
            'runtime': 0
        }
    
    def _init_audio_processing(self):
        """Initialize audio processing components."""
        self.processor = ANCAudioProcessor(mode=self.config['mode'])
        
        # Initialize filters
        nyquist = self.config['sample_rate'] / 2
        self.b, self.a = signal.butter(
            self.config['filter_order'],
            [self.config['low_cutoff']/nyquist, 
             self.config['high_cutoff']/nyquist],
            btype='band'
        )
    
    def _init_visualization(self):
        """Initialize real-time visualization if enabled."""
        if self.config['visualization']:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
            self.lines = {}
            self.lines['input'] = self.ax1.plot([], [], label='Input')[0]
            self.lines['output'] = self.ax1.plot([], [], label='Anti-noise')[0]
            self.lines['error'] = self.ax2.plot([], [], label='Error')[0]
            self.ax1.set_title('Signal Waveforms')
            self.ax2.set_title('Error History')
            self.ax1.legend()
            self.ax2.legend()
            plt.tight_layout()
    
    def update_visualization(self, input_data, output_data):
        """Update the real-time visualization plots."""
        if not self.config['visualization']:
            return
            
        try:
            # Update signal plots
            x = np.arange(len(input_data))
            self.lines['input'].set_data(x, input_data)
            self.lines['output'].set_data(x, output_data)
            
            # Update error plot
            error_history = list(self.processor.error_history)
            if error_history:
                self.lines['error'].set_data(
                    np.arange(len(error_history)), 
                    error_history
                )
            
            # Adjust plot limits
            for ax in [self.ax1, self.ax2]:
                ax.relim()
                ax.autoscale_view()
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def process_audio(self, indata, outdata, frames, time_info, status):
        """Real-time audio processing callback function."""
        if status:
            print(f"Status: {status}")
            
        try:
            # Timing and performance monitoring
            current_time = time.perf_counter()
            if self._last_callback_time is not None:
                self.latency = (current_time - self._last_callback_time) * 1000
                self.stats['avg_latency'] = (self.stats['avg_latency'] + self.latency) / 2
                self.stats['max_latency'] = max(self.stats['max_latency'], self.latency)
            self._last_callback_time = current_time
            
            # Clipping detection
            if np.any(np.abs(indata) > 0.95):
                self.clipping_count += 1
                self.stats['clipping_events'] = self.clipping_count
            
            # Signal processing
            input_signal = indata.flatten()
            filtered_signal = signal.lfilter(self.b, self.a, input_signal)
            anti_noise = self.processor.process_block(filtered_signal, self.config['mode'])
            
            # Apply safety limits and smoothing
            anti_noise = np.clip(anti_noise, -0.7, 0.7)
            
            # Update visualization
            self.update_visualization(input_signal, anti_noise)
            
            # Output the anti-noise signal
            outdata[:] = anti_noise.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")
            outdata.fill(0)
    
    def list_audio_devices(self):
        """List all available audio devices with details."""
        devices = sd.query_devices()
        print("\nAvailable Audio Devices:")
        print("-" * 60)
        for i, dev in enumerate(devices):
            print(f"Device {i}:")
            print(f"  Name: {dev['name']}")
            print(f"  Channels: {dev['max_input_channels']} in, {dev['max_output_channels']} out")
            print(f"  Sample Rates: {dev['default_samplerate']}")
            print(f"  Default: {'* ' if dev['name'] == sd.default.device[0] else ''}")
            print()
        return devices
    
    def start(self, input_device=None, output_device=None):
        """Start the ANC system with optional device selection."""
        try:
            # Device setup
            devices = self.list_audio_devices()
            
            # Stream initialization
            self.stream = sd.Stream(
                device=(input_device, output_device),
                samplerate=self.config['sample_rate'],
                blocksize=self.config['block_size'],
                channels=self.config['channels'],
                dtype=np.float32,
                callback=self.process_audio,
                latency='low'
            )
            
            # Start processing
            self.is_running = True
            self.stream.start()
            start_time = time.time()
            
            print(f"\nANC System Started")
            print(f"Mode: {self.config['mode']}")
            print(f"Sample Rate: {self.config['sample_rate']} Hz")
            print(f"Block Size: {self.config['block_size']} samples")
            print(f"Frequency Range: {self.config['low_cutoff']}-{self.config['high_cutoff']} Hz")
            print("\nPress Ctrl+C to stop")
            
            try:
                while self.is_running:
                    self.stats['runtime'] = time.time() - start_time
                    status = (
                        f"\rRuntime: {self.stats['runtime']:.1f}s | "
                        f"Latency: {self.latency:.1f}ms (avg: {self.stats['avg_latency']:.1f}ms, "
                        f"max: {self.stats['max_latency']:.1f}ms) | "
                        f"Clipping: {self.clipping_count}"
                    )
                    print(status, end='', flush=True)
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nStopping ANC system...")
                self.stop()
                
        except Exception as e:
            print(f"Error starting ANC system: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop the ANC system and cleanup resources."""
        self.is_running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        
        if self.config['visualization']:
            plt.close('all')
        
        # Save statistics
        self._save_stats()
        
        print("\nANC System stopped")
        self._print_final_stats()
    
    def _save_stats(self):
        """Save performance statistics to a file."""
        try:
            with open('anc_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Error saving statistics: {e}")
    
    def _print_final_stats(self):
        """Print final performance statistics."""
        print("\nPerformance Statistics:")
        print("-" * 40)
        print(f"Total Runtime: {self.stats['runtime']:.1f} seconds")
        print(f"Average Latency: {self.stats['avg_latency']:.1f} ms")
        print(f"Maximum Latency: {self.stats['max_latency']:.1f} ms")
        print(f"Clipping Events: {self.stats['clipping_events']}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced Active Noise Cancellation System')
    parser.add_argument('--mode', choices=['adaptive', 'frequency', 'hybrid', 'simple'],
                      default='hybrid', help='Noise cancellation mode')
    parser.add_argument('--sample-rate', type=int, default=44100,
                      help='Sample rate in Hz')
    parser.add_argument('--block-size', type=int, default=1024,
                      help='Block size in samples')
    parser.add_argument('--visualization', action='store_true',
                      help='Enable real-time visualization')
    parser.add_argument('--input-device', type=int,
                      help='Input device ID')
    parser.add_argument('--output-device', type=int,
                      help='Output device ID')
    return parser.parse_args()

def main():
    """Main function to run the ANC system."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create and start ANC system with parsed configuration
        anc = ANCSystem(
            mode=args.mode,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            visualization=args.visualization
        )
        
        # Start the system
        anc.start(
            input_device=args.input_device,
            output_device=args.output_device
        )
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
