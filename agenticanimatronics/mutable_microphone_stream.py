import noisereduce as nr
import numpy as np
import assemblyai as aai

import pyaudio
from array import array
from typing import Optional
from loguru import logger


class MutableMicrophoneStream:
    def __init__(self, sample_rate: int = 44_100, device_index: Optional[int] = None, threshold: int = 500):
        """
        Creates a stream of audio from the microphone.

        Args:
            sample_rate: The sample rate to record audio at.
            device_index: The index of the input device to use. If None, uses the default device.
            threshold: The threshold input to send audio
        """
        self._pyaudio = pyaudio.PyAudio()
        logger.info(f"connecting to default device {self._pyaudio.get_default_input_device_info()}")
        self.sample_rate = sample_rate
        self.is_muted = False
        self.threshold = threshold

        # Reduce chunk size to prevent buffer overflow
        self._chunk_size = int(self.sample_rate * 0.05)  # 50ms instead of 100ms

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=device_index,
            # Add these parameters to help prevent overflow
            start=False,  # Don't start immediately
            stream_callback=None,
        )

        # Start the stream after creation
        self._stream.start_stream()
        self._open = True

    def __iter__(self):
        """
        Returns the iterator object.
        """
        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        if not self._open:
            raise StopIteration

        try:
            if not self.is_muted:
                # Use exception_on_overflow=False to handle overflow gracefully
                data = self._stream.read(self._chunk_size, exception_on_overflow=False)

                # Check if we got the expected amount of data
                expected_bytes = self._chunk_size * 2  # 2 bytes per sample for paInt16
                if len(data) < expected_bytes:
                    # Pad with zeros if we didn't get enough data
                    data += b'\x00' * (expected_bytes - len(data))

                data_chunk = array('h', data)
                vol = max(data_chunk) if data_chunk else 0

                if vol >= self.threshold:
                    return data
                else:
                    return b'\x00' * (self._chunk_size * 2)
            else:
                # Still read from stream when muted to prevent buffer overflow
                _ = self._stream.read(self._chunk_size, exception_on_overflow=False)
                return b'\x00' * (self._chunk_size * 2)

        except KeyboardInterrupt:
            raise StopIteration
        except Exception:
            logger.exception("Error reading from microphone")
            # Return silence on error to keep the stream going
            return b'\x00' * (self._chunk_size * 2)

    def close(self):
        """
        Closes the stream.
        """
        self._open = False

        if hasattr(self, '_stream') and self._stream:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()

        if hasattr(self, '_pyaudio') and self._pyaudio:
            self._pyaudio.terminate()

    def mute(self):
        """Mute the microphone (produce silence)"""
        self.is_muted = True
        logger.info("Microphone muted")

    def unmute(self):
        """Unmute the microphone"""
        self.is_muted = False
        logger.info("Microphone unmuted")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def toggle_mute(self):
        """Toggle between mute and unmute"""
        if self.is_muted:
            self.unmute()
        else:
            self.mute()


class NoiseReducedMicrophoneStream:
    def __init__(self, sample_rate, energy_threshold=0.01):
        self.microphone_stream = aai.extras.MicrophoneStream(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.array([])
        self.buffer_size = int(sample_rate * 0.5)  # 0.5 seconds buffer
        self.energy_threshold = energy_threshold  # Minimum energy threshold
        self.is_muted = False

    def __iter__(self):
        return self

    def __next__(self):
        # Get audio chunk from microphone
        audio_chunk = next(self.microphone_stream)

        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Add to buffer
        self.buffer = np.append(self.buffer, audio_data)

        # Process when buffer is full
        if len(self.buffer) >= self.buffer_size:
            # Convert to float32 for noise reduction
            float_buffer = self.buffer.astype(np.float32) / 32768.0

            # Apply noise reduction
            # You can tweak these parameters to change the aggressiveness of the noise reduction
            reduced_noise = nr.reduce_noise(
                y=float_buffer,
                sr=self.sample_rate,
                prop_decrease=0.75,
                n_fft=1024
            )

            # Convert back to int16
            processed_chunk = (reduced_noise * 32768.0).astype(np.int16)

            # Clear buffer but keep a small overlap
            overlap = 1024
            self.buffer = self.buffer[-overlap:] if len(self.buffer) > overlap else np.array([])

            # Convert back to bytes
            return processed_chunk.tobytes() if not self.is_muted else b''

        # If buffer not full, return empty bytes
        return b''

    def mute(self):
        """Mute the microphone (produce silence)"""
        self.is_muted = True
        logger.info("Microphone muted")

    def unmute(self):
        """Unmute the microphone"""
        self.is_muted = False
        logger.info("Microphone unmuted")

    def toggle_mute(self):
        """Toggle between mute and unmute"""
        if self.is_muted:
            self.unmute()
        else:
            self.mute()
