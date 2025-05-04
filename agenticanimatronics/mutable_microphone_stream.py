from array import array
from typing import Optional

import pyaudio
import noisereduce as nr
import numpy as np
import assemblyai as aai


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
        print(f"connecting to default device {self._pyaudio.get_default_input_device_info()}")
        self.sample_rate = sample_rate
        self.is_muted = False
        self.threshold = threshold

        self._chunk_size = int(self.sample_rate * 0.1)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=device_index,
        )

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
                data = self._stream.read(self._chunk_size)
                data_chunk = array('h', data)
                vol = max(data_chunk)
                if (vol >= self.threshold):
                    return data
                else:
                    return b'\x00' * self._chunk_size
            else:
                _ = self._stream.read(self._chunk_size)
                return b'\x00' * self._chunk_size
        except KeyboardInterrupt:
            raise StopIteration

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

        if self._stream.is_active():
            self._stream.stop_stream()

        self._stream.close()
        self._pyaudio.terminate()

    def mute(self):
        """Mute the microphone (produce silence)"""
        self.is_muted = True
        print("Microphone muted")

    def unmute(self):
        """Unmute the microphone"""
        self.is_muted = False
        print("Microphone unmuted")

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

            # Calculate energy (RMS value) of the buffer
            energy = np.sqrt(np.mean(float_buffer ** 2))

            # # If energy is below threshold, return empty bytes
            # if energy < self.energy_threshold:
            #     # Clear buffer but keep a small overlap
            #     overlap = 1024
            #     self.buffer = self.buffer[-overlap:] if len(self.buffer) > overlap else np.array([])
            #     return b''

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
        print("Microphone muted")

    def unmute(self):
        """Unmute the microphone"""
        self.is_muted = False
        print("Microphone unmuted")

    def toggle_mute(self):
        """Toggle between mute and unmute"""
        if self.is_muted:
            self.unmute()
        else:
            self.mute()