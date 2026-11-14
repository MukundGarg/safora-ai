"""Opt-in offline microphone monitor. Optional sounddevice/PortAudio; no dispatch."""
import argparse
from dataclasses import asdict
import json
from math import isfinite
import time
from safora.audio.ingestion import RealtimeScreamMonitor, StreamConfig
from safora.audio.stream import Yamnet


def capture_timestamp(timing, now):
    """None means unavailable; let the monitor use its documented arrival estimate."""
    adc, current = timing.inputBufferAdcTime, timing.currentTime
    if not isfinite(adc) or not isfinite(current) or adc <= 0 or current < adc:
        return None
    stamp = now - (current - adc)
    return stamp if stamp >= 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config')
    parser.add_argument('--device', help='Local sounddevice input device name/index')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--opt-in-microphone', action='store_true', required=True)
    args = parser.parse_args()
    import sounddevice as sd
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    config = StreamConfig.load(args.config) if args.config else StreamConfig()
    if not 1 <= args.channels <= config.max_channels:
        parser.error('channels exceed configured bound')
    def status(value):
        print(json.dumps(value), flush=True)
    with Yamnet(args.model) as model:
        monitor = RealtimeScreamMonitor(model, lambda event: status(asdict(event)), status, config).start()
        def capture(indata, frames, timing, flags):
            # Clock conversion uses PortAudio's same-clock ADC/current-time difference.
            monitor.submit(indata, capture_time=capture_timestamp(timing, time.monotonic()),
                           discontinuity=bool(flags.input_overflow))
        try:
            with sd.InputStream(device=device, samplerate=config.source_rate, channels=args.channels,
                                dtype='int16', blocksize=config.max_block_samples, callback=capture):
                status({'kind': 'monitoring_only', 'source_rate': config.source_rate})
                while True:
                    time.sleep(1)
                    status(monitor.snapshot())
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop(timeout=None)  # Capture has stopped; never close an in-flight native interpreter.


if __name__ == '__main__':
    main()
