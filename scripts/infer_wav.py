"""Bounded acoustic-only WAV runner; JSONL scores/events, never emergency dispatch."""
import argparse
from dataclasses import asdict, replace
import json
from safora.audio.ingestion import StreamConfig
from safora.audio.stream import Yamnet
from safora.evaluation.scream import analyze_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('wav')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--config')
    parser.add_argument('--pad-tail', action='store_true')
    args = parser.parse_args()
    config = StreamConfig.load(args.config) if args.config else StreamConfig()
    config = replace(config, threshold=args.threshold) if args.threshold is not None else config
    def output(observation):
        data = asdict(observation)
        data.update(time=observation.end, event=bool(observation.event), acoustic_event=data['event'], monitoring_only=True)
        print(json.dumps(data, allow_nan=False))
    with Yamnet(args.model) as model:
        analyze_wav(model, args.wav, config, pad_tail=args.pad_tail,
                    on_observation=output)


if __name__ == '__main__':
    main()
