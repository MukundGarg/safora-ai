"""Fixed-config offline evaluation; never fits weights or sweeps a reused benchmark.

Run python -m safora.evaluation.scream --help. WAV input is streamed PCM16.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import wave
import numpy as np
from safora.audio.ingestion import ScreamProcessor, StreamConfig
from safora.audio.stream import Yamnet
from .metrics import event_metrics, prf


def validate_manifest(rows, base=Path('.')):
    base = Path(base)
    seen, ids = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Manifest rows must be objects')
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('Recording IDs must be unique nonempty strings')
        ids.add(row['id'])
        if row.get('split') not in {'train', 'validation', 'test', 'reused_external'}:
            raise ValueError('Explicit train/validation/test/reused_external split required')
        if not isinstance(row.get('source_id'), str) or not row['source_id']:
            raise ValueError('source_id is required for isolation; do not infer it from labels')
        if not isinstance(row.get('path'), str) or not row['path']:
            raise ValueError('WAV path required')
        keys = [('path', str((base / row['path']).resolve()))]
        for field in ('source_id', 'group_id', 'session_id', 'recording_sha256'):
            value = row.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError('Group/provenance IDs must be nonempty strings')
                keys.append((field, value))
        for key in keys:
            if key in seen and seen[key] != row['split']:
                raise ValueError(f'Source/group/path split leakage: {key}')
            seen[key] = row['split']
        label = row.get('label')
        if label is not None and not isinstance(label, bool):
            raise ValueError('label must be true, false, null or omitted')
        annotations = row.get('events')
        if annotations is not None:
            if not isinstance(annotations, list):
                raise ValueError('events must be a list or null')
            previous = 0.
            for interval in annotations:
                if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                        or any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in interval)
                        or not np.isfinite(interval).all()
                        or not 0 <= interval[0] < interval[1] or interval[0] < previous):
                    raise ValueError('Events must be ordered, finite, non-overlapping [onset,end] intervals')
                previous = interval[1]
            if label is True and not annotations:
                raise ValueError('Positive label needs timed events or events:null for unknown timing')
            if label is False and annotations:
                raise ValueError('Negative label conflicts with event annotations')
    return rows


def analyze_wav(model, path, config=None, *, pad_tail=False, on_observation=None):
    config = config or StreamConfig()
    events = []
    latency_count, latency_total, latency_max = 0, 0., None
    windows = padded_windows = 0
    peak = None
    covered_end = 0.
    with wave.open(str(path), 'rb') as recording:
        if recording.getsampwidth() != 2 or recording.getcomptype() != 'NONE':
            raise ValueError('Only uncompressed PCM16 WAV is supported')
        channels, rate = recording.getnchannels(), recording.getframerate()
        if not 1 <= channels <= config.max_channels:
            raise ValueError('WAV channels exceed configured bound')
        duration = recording.getnframes() / rate
        effective = replace(config, source_rate=rate)
        processor = ScreamProcessor(model, effective)
        def consume(observations):
            nonlocal windows, padded_windows, peak, covered_end, latency_count, latency_total, latency_max
            for observation in observations:
                windows += 1
                padded_windows += observation.padded
                peak = max(peak if peak is not None else 0., observation.scores['screaming'])
                covered_end = max(covered_end, min(duration, observation.end))
                latency_count += 1
                latency_total += observation.inference_seconds
                latency_max = max(latency_max or 0., observation.inference_seconds)
                if observation.event:
                    events.append(asdict(observation.event))
                if on_observation:
                    on_observation(observation)
        while True:
            frames = recording.readframes(effective.max_block_samples)
            if not frames:
                break
            samples = np.frombuffer(frames, dtype='<i2').reshape(-1, channels)
            consume(processor.feed(samples))
        consume(processor.finish(pad_tail=pad_tail))
    return dict(duration_seconds=duration, source_rate=rate, channels=channels,
                window_samples=model.window_samples, window_seconds=model.window_samples / 16000,
                hop_samples=config.hop_samples, hop_seconds=config.hop_samples / 16000,
                resampler_delay_seconds=processor.resampler.delay_seconds,
                windows=windows, padded_windows=padded_windows,
                analyzed_audio_seconds=covered_end, unscored_audio_seconds=max(0., duration - covered_end),
                status='analyzed' if windows else 'insufficient_audio',
                peak_screaming_score=peak, single_hit_detected=peak is not None and peak >= config.threshold,
                detected=bool(events), events=events,
                inference_latency_seconds=dict(count=latency_count,
                    mean=latency_total / latency_count if latency_count else None,
                    max=latency_max))


def evaluate(rows, model, config=None, *, split, base=Path('.'), pad_tail=False, event_tolerance=1.):
    config = config or StreamConfig()
    base = Path(base)
    if not np.isfinite(event_tolerance) or event_tolerance < 0:
        raise ValueError('Event tolerance must be finite/nonnegative')
    validate_manifest(rows, base)  # Check the ENTIRE manifest before selecting a split.
    if split not in {'train', 'validation', 'test', 'reused_external'}:
        raise ValueError('Unknown selected split')
    clips, errors, unknown = [], [], 0
    tp = fp = fn = tn = event_tp = event_fp = event_fn = 0
    negative_seconds = analyzed_negative_seconds = 0.
    negative_events = 0
    annotated_recordings = 0
    delays, misses = [], []
    selected = [row for row in rows if row['split'] == split]
    for row in selected:
        try:
            clip = analyze_wav(model, base / row['path'], config, pad_tail=pad_tail)
            truth = row.get('events')
            if truth is not None and any(end > clip['duration_seconds'] for _, end in truth):
                raise ValueError('Annotation extends beyond recording')
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            errors.append({'id': row['id'], 'error': str(exc)})
            continue
        label = row.get('label')
        clip.update(id=row['id'], source_id=row['source_id'], group_id=row.get('group_id'), label=label)
        if label is None:
            unknown += 1
        elif label:
            tp += clip['detected']
            fn += not clip['detected']
            if not clip['detected']:
                misses.append({'id': row['id'], 'status': clip['status'], 'peak_score': clip['peak_screaming_score']})
        else:
            fp += clip['detected']
            tn += not clip['detected']
            negative_seconds += clip['duration_seconds']
            analyzed_negative_seconds += clip['analyzed_audio_seconds']
            negative_events += len(clip['events'])
        if truth is not None:
            annotated_recordings += 1
            metrics = event_metrics([event['time'] for event in clip['events']], truth, 0, event_tolerance)
            event_tp += metrics['tp']; event_fp += metrics['fp']; event_fn += metrics['fn']
            delays.extend(metrics['delays_seconds'])
            clip['event_evaluation'] = metrics
        clips.append(clip)
    observed = analyzed_negative_seconds > 0
    clip_scores = prf(tp, fp, fn)
    event_scores = prf(event_tp, event_fp, event_fn)
    if tp + fp + fn + tn == 0:
        clip_scores.update(precision=None, recall=None, f1=None)
    if annotated_recordings == 0:
        event_scores.update(precision=None, recall=None, f1=None)
    return dict(version=1, split=split,
        benchmark_status='reused_not_independent' if split == 'reused_external' or any(
            r.get('reused', False) or r.get('benchmark_status') in {'reused', 'reused_not_independent'} for r in selected) else 'independence_not_verified',
        config=asdict(config), tail_policy='single_eof_zero_pad' if pad_tail else 'drop_incomplete',
        event_tolerance_seconds=event_tolerance, selected_recordings=len(selected), evaluated_recordings=len(clips),
        unknown_clip_labels=unknown, complete=not errors, errors=errors,
        clip_metrics={**clip_scores, 'tn': tn, 'confusion_matrix': [[tn, fp], [fn, tp]]},
        event_metrics={**event_scores, 'annotated_recordings': annotated_recordings,
                       'detection_delay_seconds': delays},
        false_alarms=dict(events=negative_events, negative_audio_hours=negative_seconds / 3600,
            analyzed_negative_audio_hours=analyzed_negative_seconds / 3600,
            unscored_negative_seconds=negative_seconds - analyzed_negative_seconds,
            per_negative_audio_hour=negative_events * 3600 / negative_seconds if observed else None,
            per_analyzed_negative_hour=negative_events * 3600 / analyzed_negative_seconds if observed else None),
        false_negative_clips=misses, clips=clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test', 'reused_external'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pad-tail', action='store_true', help='Explicit offline EOF experiment; never used live')
    parser.add_argument('--event-tolerance', type=float, default=1.)
    args = parser.parse_args()
    text = args.manifest.read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    config = StreamConfig.load(args.config) if args.config else StreamConfig()
    with Yamnet(args.model) as model:
        result = evaluate(rows, model, config, split=args.split, base=args.manifest.parent,
                          pad_tail=args.pad_tail, event_tolerance=args.event_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
