# Wearable / Android boundary — hardware validation BLOCKED

Repository audit found no wearable firmware, BLE/Bluetooth transport, pairing/key
provisioning, ESP32 pinout, IMU/sensor definitions or button command protocol.
The app actually provides `DecisionEngine.manual(now)` and an Activity manual SOS
control. `AudioPipeline` / `MlController` are audio-only; no physical wearable
receiver is present. No hardware or authorized physical Android phone is attached.
Do not infer a Bluetooth stack, board, gesture or sensor capability from the
product description.

## What is preserved and implemented

The app's manual SOS calls the existing engine independently of ML, microphone,
local model loading and resource mode. Existing Kotlin tests exercise manual
priority without monitoring and after lower-level cues. Host coordinator manual
hooks remain available on opt-out/permission denial and while retiring native work.
This is software-level evidence only, not a physical-button/process-death promise.
No contact/location/SMS/police dispatch was added or modified. Existing incident
manager policy and any consent/cancel/location/contact handling remain with the
Kotlin app owner; the demo only presents local generic notifications.

New `MlState.coverage()` exposes AVAILABLE/DEGRADED/UNAVAILABLE per acoustic,
speech and conversation feature. AVAILABLE means model execution readiness only,
**not validated protection**. Unknown language/disabled/failed/unloaded ASR is
UNAVAILABLE. Session queue drops degrade execution coverage; current untrained
conversation rules stay DEGRADED whenever ASR is usable. No feature reports safety
validation. App owner should marshal fixed coverage reasons to its own UI, never
retain raw transcripts. Storage preflight failures have a fixed speech-storage
coverage code; acoustic/manual paths remain independent.

## Required hardware contract before implementing a transport

Hardware/firmware owner must provide and review:

1. Actual board/firmware revision, physical intentional-button semantics, transport
   and pairing roles, packet/MTU/rate limits, Android supported versions and power
   behavior. Document what works if phone/app/service/network/location is absent.
2. A versioned authenticated command envelope bound to the paired device and user,
   actual key provisioning/rotation/revocation and secure storage on both ends.
   Encryption or a device name alone is not authentication. Define canonical bytes
   and approved test vectors before implementing a parser or cryptographic binding.
3. Stable event IDs, monotonic ordering/counters across reboot, durable replay
   protection, duplicate/retry/ack semantics, reconnect behavior and bounded queues.
   Agree atomic persistence before dispatch acknowledgement; never mark a repeated
   authenticated button event as a new intentional SOS.
4. Hardware/software debounce and long/short/hold semantics, validated against the
   real switch and user interaction. No guessed debounce time is approved here.
5. Authenticated battery/connection state, stale-state timeout and user-facing
   degraded/unavailable behavior, firmware integrity/update path and recovery.
6. Explicit distinction between a deliberate SOS button command and uncertain
   fall/motion/audio/gesture inference. Sensor inference needs its own annotation,
   independent dataset, hazard review and gates; no such sensors are assumed.

No speculative unauthenticated Bluetooth receiver, ad-hoc firmware or invented wire
protocol is introduced. Authentication/replay/ACK/reconnect reliability is **not
implemented or verified for actual hardware** because the interoperable interface
and hardware are absent. A generic HMAC example without device/key/flash/reboot
semantics would not satisfy these requirements.

## Android integration and fault validation

One existing capture owner feeds mono16kHz PCM16 blocks≤1600samples into
MlController. Native YAMNet/ASR workers and buffers are independent. Review
permission/background/foreground-service/lock/call/low-battery/process-death
behavior using the real application and OEM/API devices. Stop capture before
teardown; join native owners off the UI thread, reject stale session/generation
callbacks and preserve independent manual UI availability. Neither acoustic nor
conversation scores may call the manual path or bypass user-confirmed check-in.

Run the existing DeviceSmokeTest plus MlCoreSmokeTest on authorized devices;
compilation is not runtime validation. Add physical-button tests against reviewed
firmware: bounce, duplicates, retransmission, reconnect, replay/reorder, altered
payload/MAC, stale battery, counter wrap/reboot, full phone storage, app restart,
permission loss, locked phone and ML load/inference failure. Measure button-to-app
ack and user-confirmed incident flow delay, missed/duplicated activations and power
behavior. Product acceptance numbers must come from the joint hazard review, not
invented percentages. The hardware lead owns device/transport evidence; Kotlin
owner owns Android/incident policy; Audio ML owner owns only audio inference and
truthful execution coverage. Safety lead/product owner must approve all thresholds
and remaining residual risks before release.
