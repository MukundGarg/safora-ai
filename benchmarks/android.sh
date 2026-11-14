#!/usr/bin/env bash
set -euo pipefail
# ADB must be connected to a consented physical device. No audio/transcripts collected.
adb shell dumpsys meminfo org.safora.demo
adb shell dumpsys cpuinfo
adb shell dumpsys thermalservice
adb shell dumpsys batterystats org.safora.demo
