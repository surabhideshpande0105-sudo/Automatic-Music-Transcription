# Profile before optimizing: pyin was 79%

20s clip took 20s; timing each stage showed pyin 15.7s, HPSS 1.6s, load 2.5s. Fixes: pyin hop 512→2048 with neighbor-widened slice mapping, soundfile fast load, lighter HPSS margin. Now 8.5s with identical synth-test accuracy. App shows a live progress bar since waiting feels shorter than silence.
