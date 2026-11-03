+++
hypothesis = "Binarize frame centres computed as i · step + duration / 2 differ in the last bit from pyannote's SlidingWindow[i].middle = 0.5 · (s + (s + duration)), s = i · step."
change = "Community1Timeline.annotate now computes frame centres the pyannote way, bit for bit (found while porting DiariZen, whose 20 ms frames put centres exactly on half milliseconds). The timeline also gained frame-geometry and local-speaker parameters whose defaults are Community-1's."
measurement = "Host simulation of both formulas over PyanNet geometry (270-sample step, 991-sample window), every frame of ten minutes plus 200,000 random segments of 1 to 3,000 frames; double precision as in Swift. Pinned by AudioWorkerTests community1FrameCentresFollowPyannote."
verdict = "kept"
numbers = { frames_differing_last_bit = 2911, frames_total = 35556, json_time_changes = 0, rttm_start_changes = 0, rttm_duration_changes_fraction = 0.0070 }
links = ["../../diarizen-large-v2/log/001-final-picks-1102.md"]
+++

# Frame centres, 2 Nov 2026

The JSON result (times rounded to milliseconds) is unchanged for every frame. RTTM lines print start and duration with three decimals; durations, as differences of two centres, move by 1 ms in about 0.7% of segments. No model output or decision changes. Community-1's default and heavy tests pass with the change.
