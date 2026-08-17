# VGG16 Similarity Diagnostics

Long-running diagnostics in this folder should save intermediate progress to
disk. The runner writes start/finish/failure events under `progress_events/`
and writes completed rank/fold outputs under `partial_results/` before building
final CSV summaries and plots.

This is intentional: high-rank VGG16 checks can take long enough that progress
must be inspectable while a run is still active.
