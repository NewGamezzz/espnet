# System-specific stages

Some stages exist only on specific systems. The current ASR example is
`train_tokenizer`.

Keep stage-specific behavior in the system class, but keep all stage settings in
YAML config rather than extra CLI arguments.

