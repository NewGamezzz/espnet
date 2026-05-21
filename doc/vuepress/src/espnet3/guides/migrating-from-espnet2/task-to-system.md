# Task to system

In ESPnet3, the stage owner is a `System` class. For ASR, that is
`espnet3.systems.asr.system.ASRSystem`.

Use this mapping:

- ESPnet2 task-centric entrypoints map to `System` methods
- inference maps to `infer`
- scoring maps to `measure`
- publication maps to `pack_model` and `upload_model`

