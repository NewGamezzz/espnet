# Stages

ESPnet3 recipes run named stages. The default ASR order in
`egs3/TEMPLATE/asr/run.py` is:

1. `create_dataset`
2. `train_tokenizer`
3. `collect_stats`
4. `train`
5. `infer`
6. `measure`
7. `pack_model`
8. `upload_model`
9. `pack_demo`
10. `upload_demo`

## Stage references

- [Create dataset](./create-dataset.md)
- [Train tokenizer](./train-tokenizer.md)
- [Collect stats](./collect-stats.md)
- [Train](./train.md)
- [Inference](./inference.md)
- [Measure](./measure.md)
- [Publish](./publish.md)
- [Demo](./demo.md)
- [System-specific](./system-specific.md)

