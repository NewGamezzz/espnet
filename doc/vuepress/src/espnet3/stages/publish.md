# Publish stage

ESPnet3 publication is split into `pack_model` and `upload_model`.

## Run

```bash
python run.py --stages pack_model \
  --training_config conf/training.yaml \
  --publication_config conf/publication.yaml
python run.py --stages upload_model \
  --publication_config conf/publication.yaml
```

## Canonical implementation

- `espnet3.utils.publish_utils.pack_model`
- `espnet3.utils.publish_utils.upload_model`

## Notes

- use `inference_dir`, not `decode_dir`
- run `infer` and `measure` before packing when you want `metrics.json` in the bundle
- packed bundles are consumed by `InferenceModel`

