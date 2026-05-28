# Stages

ESPnet3 recipes run named stages.

## Stage reference

<DocCards>
  <DocCard
    title="create_dataset"
    desc="Download or build datasets for your recipe."
    icon="tabler:database"
    href="./create-dataset.html"
  />
  <DocCard
    title="collect_stats"
    desc="Compute feature shapes and global statistics."
    icon="tabler:chart-bar"
    href="./collect-stats.html"
  />
  <DocCard
    title="train"
    desc="Run Lightning training with training.yaml."
    icon="tabler:school"
    href="./train.html"
  />
  <DocCard
    title="infer"
    desc="Write hypothesis outputs under inference_dir."
    icon="tabler:bolt"
    href="./inference.html"
  />
  <DocCard
    title="measure"
    desc="Compute metrics (WER, MOS, SI-SDR, …) from inference outputs."
    icon="tabler:ruler-measure"
    href="./metrics.html"
  />
  <DocCard
    title="pack_model / upload_model"
    desc="Bundle and publish a trained model to HuggingFace Hub."
    icon="tabler:package-export"
    href="./publish.html"
  />
  <DocCard
    title="pack_demo / upload_demo"
    desc="Generate and upload a Gradio demo UI."
    icon="tabler:device-desktop"
    href="./demo.html"
  />
</DocCards>

## Related pages

<DocCards>
  <DocCard
    title="System and stages"
    desc="How run.py, BaseSystem, and named stages fit together."
    icon="tabler:puzzle"
    href="../core/system-and-stages.html"
  />
  <DocCard
    title="Adding a stage"
    desc="See the contributor guide for extending a System class and wiring stages into run.py."
    icon="tabler:tool"
    href="../contributing/adding-a-stage.html"
  />
</DocCards>
