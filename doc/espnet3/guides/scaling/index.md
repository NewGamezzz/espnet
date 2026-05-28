---
title: Scaling
author:
  name: "Masao Someki"
date: 2026-05-28
---

# Scaling

Guides for running ESPnet3 at larger scale — across multiple nodes, over large
datasets, and through batch inference pipelines.

<DocCards>
  <DocCard
    title="Multi-node training"
    desc="Configure trainer.num_nodes, strategy, and launcher settings for multi-node distributed training."
    icon="tabler:topology-star"
    href="./multi-node.html"
  />
  <DocCard
    title="Large-scale data"
    desc="Shard large datasets, tune the data pipeline, and avoid bottlenecks when training on hundreds of hours."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
  <DocCard
    title="Inference at scale"
    desc="Run batch inference with the provider/runner layer across a cluster or local multi-GPU setup."
    icon="tabler:player-play"
    href="./inference.html"
  />
</DocCards>
