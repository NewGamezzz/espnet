# Migrating from ESPnet2

ESPnet3 replaces numbered shell stages with named Python stages and splits
config by pipeline area.

<DocCards :cols="3">
  <DocCard
    title="Recipe structure"
    desc="Map the typical ESPnet2 recipe tree to the ESPnet3 recipe tree."
    icon="tabler:folder"
    href="./recipe-structure.md"
  />
  <DocCard
    title="Task to system"
    desc="See why ESPnet3 moved from task-centric ownership to system-centric workflows."
    icon="tabler:hierarchy-2"
    href="./task-to-system.md"
  />
  <DocCard
    title="Config diff"
    desc="Port training, inference, dataset, path, and parallel config surfaces."
    icon="tabler:settings-2"
    href="./config-diff.md"
  />
  <DocCard
    title="Data pipeline"
    desc="Map shell-based data preparation to builders, datasets, transforms, and collate."
    icon="tabler:database"
    href="./data-pipeline.md"
  />
  <DocCard
    title="Cluster and parallel"
    desc="Map `nj`, split SCPs, and shell jobs to provider and runner execution."
    icon="tabler:binary-tree-2"
    href="./cluster-and-parallel.md"
  />
</DocCards>
