---
home: true
icon: /assets/image/espnet.png
title: ESPnet3
heroImage: /assets/image/espnet_logo1.png
heroImageStyle:
  - height: auto
  - width: 400px

heroText: "ESPnet3: Python-first speech workflows"
tagline: "Provider/Runner architecture, Hydra configs, and Lightning-based training from laptops to clusters."

highlights:
  - header: Start with ESPnet3
    bgImage: https://theme-hope-assets.vuejs.press/bg/3-light.svg
    features:
      - title: "Docs hub"
        details: "Where to begin and how pieces fit together."
        icon: fa-solid:book
        link: ./README.md
      - title: "Provider/Runner overview"
        details: "Understand the execution model for data prep, decode, and scoring."
        icon: fa-solid:diagram-project
        link: ./provider_runner.md
      - title: "Hydra configs"
        details: "Modular YAML for model, trainer, dataloader, and parallel backends."
        icon: fa-solid:sliders
        link: ./parallel.md

  - header: Build and train
    bgImage: https://theme-hope-assets.vuejs.press/bg/2-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "Data pipeline"
        link: ./dataset.md
        details: "DataOrganizer, transforms, preprocessors, and sharding."
        icon: material-symbols:database-outline
      - title: "System entry point"
        link: ./system.md
        details: "Stages orchestrated by BaseSystem and task-specific systems."
        icon: fa-solid:diagram-project
      - title: "Recipe layout"
        link: ./recipe_directory.md
        details: "How egs3 directories, configs, and stages are organized."
        icon: fa-solid:folder-tree
      - title: "Data preparation"
        link: ./data_preparation.md
        details: "Provider/Runner patterns for feature extraction and cleaning."
        icon: fa-solid:hammer
      - title: "Callbacks"
        link: ./callbacks.md
        details: "Default Lightning callbacks and how to customize them."
        icon: fa-solid:bell
      - title: "Optimizers & schedulers"
        link: ./optimizer_configuration.md
        details: "Single vs multi-optimizer setups enforced by ESPnet3."
        icon: fa-solid:timeline
      - title: "Multi-GPU / multi-node"
        link: ./multiple_gpu.md
        details: "Configure Lightning for distributed runs."
        icon: fa-solid:network-wired

  - header: Inference and evaluation
    bgImage: https://theme-hope-assets.vuejs.press/bg/4-light.svg
    bgImageStyle:
      background-repeat: repeat
      background-size: initial
    features:
      - title: "Inference runners"
        link: ./provider_runner.md
        details: "Scale decode jobs locally or on clusters."
        icon: fa-solid:bolt
      - title: "Evaluation pipeline"
        link: ./evaluate.md
        details: "InferenceRunner + ScoreRunner with YAML-driven metrics."
        icon: fa-solid:check-double

footer: Apache License 2.0, Copyright © 2024-present ESPnet community
---

## How to cite ESPnet
```
@inproceedings{watanabe18_interspeech,
  title     = {ESPnet: End-to-End Speech Processing Toolkit},
  author    = {Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  year      = {2018},
  booktitle = {Proc. Interspeech},
  pages     = {2207--2211},
  doi       = {10.21437/Interspeech.2018-1456},
  issn      = {2958-1796},
}
```
To cite individual modules, models, or recipes, please refer to [Additional Citations](./citations.md).
