# Why ESPnet3

ESPnet3 is designed for Python-first recipe development. The main goals are
clear stage boundaries, config-driven execution, and code reuse across recipe
and publication flows.

## What ESPnet3 is good at

- Named stages instead of shell stage numbers
- YAML-driven training, inference, metrics, publication, and demo flows
- Reusing Python dataset and component code across recipes
- Packaging trained models into a runtime bundle

## What it is not trying to be

- A minimal pure-library wrapper with no recipe structure
- A shell-first workflow like classic ESPnet2 recipes
- A generic replacement for every trainer framework

## When to stay on ESPnet2

Stay on ESPnet2 when your workflow depends on existing shell recipes or you do
not want to migrate recipe layout yet.

