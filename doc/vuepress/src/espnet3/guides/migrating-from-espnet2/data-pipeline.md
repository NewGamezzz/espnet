# Data pipeline

ESPnet3 recipe data flow is Python-based. Prefer dataset references and
`DataOrganizer` over recipe-local one-off loaders.

For new code, inspect:

- `espnet3/components/data/dataset_module.py`
- `espnet3/components/data/data_organizer.py`
- `espnet3/components/data/dataset_builder.py`

