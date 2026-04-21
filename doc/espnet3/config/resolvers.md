---
title: ESPnet3 Config Resolvers
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Config Resolvers

## At a glance

| Resolver | Description |
| --- | --- |
| `load_line` | load lines from a text file into a list |
| `self_name` | use the current config file stem |

## `load_line`

Sample file:

```text
<blank>
<sos/eos>
<unk>
```

```yaml
token_list: ${load_line:conf/token_list.txt}
```

When Python reads this config, `token_list` becomes:

```python
["<blank>", "<sos/eos>", "<unk>"]
```

So `load_line` is just a way to write a string list in a text file instead of
inline YAML.

## `self_name`

`self_name` resolves to the current config file stem.

Example:

```yaml
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

If the file is `conf/tuning/training_e_branchformer.yaml`, then:

```yaml
exp_tag: training_e_branchformer
exp_dir: ./exp/training_e_branchformer
```

This is how TEMPLATE configs derive default names from the config filename.

## Common use cases

- keep large token lists out of YAML
- read one-item-per-line vocab or symbol files
- derive `exp_tag` from the config filename

## Notes

- use a path relative to the config file
- each line becomes one list item
- `self_name` uses the current config stem, not the full path
