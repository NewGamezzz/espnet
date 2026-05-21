# Freeze layers

Freezing belongs to model and optimizer configuration, not to stage wiring.
Apply it where parameter groups are built or filtered.

Use this when:

- the target dataset is small
- you only want to adapt the head
- you want a faster sanity pass before full tuning

