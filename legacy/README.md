# Legacy implementation

`qwen_lpu_model.py` is the original single-file prototype. It is intentionally preserved rather than silently rewritten so that behavior can be compared during the migration.

Known issues include import-time environment-variable backend selection, permissive checkpoint loading, duplicated/fragile RoPE implementations, and a decoder that does not reverse the legacy signed-add packing borrow. New development should use `src/qwen_int4/`; fixes should be validated against tests before the legacy file is retired.

