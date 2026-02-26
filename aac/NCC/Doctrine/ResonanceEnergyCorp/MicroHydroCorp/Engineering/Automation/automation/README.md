# Automation

- `import_measurements.py` — appends CSV run data to the canonical workbook.
- `Import_Config.json` — allowed Test_IDs and sensor mapping hints.

## Usage
1. Copy CSVs into a run folder under `tests/raw/`.
2. Run:
   ```bash
   python import_measurements.py --workbook path/to/MicroHydroV1_RnD_Export.xlsx --csv tests/raw/<run>/*.csv
   ```

NOTE: This is a skeleton; replace with your validated importer.

