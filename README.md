# 🛰️ WAVE NISAR Monitor

Web platform prototype for analyzing SAR data from the NISAR satellite (**GCOV** product) over Honduras.
It lets you draw an area on the map, choose one or two available dates, and view **water/wetland** and **forest** maps along with their reliability level.

- **Intuitive mode** (default): color maps, bar charts, and a summary in plain language.
- **Technical mode**: raw radiometric layers (HH / HV), scenes from all available NISAR products, and three reference sections (products, inventory, and glossary) that can be hidden by clicking their title bar.

> The interface the user sees is in English; the code and comments are in Spanish.

## Requirements

- Python 3.9 to 3.12
- Free NASA Earthdata token: https://urs.earthdata.nasa.gov

## Installation (Windows)

1. Run `setup.bat` (creates the `venv` virtual environment and installs the dependencies).
2. Copy `.env.example` to `.env` and paste your token into `EDL_TOKEN`.
3. Run `run.bat` and open http://localhost:5000

Without `.bat`: `python -m venv venv`, activate the environment, `pip install -r requirements.txt`, and `python app.py`.

## Tests

```
python -m unittest
```

`test_parameters.py` does not need internet; `test_gcov_analysis.py` simulates the ASF queries and the HDF5 files.

## Structure

| File | Purpose |
|---|---|
| `app.py` | Flask server: API routes and cache |
| `gcov_analysis.py` | ASF search, GCOV file reading, and result building |
| `parameters.py` | Water/forest thresholds and classification (numpy only) |
| `templates/index.html` | Complete interface (HTML, CSS, and JavaScript) |

## Scientific notice

The thresholds in `parameters.py` are **initial values and are not calibrated for Honduras**. A change in radar signal does not by itself prove deforestation, flooding, or any other phenomenon; the results must be validated with field data.

## Security

The `.env` file contains your personal token and is in `.gitignore`: **never upload it to GitHub**.
