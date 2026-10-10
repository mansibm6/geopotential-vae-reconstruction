# Geopotential visualization and reconstruction

Visualizes geopotential grids and trains a convolutional variational autoencoder on historical fields.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [LSTM Notebook (Geopotential Data).ipynb](LSTM%20Notebook%20%28Geopotential%20Data%29.ipynb)
- [Visualization Notebook (Geopotential Data).ipynb](Visualization%20Notebook%20%28Geopotential%20Data%29.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `geopotential_500/*.nc`

## Outputs

Grid plots, an MP4 visualization, and an in-memory variational autoencoder.

## Implementation and limitations

The notebook name says LSTM, but its LSTM layer is commented out and training uses x as both input and target: this is reconstruction, not validated next-step forecasting. NetCDF files must expose z and time/lat/lon coordinates; training selects 2015–2016 and reserves 2017–2018. temperature_2m and toa_incident_solar_radiation paths are defined but not used by the current model. Supply them only if extending the workflow. Use a NetCDF backend plus dask for open_mfdataset.

Reusable model/analysis functions live in `experiment_lstm_notebook_geopotential_data.py`, `experiment_visualization_notebook_geopotential_data.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

## Original references

These links were retained from the original project notes; availability and licensing must be checked by the person obtaining the data.

- https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F5.625deg
