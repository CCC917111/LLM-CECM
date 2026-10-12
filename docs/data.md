# Input data

Run commands from the repository root unless a command explicitly starts with `cd src`.

## Files included in the repository

| Asset | Path | Use |
|---|---|---|
| Fixed 50-agent configuration | `src/true_digital_twin_agents.json` | Generator agents used by the simulation |
| Fuel operating ratios | `src/tools/fuel_dynamic_ratios.json` | Parameters used by the agent generator |
| NSW1 hourly RRP reference | `src/data_integration/real_aemo_data_config.json` | 72 price values and provenance metadata |
| Timestamped price table | `src/data_integration/nsw1_hourly_rrp_20240715_17.csv` | The same 72 values, with hour labels and interval counts |
| PPO policies | `src/models/multi_agent_high_price/` | Saved policies for the RL baseline |

The fixed agent JSON is already available. The Excel and SCADA sources below are needed when regenerating
an agent population. The IEEE 118-bus test network comes from the installed PYPOWER package (`case118`).

## Download the agent-generation sources

### July 2025 Generation Information

Download [the official July 2025 Excel workbook](https://www.aemo.com.au/-/media/files/electricity/nem/planning_and_forecasting/generation_information/2025/nem-generation-information-july-2025.xlsx?rev=bd6ca3641590414ea2b5c981cc6f4b04&sc_lang=en).
The [AEMO Generation Information page](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/forecasting-and-planning-data/generation-information)
also lists this edition under **Archive -> NEM July 2025 Generation Information**. Select the dated edition;
the newest edition on that page changes over time.

The existing generator expects the local filename `NEM Generation Information July 2025 (1).xlsx`.
Rename the downloaded workbook to that name and place it in `src/downloads/`. The `(1)` suffix is part
of the current script's filename, rather than an AEMO version identifier. The script reads the sheet
`ExistingGeneration&NewDevs`, with `header=1`. The original workbook checksum was not retained in this
release, so the linked archived edition has not been established as byte-identical to the original local copy.

### April 2022 DISPATCH_UNIT_SCADA

Download [PUBLIC_DVD_DISPATCH_UNIT_SCADA_202204010000.zip](https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2022/MMSDM_2022_04/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCH_UNIT_SCADA_202204010000.zip)
from the [official April 2022 archive](https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2022/MMSDM_2022_04/MMSDM_Historical_Data_SQLLoader/DATA/).
Extract `PUBLIC_DVD_DISPATCH_UNIT_SCADA_202204010000.CSV` into `src/downloads/`.

The resulting layout is:

```text
src/
├── downloads/
│   ├── NEM Generation Information July 2025 (1).xlsx
│   └── PUBLIC_DVD_DISPATCH_UNIT_SCADA_202204010000.CSV
├── tools/fuel_dynamic_ratios.json
├── true_digital_twin_agents.json
└── data_integration/
    ├── real_aemo_data_config.json
    └── nsw1_hourly_rrp_20240715_17.csv
```

Regeneration requires the project's dependencies and an Excel reader (`pip install openpyxl`). From `src/`,
`python generate_true_digital_twin_agents.py` reads these three inputs and writes `true_digital_twin_agents.json`
in the current directory, replacing any existing file. Use a separate working copy for regeneration.
The generator samples units and costs randomly, so retain the bundled JSON for the fixed released population.

## The 72-hour NSW1 price reference

The JSON and CSV contain 72 hourly bins labelled from **15 July 2024 00:00** through **17 July 2024 23:00**
in **AEMO market time (AEST, UTC+10)**. Prices are **AUD/MWh**.

The source is `PUBLIC_DVD_DISPATCHPRICE_202407010000.CSV` from the
[official July 2024 MMSDM archive](https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2024/MMSDM_2024_07/).
Its SHA-256 is:

```text
1415bb9cc864c0c59c9322bdef31f55510a4034fc67fb949a86e28260595c087
```

To retain the existing reference values, observations are selected with
`2024-07-15 00:00 < SETTLEMENTDATE < 2024-07-18 00:00`, filtered to `REGIONID=NSW1`,
and grouped by the calendar hour of their recorded timestamp. Each bin is an arithmetic mean.
The first bin therefore uses **11 observations (00:05 through 00:55)**, while the remaining 71 bins use
12 each. Including the source observation at exactly 00:00 would change the first mean from
82.69886090909091 to 82.20062416666667 AUD/MWh. This publication preserves the existing 72 values and
records their exact boundary; it does not replace them with a different aggregation.

The CSV columns are `round_index` (0-71), `hour_start_aest` (ISO timestamp), `rrp_aud_per_mwh` and
`n_observations`. This index describes the ordering of the reference series, rather than a completed run.

### Verify the supplied series offline

Put the original monthly price CSV outside the repository, then run:

```bash
python3 scripts/verify_aemo_reference.py --source /path/to/PUBLIC_DVD_DISPATCHPRICE_202407010000.CSV
```

This standard-library-only command checks the source checksum, all hourly values, boundaries, interval counts,
and JSON/CSV agreement. It performs no download, model call or simulation. The source and reference files
are read-only. The published series was verified against the source file with the checksum above.

`src/data_integration/nemosis_fetcher.py` is the existing optional download helper. Its default configuration
generator retains 10 hours, so use the supplied files and verification command for this 72-hour release.

## How the current experiments consume prices

Experiment 1 runs 10 rounds and uses the first 10 values of the bundled JSON. Experiment 2 and the existing
30-round experiment 3 define their own scenario arrays in `src/experiments/experiment_configs.py`.
The published 72-hour reference is available for an explicitly configured evaluation; publishing it does
not switch these scenarios to 72 rounds, change operational assumptions, or reproduce the paper's results.

Gas-price, outage, demand and carbon-price scenario arrays are simulation settings. The NSW1 RRP series
is an observed reference used for comparison; it does not force the simulated clearing price.
