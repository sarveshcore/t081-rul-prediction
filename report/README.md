# Report assembly

Place the main document at `report/main.tex` (issue #9) and compile with
`report/` as the working directory. Include fragments with
`\input{sections/data}` and use `\bibliography{refs}`. Figure paths in the
fragments are explicitly relative to this working directory, for example
`../figures/eda_sensor_trends.png`; no additional graphicspath is required.

The main document needs `graphicx` for figures. The settings and sensor-variation
figures use one IEEE column; wider multi-panel figures span both columns.
This documents the path convention for the report owner without creating or
replacing their main document.
