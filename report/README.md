# Report assembly

The main document is `report/main.tex`. Compile it with `report/` as the
working directory. Fragments are included with `\input{sections/data}` and
the bibliography with `\bibliography{refs}`. Figure paths in the fragments are
explicitly relative to this working directory, for example
`../figures/eda_sensor_trends.png`; no additional graphicspath is required.

The main document needs `graphicx` for figures. The settings and sensor-variation
figures use one IEEE column; wider multi-panel figures span both columns.
