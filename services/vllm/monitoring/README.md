# Setup Prometheus & Grafana

`vLLM` automatically generates inference metrics which can be picked up by `prometheus` time series database. Then `grafana` can be used to pull data from `prometheus` into web dashboard visualizations.

1. Setup Prometheus.
    * Enable docker container to read the `prometheus.yml` config file `chmod 664 prometheus.yml`.
    * Launch `prometheus` container: `docker compose up prometheus -d`. The docker compose config will pull in the `prometheus.yml` file defined in this directory.
    * NOTE: The `prometheus.yml` file configures specific ports for discovery. It expects the `vLLM` services providing the metric data to be present on ports `8000`, `8100`, `8101`, `8102` or `8103`
    * Navigating to `${SERVER_URL}:9090/` will bring you to the `prometheus` database dashboard where you can query individual metric variables.
2. Setup Grafana.
    * Launch `grafana` container: `docker compose up grafana -d`
    * Navigate to `${SERVER_URL}:3000/` where `${SERVER_URL}` is `localhost` if on current server or if using port forwarding from a remote server. This gets you to the Grafana UI interface.
    * In Grafana, add connection (Home > Connections > Data sources). Select "prometheus" as a data source. For Prometheus server URL, enter `http://prometheus:9090`. Then click "Save & test".
    * In Grafana, create a dashboard (Home > Dashboards > Import dashboard) and use the Imprt option. Upload the `grafana.json` configuration or copy and paste the JSON model. Then click "Load"

Now you should have a `grafana` dashboard visualization of the time-series metric data flowing into `prometheus`.
