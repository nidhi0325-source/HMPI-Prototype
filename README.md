## HMPI: Heavy Metal Pollution Index Dashboard

An interactive Streamlit app to analyze water quality using Heavy Metal Pollution Index (HPI) and Heavy Metal Evaluation Index (HEI). Includes light/dark mode, charts, correlation heatmap, map view, and PDF report export.

### Features
- **Theme toggle**: Light/Dark mode
- **Flexible data input**: Manual entry, CSV/Excel upload, sample data
- **Calculations**: HPI, HEI, and risk levels
- **Visualizations**:
  - Bar and line charts with safe limit lines
  - Pie chart per site
  - Correlation heatmap
  - Mapbox scatter map by site with HPI-sized markers
- **PDF export**: Key metrics and embedded charts

### Safe Limits (defaults)
- Pb: 0.06, Hg: 0.02, Cd: 0.02, As: 0.005, Cr: 0.025

### Requirements
See `requirements.txt`.

```txt
streamlit>=1.33
pandas>=2.0
numpy>=1.24
plotly>=5.20
kaleido>=0.2.1
seaborn>=0.13
matplotlib>=3.8
fpdf2>=2.7
openpyxl>=3.1
```

### Quick Start (Windows, PowerShell)
1. Put `hmpi_app.py` and `requirements.txt` in the same folder.
2. (Optional) Virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate
   ```
3. Install:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   streamlit run hmpi_app.py
   ```
   - Open `http://localhost:8501`
   - If port busy: `streamlit run hmpi_app.py --server.port 8502`

### Data Input
- **Manual Entry**: Provide comma-separated site names and metal concentrations for Pb, Hg, Cd, As, Cr.
- **Upload CSV/Excel**: Must include `Site` and any subset of metals; missing metals are filled as 0. Include `Latitude`/`Longitude` or they are auto-generated.
- **Sample Data**: Built-in example for quick demo.

### PDF Export
- Click “Export PDF Report” to generate a multi-page PDF with key metrics and charts.
- Requires `plotly[kaleido]` and `fpdf2` (already in `requirements.txt`).

### Troubleshooting
- PDF export error:
  - Ensure Kaleido is installed: `pip install "plotly[kaleido]" fpdf2`
  - Restart Streamlit after installing new packages.
- Excel upload error:
  - Ensure `openpyxl` is installed (in `requirements.txt`).
- Dark/Light mode not switching:
  - The app uses a sidebar toggle; ensure you’re on Streamlit >= 1.25 (fallback to checkbox is included).

### License
MIT (add your license if different).

### Acknowledgments
- HPI and HEI metrics based on standard environmental assessment methods.
