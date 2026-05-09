# 📊 PhDPlot — Scientific Data Visualization Tool

> Built for PhD research at **New Mansoura University, Faculty of Engineering**

A fully interactive data visualization tool for plotting, curve fitting, and exporting
publication-quality figures from CSV and Excel results files.

---

## ✨ Features

| Feature | Details |
|---|---|
| **File Support** | CSV, Excel (.xlsx / .xls), tab-separated TXT |
| **Multiple Datasets** | Load and switch between multiple files per session |
| **Multiple Plots** | Independent plot tabs, each with its own configuration |
| **Chart Types** | Line, Scatter, Bar, Area |
| **Curve Fitting** | Linear, Quadratic, Exponential, Power Law — with R² |
| **Error Bars** | Select any ±std column per series |
| **Dual Y-Axis** | Overlay two quantities with different units |
| **Export** | High-res PNG (2×), vector SVG, interactive HTML, filtered CSV |
| **Axis Control** | Custom labels, manual ranges, grid, smooth curves |
| **Multi-Series** | Color picker per series, legend control |

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

```bash
# Clone or create your repo
git init
git add .
git commit -m "Initial commit: PhDPlot"

# Push to GitHub (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/phd-plotter.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository and branch (`main`)
5. Set **Main file path** to `app.py`
6. Click **"Deploy!"**

Your app will be live at:
```
https://YOUR_USERNAME-phd-plotter-app-XXXX.streamlit.app
```

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/phd-plotter.git
cd phd-plotter

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
phd-plotter/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .streamlit/
│   └── config.toml         ← Theme & server settings
└── .gitignore
```

---

## 📊 Supported File Formats

Your CSV/Excel files should have:
- **Row 1** = column headers (e.g. `Time`, `Temperature`, `COP`, `std_COP`)
- **Rows 2+** = numeric data

**Example CSV:**
```
Time_h,Energy_kWh,COP,std_COP,Fouling_m2K_W
0,0.0,4.2,0.12,0.0
100,450.3,3.9,0.15,0.00012
200,910.1,3.5,0.18,0.00025
...
```

---

## 🔧 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web UI framework |
| plotly | ≥ 5.18 | Interactive charts |
| pandas | ≥ 2.0 | Data loading & processing |
| numpy | ≥ 1.24 | Numerical operations |
| scipy | ≥ 1.11 | Curve fitting (exponential, power law) |
| openpyxl | ≥ 3.1 | Excel file reading |
| kaleido | == 0.2.1 | Static PNG / SVG export |

---

## 📝 License

MIT — Free to use and modify for research and academic purposes.

---

*PhDPlot · New Mansoura University · Faculty of Engineering*
