import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")

def generate_visual_charts(df, rfm_summary, retention_matrix, affinity_matrix):
    """
    Generates high-resolution analytics figures.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "figure.dpi": 300})

    # 1. Monthly Revenue & Profit Trend
    monthly = df.groupby(df["order_date"].dt.to_period("M")).agg(
        Net_Revenue=("net_revenue", "sum"),
        Gross_Profit=("gross_profit", "sum")
    ).reset_index()
    monthly["order_month_str"] = monthly["order_date"].astype(str)

    plt.figure(figsize=(12, 5.5))
    plt.plot(monthly["order_month_str"], monthly["Net_Revenue"] / 1000, marker="o", color="#2563eb", lw=2.5, label="Net Revenue ($k)")
    plt.plot(monthly["order_month_str"], monthly["Gross_Profit"] / 1000, marker="s", color="#10b981", lw=2.2, label="Gross Profit ($k)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title("Monthly Revenue & Gross Profit Trajectory (24-Month Trend)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Amount ($ in Thousands)", fontsize=11, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart1 = os.path.join(CHARTS_DIR, "monthly_sales_trend.png")
    plt.savefig(chart1, dpi=300)
    plt.close()
    print(f"[OK] Saved Monthly Sales Trend to: {chart1}")

    # 2. Cohort Retention Heatmap
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        retention_matrix.iloc[:12, :12],
        annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
        cbar_kws={"label": "Retention Rate (%)"}
    )
    plt.title("Monthly Customer Retention Cohort Analysis (%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Months Since Acquisition (Cohort Index)", fontsize=11, fontweight="bold")
    plt.ylabel("Acquisition Cohort", fontsize=11, fontweight="bold")
    plt.tight_layout()
    chart2 = os.path.join(CHARTS_DIR, "cohort_retention_heatmap.png")
    plt.savefig(chart2, dpi=300)
    plt.close()
    print(f"[OK] Saved Cohort Retention Heatmap to: {chart2}")

    # 3. RFM Customer Segments Breakdown
    plt.figure(figsize=(10, 5))
    barplot = sns.barplot(
        x="Total_Revenue", y="Segment", data=rfm_summary,
        palette="viridis"
    )
    for p in barplot.patches:
        barplot.annotate(
            f"${p.get_width()/1000:,.1f}k",
            (p.get_width() + 1000, p.get_y() + p.get_height() / 2),
            ha="left", va="center", fontsize=9, fontweight="bold"
        )
    plt.title("Revenue Contribution by RFM Customer Segment", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Net Revenue ($)", fontsize=11, fontweight="bold")
    plt.ylabel("Customer Segment", fontsize=11, fontweight="bold")
    plt.xlim(0, rfm_summary["Total_Revenue"].max() * 1.18)
    plt.tight_layout()
    chart3 = os.path.join(CHARTS_DIR, "rfm_customer_distribution.png")
    plt.savefig(chart3, dpi=300)
    plt.close()
    print(f"[OK] Saved RFM Customer Distribution to: {chart3}")

    # 4. Product Category Basket Affinity Matrix
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        affinity_matrix, annot=True, fmt="d", cmap="magma_r",
        linewidths=0.5, linecolor="#cbd5e1"
    )
    plt.title("Cross-Category Basket Co-Occurrence Affinity Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    chart4 = os.path.join(CHARTS_DIR, "product_affinity_heatmap.png")
    plt.savefig(chart4, dpi=300)
    plt.close()
    print(f"[OK] Saved Product Affinity Heatmap to: {chart4}")

def generate_executive_html_dashboard(kpis: dict, rfm_summary: pd.DataFrame, sc_summary: pd.DataFrame):
    """
    Builds a complete interactive executive sales & supply chain dashboard in HTML.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_file = os.path.join(REPORTS_DIR, "executive_sales_dashboard.html")

    rfm_rows = ""
    for _, row in rfm_summary.iterrows():
        rfm_rows += f"""
        <tr>
            <td><strong>{row['Segment']}</strong></td>
            <td>{row['Customer_Count']:,}</td>
            <td>${row['Total_Revenue']:,.2f}</td>
            <td><strong>{row['Revenue_Share_Pct']}%</strong></td>
            <td>{row['Avg_Recency_Days']} days</td>
            <td>{row['Avg_Frequency']}</td>
            <td>${row['Avg_Monetary']:,.2f}</td>
        </tr>
        """

    sc_rows = ""
    for _, row in sc_summary.head(8).iterrows():
        sc_rows += f"""
        <tr>
            <td>{row['Category']}</td>
            <td><strong>{row['Product_Name']}</strong></td>
            <td>{row['Total_Units_Sold']:,}</td>
            <td>{row['Inventory_Turnover_Ratio']}x</td>
            <td>{row['Days_Sales_Inventory']}d</td>
            <td><strong>{row['GMROI']}</strong></td>
            <td>{row['Safety_Stock_Units']} units</td>
            <td>{row['Reorder_Point_Units']} units</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataVoyager — Executive E-Commerce & Supply Chain Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --card: rgba(17, 24, 39, 0.9);
            --border: rgba(55, 65, 81, 0.5);
            --accent: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --text: #f9fafb;
            --text-sub: #9ca3af;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}
        body {{
            background: radial-gradient(circle at top right, #111827 0%, #030712 100%);
            color: var(--text);
            padding: 30px 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 25px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 35px;
        }}
        .kpi-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px;
            backdrop-filter: blur(10px);
        }}
        .kpi-label {{ font-size: 12px; font-weight: 600; color: var(--text-sub); text-transform: uppercase; }}
        .kpi-val {{ font-size: 28px; font-weight: 800; margin: 8px 0; color: #fff; }}
        .kpi-sub {{ font-size: 12px; color: var(--success); font-weight: 500; }}
        
        .section {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 26px;
            margin-bottom: 30px;
        }}
        .sec-title {{ font-size: 19px; font-weight: 700; margin-bottom: 18px; color: #f3f4f6; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; text-align: left; }}
        th {{ background: rgba(31, 41, 55, 0.7); color: var(--text-sub); padding: 12px 14px; font-weight: 600; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #d1d5db; }}
        
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 24px;
        }}
        .chart-card {{
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .chart-card h4 {{ margin-bottom: 12px; font-size: 14px; color: #93c5fd; }}
        .chart-card img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        
        footer {{ text-align: center; color: var(--text-sub); font-size: 13px; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>DataVoyager — Global Retail & Supply Chain Intelligence</h1>
                <p style="color: var(--text-sub); font-size: 14px; margin-top: 4px;">NumPy Vectorized Analytics, RFM Segmentation & Cohort Performance</p>
            </div>
            <div class="badge">Active Analytics Pipeline</div>
        </header>

        <!-- KPI Metrics -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Total Net Revenue</div>
                <div class="kpi-val">${kpis.get('total_revenue', 0):,.2f}</div>
                <div class="kpi-sub">Across 15,000 Transactions</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Gross Profit</div>
                <div class="kpi-val">${kpis.get('total_profit', 0):,.2f}</div>
                <div class="kpi-sub">Margin: {kpis.get('profit_margin', 0):.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Active Customer Base</div>
                <div class="kpi-val">{kpis.get('total_customers', 0):,}</div>
                <div class="kpi-sub">Avg Order Value: ${kpis.get('aov', 0):.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Return Rate</div>
                <div class="kpi-val">{kpis.get('return_rate', 0):.2f}%</div>
                <div class="kpi-sub">Industry Benchmark: &lt; 6.0%</div>
            </div>
        </div>

        <!-- RFM Segmentation Table -->
        <div class="section">
            <div class="sec-title">👥 RFM Customer Segmentation & Spend Analysis</div>
            <table>
                <thead>
                    <tr>
                        <th>Customer Tier</th>
                        <th>Cohort Size</th>
                        <th>Net Revenue</th>
                        <th>Rev Share</th>
                        <th>Avg Recency</th>
                        <th>Avg Frequency</th>
                        <th>Avg Spend</th>
                    </tr>
                </thead>
                <tbody>
                    {rfm_rows}
                </tbody>
            </table>
        </div>

        <!-- Visual Analytics Grid -->
        <div class="section">
            <div class="sec-title">📈 Analytics Charts & Performance Visuals</div>
            <div class="charts-container">
                <div class="chart-card">
                    <h4>Monthly Revenue & Profit Dynamics</h4>
                    <img src="charts/monthly_sales_trend.png" alt="Sales Trend">
                </div>
                <div class="chart-card">
                    <h4>Monthly Customer Retention Cohort Matrix</h4>
                    <img src="charts/cohort_retention_heatmap.png" alt="Cohort Heatmap">
                </div>
                <div class="chart-card">
                    <h4>Revenue Distribution Across RFM Segments</h4>
                    <img src="charts/rfm_customer_distribution.png" alt="RFM Distribution">
                </div>
                <div class="chart-card">
                    <h4>Product Category Basket Co-Occurrence Affinity</h4>
                    <img src="charts/product_affinity_heatmap.png" alt="Affinity Heatmap">
                </div>
            </div>
        </div>

        <!-- Supply Chain KPIs -->
        <div class="section">
            <div class="sec-title">📦 Supply Chain, Inventory Turnover & Reorder Point Optimization</div>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Product</th>
                        <th>Units Sold</th>
                        <th>Turnover (ITR)</th>
                        <th>DSI (Days)</th>
                        <th>GMROI</th>
                        <th>Safety Stock</th>
                        <th>Reorder Point (ROP)</th>
                    </tr>
                </thead>
                <tbody>
                    {sc_rows}
                </tbody>
            </table>
        </div>

        <footer>
            DataVoyager Analytics Pipeline • Production Ready • Automated Execution
        </footer>
    </div>
</body>
</html>
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Interactive Executive Dashboard compiled at: {report_file}")
    return report_file
