"""
Reusable SaaS Visual Components for Streamlit Bug Intelligence
Interactive Plotly charts, KPI metrics, candidate cards, and headers.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any
from .styles import SEVERITY_COLORS, get_severity_badge_html


def render_header(title: str = "Bug Intelligence", subtitle: str = "Enterprise Duplicate Detection & Severity Prediction System"):
    """Renders the top application header with modern dark styling."""
    st.markdown(
        f"""
        <div class="saas-header">
            <div>
                <h1 class="saas-header-title">🛡️ {title}</h1>
                <div class="saas-header-sub">{subtitle}</div>
            </div>
            <div style="text-align: right;">
                <span class="saas-badge-live">● PRODUCTION ML ENGINE READY</span>
                <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Eclipse Bugzilla v2.4 • MiniLM-L6-v2</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_card(title: str, value: str, subtext: str = "", delta: str = None):
    """Renders a single SaaS KPI metric card."""
    delta_html = f'<div style="font-size: 11px; color: #10b981; margin-top: 2px;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{title}</div>
            <div class="kpi-sub">{subtext}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_duplicate_card(result: Dict[str, Any], show_details: bool = True):
    """Renders a single candidate duplicate report card."""
    rank = result.get("rank", 1)
    bug_id = result.get("bug_id", "N/A")
    title = result.get("title", "Untitled Bug Report")
    description = result.get("description", "No description provided.")
    hybrid_sim = result.get("hybrid_similarity", result.get("similarity_score", 0.0))
    tfidf_sim = result.get("tfidf_similarity", 0.0)
    sbert_sim = result.get("sbert_similarity", 0.0)
    severity = result.get("severity", "normal")
    badge_html = get_severity_badge_html(severity)
    
    dup_flag = '<span style="color:#ef4444; font-size:11px; font-weight:700; margin-left: 6px;">[KNOWN DUPLICATE]</span>' if result.get("is_known_duplicate") else ""

    pct_hybrid = round(float(hybrid_sim) * 100, 1)
    pct_tfidf = round(float(tfidf_sim) * 100, 1)
    pct_sbert = round(float(sbert_sim) * 100, 1)

    st.markdown(
        f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div>
                    <span style="font-weight: 700; color: #818cf8; font-size: 15px; font-family: 'JetBrains Mono', monospace;">
                        #{rank} &bull; Bug #{bug_id}
                    </span>
                    {dup_flag}
                    <div class="candidate-title" style="margin-top: 4px;">{title}</div>
                </div>
                <div style="text-align: right;">
                    <span class="score-pill-hybrid">{pct_hybrid}% Hybrid</span>
                </div>
            </div>
            <div style="display: flex; gap: 8px; margin-bottom: 10px; align-items: center;">
                <span class="score-pill-sub">TF-IDF: {pct_tfidf}%</span>
                <span class="score-pill-sub">SBERT: {pct_sbert}%</span>
                <span style="margin-left: auto;">{badge_html}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if show_details:
        with st.expander(f"📄 View Bug #{bug_id} Technical Description & Stack Trace"):
            st.markdown(f"```text\n{description}\n```")


def plot_severity_distribution_bar(probabilities: Dict[str, float]):
    """Renders interactive Plotly horizontal bar chart for severity probabilities."""
    classes = ["blocker", "critical", "major", "normal", "minor", "trivial"]
    vals = [probabilities.get(c, 0.0) * 100 for c in classes]
    colors = [SEVERITY_COLORS.get(c, "#3b82f6") for c in classes]

    fig = go.Figure(go.Bar(
        x=vals,
        y=[c.capitalize() for c in classes],
        orientation='h',
        marker=dict(color=colors, line=dict(color='#1e293b', width=1)),
        text=[f"{v:.1f}%" for v in vals],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=12, family="Inter")
    ))

    fig.update_layout(
        plot_bgcolor='#0b0f19',
        paper_bgcolor='#0b0f19',
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(
            title="Predicted Class Probability (%)",
            range=[0, max(max(vals) * 1.25, 40)],
            showgrid=True,
            gridcolor='#1e293b',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color='#f1f5f9', size=12, family="Inter")
        ),
        height=260,
        font=dict(family="Inter", color="#f1f5f9")
    )
    return fig


def plot_dataset_severity_breakdown(full_counts: Dict[str, int], verified_counts: Dict[str, int]):
    """Renders grouped bar chart comparing Full vs Verified-only severity counts."""
    classes = ["normal", "major", "minor", "critical", "trivial", "blocker"]
    full_vals = [full_counts.get(c, 0) for c in classes]
    ver_vals = [verified_counts.get(c, 0) for c in classes]

    fig = go.Figure(data=[
        go.Bar(name='Full Dataset (84,653)', x=[c.capitalize() for c in classes], y=full_vals, marker_color='#3b82f6'),
        go.Bar(name='Directly Verified (14,924)', x=[c.capitalize() for c in classes], y=ver_vals, marker_color='#818cf8')
    ])

    fig.update_layout(
        barmode='group',
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(font=dict(color='#f1f5f9'), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='#1f2937', tickfont=dict(color='#94a3b8')),
        yaxis=dict(gridcolor='#1f2937', tickfont=dict(color='#94a3b8'), title="Number of Bug Reports"),
        height=320,
        font=dict(family="Inter", color="#f1f5f9")
    )
    return fig


def plot_duplicate_metrics_bar(metrics_df: pd.DataFrame):
    """Renders comparison bar chart for Duplicate Retrieval models."""
    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={
            "TF-IDF (Lexical)": "#60a5fa",
            "SBERT (Dense Semantic)": "#a78bfa",
            "Hybrid (Dense + Sparse)": "#34d399"
        }
    )
    fig.update_layout(
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(font=dict(color='#f1f5f9'), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='#1f2937', tickfont=dict(color='#94a3b8'), title=""),
        yaxis=dict(gridcolor='#1f2937', tickfont=dict(color='#94a3b8'), title="Evaluation Score", range=[0, 0.75]),
        height=340,
        font=dict(family="Inter", color="#f1f5f9")
    )
    return fig
