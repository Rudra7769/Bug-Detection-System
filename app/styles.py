"""
Professional SaaS Dark Theme Styles and Custom CSS for Bug Intelligence
Linear / GitHub / Enterprise Software Intelligence Dashboard aesthetics.
"""

CUSTOM_CSS = """
<style>
/* Modern Font import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Root & Global Overrides */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #f1f5f9;
}

/* Background canvas */
.stApp {
    background-color: #0b0f19;
}

/* Block container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1240px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Modern Top Header Banner */
.saas-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.saas-header-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.saas-header-sub {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 4px;
}

.saas-badge-live {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Metric / KPI Card */
.kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 18px 20px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.kpi-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 6px 18px rgba(59, 130, 246, 0.15);
    transform: translateY(-2px);
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.5px;
}

.kpi-label {
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}

/* Candidate Bug Card */
.candidate-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    transition: all 0.2s ease;
}

.candidate-card:hover {
    border-color: #4f46e5;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.12);
}

.candidate-title {
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 6px;
}

.candidate-desc {
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 12px;
}

/* Similarity Pills */
.score-pill-hybrid {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid #4f46e5;
    color: #a5b4fc;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
}

.score-pill-sub {
    background: #1f2937;
    color: #94a3b8;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
}

/* Severity Badges (Dark Mode SaaS Style) */
.badge-blocker {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #f87171;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-critical {
    background: rgba(249, 115, 22, 0.15);
    border: 1px solid rgba(249, 115, 22, 0.5);
    color: #fb923c;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-major {
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.5);
    color: #facc15;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-normal {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.5);
    color: #60a5fa;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-minor {
    background: rgba(148, 163, 184, 0.15);
    border: 1px solid rgba(148, 163, 184, 0.4);
    color: #cbd5e1;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-trivial {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

/* Callout Box */
.info-callout {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #3b82f6;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 16px 0;
    font-size: 14px;
    color: #cbd5e1;
}

/* Architecture Card */
.arch-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}

.arch-title {
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
"""

SEVERITY_COLORS = {
    "blocker": "#ef4444",   # Red
    "critical": "#f97316",  # Orange-red
    "major": "#eab308",     # Amber/Yellow
    "normal": "#3b82f6",    # Blue
    "minor": "#94a3b8",     # Slate
    "trivial": "#10b981"    # Green
}

def get_severity_badge_html(severity: str) -> str:
    """Returns styled SaaS HTML badge for bug severity."""
    sev = str(severity).lower()
    if sev not in SEVERITY_COLORS:
        sev = "normal"
    return f'<span class="badge-{sev}">{sev}</span>'
