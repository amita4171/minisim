"""
MiniSim Streamlit Dashboard — 6-panel swarm prediction visualization.

Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import statistics

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.offline_engine import swarm_score_offline

st.set_page_config(page_title="MiniSim Swarm Prediction", layout="wide")
st.title("MiniSim — Swarm Prediction Engine")

# =====================================================================
# Panel 1: Input Form (Sidebar)
# =====================================================================
with st.sidebar:
    st.header("Simulation Parameters")
    question = st.text_input(
        "Prediction Question",
        value="Will the Fed cut rates in May 2026?",
    )
    context = st.text_area("Additional Context", value="", height=100)
    n_agents = st.slider("Number of Agents", 10, 500, 50, step=10)
    n_rounds = st.slider("Deliberation Rounds", 1, 10, 4)
    market_price = st.number_input(
        "Market Price (0-1)", min_value=0.0, max_value=1.0, value=0.40, step=0.05
    )
    peer_sample_size = st.slider("Peer Sample Size", 3, 15, 5)
    use_web_research = st.checkbox("Enable Web Research (RAG)", value=False)

    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Run simulation
if run_button:
    with st.spinner(f"Running {n_agents} agents x {n_rounds} rounds..."):
        result = swarm_score_offline(
            question=question,
            context=context,
            n_agents=n_agents,
            rounds=n_rounds,
            market_price=market_price,
            peer_sample_size=peer_sample_size,
            use_web_research=use_web_research,
        )
    st.session_state["result"] = result
    st.session_state["question"] = question

if "result" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Simulation** to start.")
    st.stop()

result = st.session_state["result"]
agents_data = result["agents"]
df_agents = pd.DataFrame(agents_data)

# =====================================================================
# Top-level metrics
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Swarm P(YES)", f"{result['swarm_probability_yes']:.3f}")
col2.metric("Market Price", f"{result.get('market_price', 'N/A')}")
col3.metric("Edge", f"{result.get('edge', 0):+.3f}")
col4.metric("Diversity", f"{result.get('diversity_score', 0):.3f}")

st.divider()

# =====================================================================
# Panel 2: Probability Gauge + Panel 3: Convergence Chart (side by side)
# =====================================================================
gauge_col, conv_col = st.columns(2)

with gauge_col:
    st.subheader("Probability Gauge")
    ci = result["confidence_interval"]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result["swarm_probability_yes"],
        delta={"reference": market_price, "position": "bottom", "prefix": "vs market: "},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 2},
            "bar": {"color": "#1f77b4", "thickness": 0.6},
            "steps": [
                {"range": [0, 0.3], "color": "#ffcccc"},
                {"range": [0.3, 0.7], "color": "#ffffcc"},
                {"range": [0.7, 1.0], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.8,
                "value": market_price,
            },
        },
        title={"text": "Swarm P(YES)"},
        number={"font": {"size": 48}},
    ))
    # Add CI band as annotation
    fig_gauge.add_annotation(
        x=0.5, y=-0.15,
        text=f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]",
        showarrow=False,
        font={"size": 14},
        xref="paper", yref="paper",
    )
    fig_gauge.update_layout(height=350, margin={"t": 50, "b": 50, "l": 30, "r": 30})
    st.plotly_chart(fig_gauge, use_container_width=True)

with conv_col:
    st.subheader("Convergence Chart")
    convergence = result.get("convergence", [])
    if convergence:
        df_conv = pd.DataFrame(convergence)

        fig_conv = go.Figure()

        # Individual agent traces (thin, transparent)
        for agent in agents_data[:30]:  # limit to 30 for performance
            fig_conv.add_trace(go.Scatter(
                x=list(range(len(agent["score_history"]))),
                y=agent["score_history"],
                mode="lines",
                line={"width": 0.5, "color": "rgba(150,150,150,0.3)"},
                showlegend=False,
                hoverinfo="skip",
            ))

        # Mean line
        fig_conv.add_trace(go.Scatter(
            x=df_conv["round"],
            y=df_conv["mean_score"],
            mode="lines+markers",
            name="Mean Score",
            line={"width": 3, "color": "#1f77b4"},
            marker={"size": 8},
        ))

        # Std dev band
        fig_conv.add_trace(go.Scatter(
            x=list(df_conv["round"]) + list(df_conv["round"][::-1]),
            y=list(df_conv["mean_score"] + df_conv["stdev"]) + list((df_conv["mean_score"] - df_conv["stdev"])[::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.15)",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ))

        # Market price reference
        fig_conv.add_hline(y=market_price, line_dash="dash", line_color="red",
                          annotation_text="Market Price")

        fig_conv.update_layout(
            xaxis_title="Round",
            yaxis_title="P(YES)",
            yaxis_range=[0, 1],
            height=350,
            margin={"t": 30, "b": 50},
        )
        st.plotly_chart(fig_conv, use_container_width=True)

st.divider()

# =====================================================================
# Panel 4: Agent Scatter + Panel 5: Opinion Distribution (side by side)
# =====================================================================
scatter_col, dist_col = st.columns(2)

with scatter_col:
    st.subheader("Agent Scatter: Initial vs Final Score")
    fig_scatter = px.scatter(
        df_agents,
        x="initial_score",
        y="final_score",
        color="temp_tier",
        hover_data=["name", "background_category", "personality", "confidence"],
        color_discrete_map={
            "analyst": "#1f77b4",
            "calibrator": "#2ca02c",
            "contrarian": "#ff7f0e",
            "creative": "#d62728",
        },
        labels={
            "initial_score": "Initial P(YES)",
            "final_score": "Final P(YES)",
            "temp_tier": "Temperature Tier",
        },
    )
    # Diagonal line (no change)
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line={"dash": "dash", "color": "gray", "width": 1},
    )
    fig_scatter.update_layout(
        height=400,
        margin={"t": 30, "b": 50},
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with dist_col:
    st.subheader("Opinion Distribution (Final)")
    histogram = result.get("histogram", {})
    if histogram:
        df_hist = pd.DataFrame({
            "bucket": list(histogram.keys()),
            "count": list(histogram.values()),
        })
        fig_hist = px.bar(
            df_hist, x="bucket", y="count",
            color="count",
            color_continuous_scale="RdYlGn",
            labels={"bucket": "P(YES) Range", "count": "Agent Count"},
        )
        fig_hist.update_layout(
            height=400,
            margin={"t": 30, "b": 50},
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# =====================================================================
# Panel 6: Top Voices + Mind Changers + Dissenting Voices
# =====================================================================
st.subheader("Key Voices")

yes_col, no_col, change_col = st.columns(3)

with yes_col:
    st.markdown("**Top YES Voices**")
    for v in result.get("top_yes_voices", [])[:3]:
        with st.container(border=True):
            st.markdown(f"**{v['name']}** — {v['background']}")
            st.markdown(f"Score: `{v['initial_score']:.2f}` -> `{v['final_score']:.2f}` | Conf: `{v['confidence']:.2f}`")
            reasoning = v.get("reasoning", "")
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            st.caption(reasoning)

with no_col:
    st.markdown("**Top NO Voices**")
    for v in result.get("top_no_voices", [])[:3]:
        with st.container(border=True):
            st.markdown(f"**{v['name']}** — {v['background']}")
            st.markdown(f"Score: `{v['initial_score']:.2f}` -> `{v['final_score']:.2f}` | Conf: `{v['confidence']:.2f}`")
            reasoning = v.get("reasoning", "")
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            st.caption(reasoning)

with change_col:
    st.markdown("**Mind Changers**")
    mind_changers = result.get("mind_changers", [])
    if mind_changers:
        for mc in mind_changers[:3]:
            with st.container(border=True):
                direction_emoji = "+" if mc["shift"] > 0 else ""
                st.markdown(f"**{mc['name']}** — {mc['background']}")
                st.markdown(f"Shift: `{mc['initial_score']:.2f}` -> `{mc['final_score']:.2f}` ({direction_emoji}{mc['shift']:.2f})")
                st.caption(mc.get("shift_direction", ""))
    else:
        st.info("No significant mind changes detected.")

# Dissenting voices
if result.get("dissenting_voices"):
    with st.expander("Dissenting Voices (Statistical Outliers)"):
        for dv in result["dissenting_voices"]:
            st.markdown(
                f"**{dv['name']}** ({dv['background']}): "
                f"P(YES) = {dv['final_score']:.2f} | z-score = {dv.get('z_score', 0):.1f}"
            )
            st.caption(dv.get("last_reflection", ""))

# =====================================================================
# Reasoning Shift Summary + Clusters
# =====================================================================
st.divider()
col_summary, col_clusters = st.columns(2)

with col_summary:
    st.subheader("Reasoning Shift Summary")
    st.write(result.get("reasoning_shift_summary", ""))

    st.subheader("Simulation Config")
    config = result.get("config", {})
    st.json(config)

with col_clusters:
    st.subheader("Opinion Clusters")
    clusters = result.get("opinion_clusters", [])
    if clusters:
        df_clusters = pd.DataFrame(clusters)
        fig_clusters = px.bar(
            df_clusters, x="label", y="n_agents",
            color="mean_score",
            color_continuous_scale="RdYlGn",
            labels={"label": "Cluster", "n_agents": "Agents", "mean_score": "Mean Score"},
        )
        fig_clusters.update_layout(height=300, margin={"t": 30, "b": 50})
        st.plotly_chart(fig_clusters, use_container_width=True)

# =====================================================================
# Raw data export
# =====================================================================
with st.expander("Raw JSON Output"):
    st.json(result)
