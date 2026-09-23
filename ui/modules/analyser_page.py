"""Analyser module page."""

from typing import Any

import streamlit as st
from streamlit.components.v1 import html as components_html

from common import api_post, format_response, parse_comma_list, parse_optional, parse_optional_float, parse_optional_int


def _maybe_pie_chart(data: dict[str, float], title: str) -> None:
    st.markdown(f"### {title}")
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.bar_chart(data)
        return

    values = [{"label": label, "value": float(value)} for label, value in data.items() if value is not None]
    if not values:
        return

    df = pd.DataFrame(values)
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="label", type="nominal"),
            tooltip=["label", "value"],
        )
        .properties(title=title)
    )
    st.altair_chart(chart, width="content")


def _build_loops_html(response: dict) -> str | None:
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    net = Network(height="650px", width="100%", directed=True)
    net.toggle_physics(True)
    net.barnes_hut()

    seen_nodes: set[str] = set()

    for loop_type, entries, color in [
        ("self", response.get("self_loops", []), "blue"),
        ("non-self", response.get("non_self_loops", []), "orange"),
        ("repeated", response.get("repeated_patterns", []), "green"),
    ]:
        for item in entries:
            pattern = str(item.get("pattern", "")).strip()
            support = item.get("support", 0)
            nodes = [node.strip() for node in pattern.split("->") if node.strip()]
            for node in nodes:
                if node not in seen_nodes:
                    net.add_node(node, label=node, title=f"Activity {node}", color="lightgray")
                    seen_nodes.add(node)

            if len(nodes) == 1:
                net.add_edge(nodes[0], nodes[0], label=f"{support:.2f}", title=f"{pattern} ({support})", color=color)
            else:
                for i in range(len(nodes) - 1):
                    net.add_edge(
                        nodes[i],
                        nodes[i + 1],
                        label=f"{support:.2f}",
                        title=f"{pattern} ({support})",
                        color=color,
                    )

    return net.generate_html()


def _render_analyser_response(response: Any, method: str) -> None:
    if isinstance(response, dict) and (response.get("error") or (response.get("status_code") and response["status_code"] >= 400)):
        format_response(response)
        return

    if isinstance(response, dict) and response.get("code") is not None:
        code = response["code"]
        message = response.get("message")
        if code != 200:
            st.error(f"Error (code {code})" + (f": {message}" if message else ""))
            return
        st.success(f"Success ({code})")

    if method == "loop_detection" and isinstance(response, dict):
        total_groups = response.get("total_groups")
        self_loops = response.get("self_loops", []) or []
        non_self_loops = response.get("non_self_loops", []) or []
        repeated_patterns = response.get("repeated_patterns", []) or []

        st.subheader("Loop detection summary")
        if total_groups is not None:
            st.metric("Total groups", total_groups)
        st.metric("Self-loop patterns", len(self_loops))
        st.metric("Non-self-loop patterns", len(non_self_loops))
        if repeated_patterns:
            st.metric("Repeated patterns", len(repeated_patterns))

        counts = {
            "Self loops": len(self_loops),
            "Non-self loops": len(non_self_loops),
        }
        if repeated_patterns:
            counts["Repeated patterns"] = len(repeated_patterns)
        _maybe_pie_chart(counts, "Loop pattern types")

        if self_loops:
            st.subheader("Self loops")
            st.table(self_loops)

        if non_self_loops:
            st.subheader("Non-self loops")
            st.table(non_self_loops)

        if repeated_patterns:
            st.subheader("Repeated patterns")
            st.table(repeated_patterns)

        loop_support = {}
        for item in self_loops + non_self_loops + repeated_patterns:
            loop_support[item.get("pattern", "?")] = float(item.get("support", 0) or 0)
        if loop_support:
            st.markdown("### Loop support by pattern")
            st.bar_chart(loop_support)

        graph_html = _build_loops_html(response)
        with st.expander("Loop visualization"):
            st.markdown(
                "Visual representation of loop transitions. Self-loops are shown as self-edges; non-self loops are displayed as directed paths. "
                "Hover over edges to see pattern support."
            )
            if graph_html:
                components_html(graph_html, height=700, scrolling=True)
            else:
                st.info("Install pyvis for a richer loop visualization, or view the loop tables above.")

        with st.expander("Raw response"):
            st.json(response)
        return

    if isinstance(response, dict) and "data" in response:
        data = response["data"]
        st.subheader(f"{method.replace('_', ' ').title()} results")
        st.table(data)

        if method == "durations":
            support_chart = {
                item.get("activity", "?"): float(item.get("avg_duration_sec", 0) or 0)
                for item in data
                if item.get("activity") is not None
            }
            count_chart = {
                item.get("activity", "?"): int(item.get("occurrence_count", 0) or 0)
                for item in data
                if item.get("activity") is not None
            }
            if support_chart:
                st.markdown("### Average duration by activity")
                st.bar_chart(support_chart)
            if count_chart:
                st.markdown("### Occurrence count by activity")
                st.line_chart(count_chart)

        if method == "directly_follows":
            support_chart = {
                f"{item.get('source','?')}→{item.get('target','?')}": float(item.get("support", 0) or 0)
                for item in data
            }
            duration_chart = {
                f"{item.get('source','?')}→{item.get('target','?')}": float(item.get("avg_duration_sec", 0) or 0)
                for item in data
            }
            if support_chart:
                st.markdown("### Support by direct follow edge")
                st.bar_chart(support_chart)
            if duration_chart:
                st.markdown("### Average duration by direct follow edge")
                st.line_chart(duration_chart)

        with st.expander("Raw response"):
            st.json(response)
        return

    if method == "attribute_deviations":
        # HTML output: api_post falls back to {"status_code": 200, "text": "<html...>"}
        if isinstance(response, dict) and response.get("status_code") == 200 and "text" in response:
            components_html(response["text"], height=800, scrolling=True)
            return
        # JSON output: {"log_name": ..., "total_deviations": ..., "deviations": [...]}
        if isinstance(response, dict) and "deviations" in response:
            st.subheader("Attribute deviations")
            st.metric("Total deviations", response.get("total_deviations", 0))
            deviations = response.get("deviations") or []
            if deviations:
                st.table(deviations)
            with st.expander("Raw response"):
                st.json(response)
            return

    format_response(response)


def render(base_url: str) -> None:
    if "analyser_running" not in st.session_state:
        st.session_state.analyser_running = False
    if "analyser_submit_requested" not in st.session_state:
        st.session_state.analyser_submit_requested = False

    st.title("🧠 Analyser")
    st.markdown("Run analysis methods on indexed logs, such as directly-follow, durations, loop detection, and attribute deviations.")

    method = st.selectbox(
        "Analyser method",
        ["directly_follows", "durations", "loop_detection", "attribute_deviations"],
    )

    with st.form("analyser_form"):
        log_name = st.text_input("Log name", "example_log")
        storage_namespace = st.text_input("Storage namespace", "siesta")

        if method == "directly_follows":
            min_timestamp = parse_optional(st.text_input("Minimum timestamp (ISO 8601)", ""))
            end_time = parse_optional(st.text_input("End time attribute", ""))
            support_threshold = parse_optional_float(st.text_input("Support threshold", ""))
            filter_out = st.checkbox("Filter out rare items", value=False)
            include_traces = st.checkbox("Include trace IDs", value=False)
            analyser_config = {
                "log_name": log_name,
                "storage_namespace": storage_namespace,
                "min_timestamp": min_timestamp,
                "end_time": end_time,
                "support_threshold": support_threshold,
                "filter_out": filter_out,
                "include_traces": include_traces,
                "return_csv": False,
                "output_path": f"output/{log_name}_directly_follows",
            }

        elif method == "durations":
            duration_mode = st.selectbox("Duration mode", ["activity", "group"], index=0)
            min_timestamp = parse_optional(st.text_input("Minimum timestamp (ISO 8601)", ""))
            end_time = parse_optional(st.text_input("End time attribute", ""))
            grouping_key = parse_comma_list(st.text_input("Grouping key(s)", ""))
            grouping_value = parse_comma_list(st.text_input("Grouping value(s)", ""))
            per_group = st.checkbox("Per group output", value=False)
            analyser_config = {
                "log_name": log_name,
                "storage_namespace": storage_namespace,
                "duration_mode": duration_mode,
                "min_timestamp": min_timestamp,
                "end_time": end_time,
                "grouping_key": grouping_key,
                "grouping_value": grouping_value,
                "per_group": per_group,
                "return_csv": False,
                "output_path": f"output/{log_name}_durations",
            }

        elif method == "loop_detection":
            grouping_key = parse_comma_list(st.text_input("Grouping key(s)", ""))
            grouping_value = parse_comma_list(st.text_input("Grouping value(s)", ""))
            min_timestamp = parse_optional(st.text_input("Minimum timestamp (ISO 8601)", ""))
            support_threshold = parse_optional_float(st.text_input("Support threshold", ""))
            filter_out = st.checkbox("Filter out rare loops", value=False)
            top_k = parse_optional_int(st.text_input("Top K loops", ""))
            trace_based = st.checkbox("Include trace IDs", value=False)
            repeated_patterns = st.checkbox("Detect repeated patterns (A→B→C … A→B→C)", value=False)
            max_pattern_window = st.number_input(
                "Max repeated-pattern window", value=8, min_value=2, disabled=not repeated_patterns
            )
            analyser_config = {
                "log_name": log_name,
                "storage_namespace": storage_namespace,
                "grouping_key": grouping_key,
                "grouping_value": grouping_value,
                "min_timestamp": min_timestamp,
                "support_threshold": support_threshold,
                "filter_out": filter_out,
                "top_k": top_k,
                "trace_based": trace_based,
                "repeated_patterns": repeated_patterns,
                "max_pattern_window": int(max_pattern_window),
                "return_csv": False,
                "output_path": f"output/{log_name}_loop_detection",
            }

        else:
            _STEP_LABELS = {
                "0 – Value frequency (inter + intra-trace)": 0,
                "1 – Activity * Attribute anomalies": 1,
                "2 – Position-conditioned anomalies": 2,
                "3 – N-gram context anomalies": 3,
                "4 – Value transitions (categorical)": 4,
            }
            min_timestamp = parse_optional(st.text_input("Minimum timestamp (ISO 8601)", ""))
            selected_step_labels = st.multiselect(
                "Steps to run",
                list(_STEP_LABELS.keys()),
                default=list(_STEP_LABELS.keys()),
            )
            steps = [_STEP_LABELS[lbl] for lbl in selected_step_labels]
            excluded_attributes = parse_comma_list(st.text_input("Excluded attributes", ""))
            surprise_threshold = st.number_input("Surprise threshold", value=4.0, format="%.2f")
            zscore_threshold = st.number_input("Z-score threshold", value=3.5, format="%.2f")
            ngram_n = st.number_input("N-gram length", value=2, min_value=1)
            min_group_size = st.number_input("Minimum group size", value=5, min_value=1)
            n_buckets = st.number_input("N buckets", value=5, min_value=1)
            support_threshold = parse_optional_float(st.text_input("Support threshold", ""))
            filter_out = st.checkbox("Filter out by support", value=False)
            on_rare = parse_optional_float(st.text_input("Rare-mode support threshold", ""))
            analyser_config = {
                "log_name": log_name,
                "storage_namespace": storage_namespace,
                "min_timestamp": min_timestamp,
                "steps": steps,
                "excluded_attributes": excluded_attributes,
                "surprise_threshold": surprise_threshold,
                "zscore_threshold": zscore_threshold,
                "ngram_n": ngram_n,
                "min_group_size": min_group_size,
                "n_buckets": n_buckets,
                "support_threshold": support_threshold,
                "filter_out": filter_out,
                "on_rare": on_rare,
                "output_format": "html",
                "output_path": f"output/{log_name}_analyser_results",
            }

        submit = st.form_submit_button("Run analyser", disabled=st.session_state.analyser_running, key="run_analyser_button")
        if submit:
            st.session_state.analyser_submit_requested = True

    if st.session_state.analyser_submit_requested and not st.session_state.analyser_running:
        st.session_state.analyser_running = True
        st.session_state.analyser_submit_requested = False
        try:
            with st.spinner("Running analyser..."):
                response = api_post(f"analysing/{method}", base_url, payload=analyser_config)
                _render_analyser_response(response, method)
        finally:
            st.session_state.analyser_running = False

    with st.expander("Need help?"):
        st.markdown(
            "- Choose the analyser method to update the form fields automatically.\n"
            "- Use empty values for optional settings to keep the default backend behavior."
        )
