"""
streamlit_app.py
================
Presentation-ready Streamlit UI + Gemini chatbot for the Synthetic Data Framework.
"""

import os
import time
import io
import zipfile
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

import config
from modules.data_loader import load_synthea_data
from modules.gan_trainer import train_all_models, generate_synthetic
from modules.privacy_layer import apply_differential_privacy, run_membership_inference_attack
from modules.quality_evaluator import run_quality_report, run_ml_utility_test
from modules.bias_auditor import run_bias_audit


load_dotenv()

st.set_page_config(
    page_title="Synthetic Data Framework Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


MODEL_LABELS = {
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "gaussian": "GaussianCopula",
}


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
          .main > div {
            padding-top: 1.2rem;
            max-width: 1300px;
          }
          .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
          }
          .title-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            border: 1px solid #d8e3ff;
            background: #eef3ff;
            color: #274690;
            margin-bottom: 0.6rem;
          }
          .section-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            background: #ffffff;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _save_synthetic(df: pd.DataFrame, model_name: str) -> str:
    os.makedirs(config.DATA_SYNTHETIC_DIR, exist_ok=True)
    file_name = f"synthetic_patients_{model_name}.csv"
    save_path = os.path.join(config.DATA_SYNTHETIC_DIR, file_name)
    df.to_csv(save_path, index=False)
    return save_path


def _summary_from_results(results: Dict[str, Any]) -> str:
    lines = []
    specs = results.get("specs", {})
    lines.append("Project: Synthetic Data Framework")
    lines.append(f"Problem focus: {specs.get('problem_focus', 'Not specified')}")
    lines.append(f"Privacy strictness: {specs.get('privacy_level', 'Not specified')}")
    lines.append(f"Selected models: {', '.join(specs.get('selected_models', []))}")
    lines.append("")
    lines.append("Model-wise metrics:")
    for model_key, item in results.get("model_results", {}).items():
        q = item["quality"]["overall"] * 100
        p = item["privacy_score"]
        b = item["bias"]["bias_reduction_pct"]
        tstr = item["ml"]["tstr_accuracy"] * 100
        trtr = item["ml"]["trtr_accuracy"] * 100
        lines.append(
            f"- {MODEL_LABELS.get(model_key, model_key)}: quality={q:.2f}%, "
            f"privacy_risk={p:.2f}, bias_reduction={b:.2f}%, "
            f"TSTR={tstr:.2f}%, TRTR={trtr:.2f}%"
        )
    return "\n".join(lines)


def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def _ask_gemini(user_question: str, results: Dict[str, Any]) -> str:
    model = _get_gemini_model()
    if model is None:
        return (
            "Gemini key not found. Add `GEMINI_API_KEY` in `.env`, then restart Streamlit."
        )

    context = _summary_from_results(results) if results else "No pipeline results available yet."
    prompt = (
        "You are an AI assistant for an M.Tech synthetic data demo.\n"
        "Explain clearly, short and practical.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_question}\n"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip() or "No response generated."


def _run_pipeline(
    n_samples: int,
    dp_epsilon: float,
    force_retrain: bool,
    selected_models: List[str],
    specs: Dict[str, Any],
) -> Dict[str, Any]:
    train_df, test_df, full_df = load_synthea_data()
    models = train_all_models(train_df, force_retrain=force_retrain)

    model_results: Dict[str, Any] = {}
    for model_name in selected_models:
        synthetic_df = generate_synthetic(models[model_name], n=n_samples)
        dp_synthetic_df = apply_differential_privacy(synthetic_df, epsilon=dp_epsilon)
        saved_path = _save_synthetic(dp_synthetic_df, model_name=model_name)
        quality_results = run_quality_report(full_df, dp_synthetic_df)
        ml_results = run_ml_utility_test(full_df, dp_synthetic_df)
        privacy_score = run_membership_inference_attack(full_df, dp_synthetic_df)
        bias_results = run_bias_audit(full_df, dp_synthetic_df)

        model_results[model_name] = {
            "synthetic_df": dp_synthetic_df,
            "quality": quality_results,
            "ml": ml_results,
            "privacy_score": privacy_score,
            "bias": bias_results,
            "saved_path": saved_path,
        }

    primary_model = selected_models[0]
    return {
        "train_df": train_df,
        "test_df": test_df,
        "real_df": full_df,
        "primary_model": primary_model,
        "model_results": model_results,
        "specs": specs,
    }


def _show_overview() -> None:
    st.markdown('<span class="title-chip">M.Tech Demo Dashboard</span>', unsafe_allow_html=True)
    st.title("Privacy-Preserving Synthetic Patient Data Generation")
    st.caption(
        "CTGAN + TVAE + GaussianCopula + Differential Privacy + Gemini Chatbot"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", "CTGAN / TVAE / Gaussian")
    c2.metric("Privacy Method", "Laplace DP")
    c3.metric("Evaluation", "Quality + Utility + Bias")
    c4.metric("AI Assistant", "Gemini 2.5 Flash")

    st.info("Use the sidebar to set specs first, then run model-wise execution.")


def _apply_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    filtered = df.copy()
    with st.expander("Filter Data", expanded=True):
        search = st.text_input(
            "Search (any column)",
            value="",
            key=f"{key_prefix}_search",
            placeholder="Type keyword to filter rows...",
        ).strip()
        if search:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask.any(axis=1)]

        num_cols = _numeric_columns(filtered)
        cat_cols = _categorical_columns(filtered)
        left, right = st.columns(2)
        with left:
            selected_num = st.selectbox(
                "Numeric column filter",
                options=["None"] + num_cols,
                key=f"{key_prefix}_num_col",
            )
            if selected_num != "None" and not filtered.empty:
                col_min = float(filtered[selected_num].min())
                col_max = float(filtered[selected_num].max())
                min_val, max_val = st.slider(
                    "Range",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"{key_prefix}_num_range",
                )
                filtered = filtered[
                    (filtered[selected_num] >= min_val)
                    & (filtered[selected_num] <= max_val)
                ]
        with right:
            selected_cat = st.selectbox(
                "Categorical column filter",
                options=["None"] + cat_cols,
                key=f"{key_prefix}_cat_col",
            )
            if selected_cat != "None" and not filtered.empty:
                values = sorted(filtered[selected_cat].astype(str).dropna().unique().tolist())
                selected_values = st.multiselect(
                    "Values",
                    options=values,
                    default=values[: min(5, len(values))],
                    key=f"{key_prefix}_cat_vals",
                )
                if selected_values:
                    filtered = filtered[filtered[selected_cat].astype(str).isin(selected_values)]

    return filtered


def _show_data_explorer(df: pd.DataFrame, key_prefix: str) -> None:
    filtered = _apply_filters(df, key_prefix=key_prefix)
    st.write(f"Showing **{len(filtered)}** / **{len(df)}** rows")
    st.dataframe(filtered, use_container_width=True, height=380)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(filtered))
    c2.metric("Columns", filtered.shape[1])
    c3.metric("Missing Values", int(filtered.isna().sum().sum()))

    summary = filtered.describe(include="all").transpose().fillna("")
    st.markdown("**Column Summary**")
    st.dataframe(summary, use_container_width=True, height=300)


def _show_compare_view(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> None:
    st.subheader("Real vs Synthetic Comparison")
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    if not common_cols:
        st.warning("No common columns between real and synthetic datasets.")
        return

    col = st.selectbox("Choose column to compare", options=common_cols, key="compare_col")
    is_numeric = pd.api.types.is_numeric_dtype(real_df[col]) and pd.api.types.is_numeric_dtype(syn_df[col])

    left, right = st.columns(2)
    with left:
        st.markdown("**Real Data Stats**")
        st.write(real_df[col].describe())
    with right:
        st.markdown("**Synthetic Data Stats**")
        st.write(syn_df[col].describe())

    st.markdown("**Distribution Preview**")
    if is_numeric:
        bins = 20
        real_hist, edges = np.histogram(real_df[col].dropna().astype(float), bins=bins)
        syn_hist, _ = np.histogram(syn_df[col].dropna().astype(float), bins=edges)
        plot_df = pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "Real": real_hist,
                "Synthetic": syn_hist,
            }
        ).set_index("bin_left")
        st.line_chart(plot_df, use_container_width=True)
    else:
        real_counts = real_df[col].astype(str).value_counts(normalize=True).rename("Real")
        syn_counts = syn_df[col].astype(str).value_counts(normalize=True).rename("Synthetic")
        cat_df = (
            pd.concat([real_counts, syn_counts], axis=1)
            .fillna(0)
            .sort_values("Real", ascending=False)
            .head(15)
        )
        st.bar_chart(cat_df, use_container_width=True)


def _show_model_wise_results(results: Dict[str, Any], elapsed: float) -> None:
    st.subheader("Individual GAN/Model Execution")
    model_rows = []
    for key, item in results["model_results"].items():
        model_rows.append(
            {
                "Model": MODEL_LABELS.get(key, key),
                "Quality %": round(item["quality"]["overall"] * 100, 2),
                "Privacy Risk": item["privacy_score"],
                "Bias Reduction %": item["bias"]["bias_reduction_pct"],
                "TSTR %": round(item["ml"]["tstr_accuracy"] * 100, 2),
                "TRTR %": round(item["ml"]["trtr_accuracy"] * 100, 2),
                "CSV Path": item["saved_path"],
            }
        )
    model_df = pd.DataFrame(model_rows).sort_values("Quality %", ascending=False)
    st.dataframe(model_df, use_container_width=True, height=230)
    st.caption(f"Total pipeline runtime: {elapsed:.1f}s")

    st.markdown("### Download Generated Datasets")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for model_key, item in results["model_results"].items():
            csv_name = f"synthetic_patients_{model_key}.csv"
            csv_bytes = item["synthetic_df"].to_csv(index=False).encode("utf-8")
            zf.writestr(csv_name, csv_bytes)
    zip_buffer.seek(0)
    st.download_button(
        label="Download All Model Datasets (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="all_synthetic_datasets.zip",
        mime="application/zip",
        use_container_width=True,
    )

    selected = st.selectbox(
        "Choose model for deep dive",
        options=list(results["model_results"].keys()),
        format_func=lambda x: MODEL_LABELS.get(x, x),
    )
    model_result = results["model_results"][selected]
    syn_df = model_result["synthetic_df"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Explorer", "Real vs Synthetic", "Quality", "Bias + Utility", "Charts + Export"]
    )
    with tab1:
        _show_data_explorer(syn_df, key_prefix=f"{selected}_syn")
    with tab2:
        _show_compare_view(results["real_df"], syn_df)
    with tab3:
        q = model_result["quality"]
        js_df = pd.DataFrame(
            [{"column": k, "js_divergence": v, "fidelity": 1 - v} for k, v in q["js_scores"].items()]
        )
        st.metric("Overall Quality", f"{q['overall'] * 100:.2f}%")
        st.dataframe(js_df.sort_values("fidelity", ascending=False), use_container_width=True)
    with tab4:
        ml = model_result["ml"]
        b = model_result["bias"]
        c1, c2, c3 = st.columns(3)
        c1.metric("TSTR Accuracy", f"{ml['tstr_accuracy'] * 100:.2f}%")
        c2.metric("TRTR Accuracy", f"{ml['trtr_accuracy'] * 100:.2f}%")
        c3.metric("Bias Reduction", f"{b['bias_reduction_pct']:.2f}%")
        st.metric("Privacy Risk", f"{model_result['privacy_score']:.2f}/100")
    with tab5:
        chart_paths = [
            ("Training Loss", config.TRAINING_LOSS_PNG),
            ("Quality Scores", config.QUALITY_SCORES_PNG),
            ("Bias Comparison", config.BIAS_COMPARISON_PNG),
        ]
        for title, path in chart_paths:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Chart not found: {path}")
        csv_data = syn_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {MODEL_LABELS.get(selected, selected)} CSV",
            data=csv_data,
            file_name=f"synthetic_patients_{selected}.csv",
            mime="text/csv",
        )


def _show_gemini_chatbot(results: Dict[str, Any]) -> None:
    st.subheader("Gemini Chatbot (gemini-2.5-flash)")
    st.caption("Ask anything about generated results, quality, privacy, bias, and model selection.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask Gemini about this run...")
    if user_msg:
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Gemini 2.5 Flash..."):
                answer = _ask_gemini(user_msg, results)
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def _collect_specs() -> Dict[str, Any]:
    st.sidebar.header("Generation Specifications")
    problem_focus = st.sidebar.selectbox(
        "Project focus",
        options=["Balanced quality/privacy", "Higher privacy", "Higher utility", "Fairness-focused"],
    )
    privacy_level = st.sidebar.selectbox(
        "Privacy strictness",
        options=["Low", "Medium", "High"],
        index=1,
    )
    selected_models = st.sidebar.multiselect(
        "Models to execute individually",
        options=["ctgan", "tvae", "gaussian"],
        default=["ctgan", "tvae", "gaussian"],
        format_func=lambda x: MODEL_LABELS.get(x, x),
    )
    target_use_case = st.sidebar.text_input(
        "Target use case",
        value="Academic demo for M.Tech guide",
    )
    must_include = st.sidebar.multiselect(
        "Important fields to prioritize",
        options=list(config.CORE_COLUMNS),
        default=["AGE", "GENDER", "RACE"],
    )
    confirm_specs = st.sidebar.checkbox("I confirm these specs before generation", value=False)
    return {
        "problem_focus": problem_focus,
        "privacy_level": privacy_level,
        "selected_models": selected_models,
        "target_use_case": target_use_case,
        "must_include": must_include,
        "confirmed": confirm_specs,
    }


def main() -> None:
    _inject_theme()
    _show_overview()

    with st.sidebar:
        st.header("Run Controls")
        n_samples = st.number_input(
            "Synthetic Samples",
            min_value=100,
            max_value=10000,
            value=int(config.N_SYNTHETIC_SAMPLES),
            step=100,
        )
        dp_epsilon = st.slider(
            "DP Epsilon",
            min_value=0.1,
            max_value=10.0,
            value=float(config.DP_EPSILON),
            step=0.1,
        )
        force_retrain = st.toggle("Force Model Retraining", value=False)

    specs = _collect_specs()
    can_run = specs["confirmed"] and len(specs["selected_models"]) > 0
    if not specs["confirmed"]:
        st.warning("Please confirm generation specifications in sidebar before running.")
    if len(specs["selected_models"]) == 0:
        st.warning("Select at least one model for individual execution.")

    run_btn = st.sidebar.button(
        "Run Full Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if run_btn:
        start = time.time()
        with st.spinner("Running model-wise pipeline... this can take a few minutes."):
            results = _run_pipeline(
                n_samples=int(n_samples),
                dp_epsilon=float(dp_epsilon),
                force_retrain=force_retrain,
                selected_models=specs["selected_models"],
                specs=specs,
            )
        elapsed = time.time() - start
        st.session_state["pipeline_results"] = results
        st.session_state["pipeline_elapsed"] = elapsed
        st.success("Pipeline completed successfully.")

    if "pipeline_results" in st.session_state:
        _show_model_wise_results(
            st.session_state["pipeline_results"],
            st.session_state.get("pipeline_elapsed", 0.0),
        )
        st.markdown("---")
        _show_gemini_chatbot(st.session_state["pipeline_results"])


if __name__ == "__main__":
    main()
