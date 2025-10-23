import streamlit as st
import pandas as pd
import joblib

from feature_engineering import FeatureBuilder  # noqa: F401 keeps class available for joblib

# Set page config
st.set_page_config(page_title="Pokemon Legendary Predictor", layout="wide")


@st.cache_resource
def load_model():
    artifacts = joblib.load("pokemon_legendary_model.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    feature_builder = joblib.load("feature_builder.joblib")
    model = artifacts["model"]
    threshold = artifacts.get("classification_threshold", 0.5)
    metrics = artifacts.get("metrics", {})
    feature_importance = artifacts.get("feature_importance", {})
    return (
        model,
        feature_columns,
        feature_builder,
        threshold,
        metrics,
        feature_importance,
    )


@st.cache_data
def load_pokemon_data():
    return pd.read_csv("./pokemon_data.csv")


(
    model,
    feature_columns,
    feature_builder,
    prediction_threshold,
    model_metrics,
    feature_importance,
) = load_model()
pokemon_df = load_pokemon_data()

FEATURE_SNAPSHOT_COLUMNS = [
    "ability_mean_legendary_rate",
    "ability_max_legendary_rate",
    "type_combo_legendary_rate",
    "generation_legendary_rate",
    "base_experience",
    "stat_mean",
    "base_total",
]


def format_float(value, precision=3):
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    engineered = feature_builder.transform(raw_df)
    return engineered[feature_columns]


def predict_from_raw(raw_df: pd.DataFrame):
    features_df = prepare_features(raw_df)
    probabilities = model.predict_proba(features_df)[0]
    prediction = int(probabilities[1] >= prediction_threshold)
    return prediction, probabilities, features_df


def display_prediction(prediction, probability, threshold):
    if prediction == 1:
        st.success("🌟 This Pokemon is LEGENDARY!")
    else:
        st.info("⚪ This Pokemon is NOT Legendary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non-Legendary Probability", f"{probability[0]:.2%}")
    with col2:
        st.metric("Legendary Probability", f"{probability[1]:.2%}")
    st.caption(f"Decision threshold: {threshold:.2f}")


def show_feature_snapshot(feature_frame: pd.DataFrame):
    available_cols = [
        col for col in FEATURE_SNAPSHOT_COLUMNS if col in feature_frame.columns
    ]
    if not available_cols:
        st.info("No engineered feature snapshot available.")
        return
    snapshot = feature_frame.iloc[0][available_cols].rename("value")
    st.dataframe(snapshot.to_frame())


type_options = sorted(
    {
        t.strip().lower()
        for types_str in pokemon_df["types"].unique()
        for t in types_str.split(",")
    }
)

st.title("🔮 Unown - Pokemon Legendary Status Predictor")
st.markdown("---")

with st.sidebar:
    st.header("Model Snapshot")
    st.metric("Decision Threshold", format_float(prediction_threshold, 2))
    st.metric("ROC AUC", format_float(model_metrics.get("auc"), 3))
    st.metric("F1 (optimized)", format_float(model_metrics.get("f1_optimized"), 2))
    st.metric(
        "Recall (optimized)", format_float(model_metrics.get("recall_optimized"), 2)
    )
    if feature_importance:
        top_features = (
            pd.Series(feature_importance)
            .sort_values(ascending=False)
            .head(5)
            .rename_axis("Feature")
            .reset_index(name="Importance")
        )
        st.caption("Top model features")
        st.table(top_features)
    else:
        st.caption("Feature importances unavailable.")

tab1, tab2 = st.tabs(["Predict Existing Pokemon", "Create New Pokemon"])

with tab1:
    st.header("Select an Existing Pokemon")
    pokemon_names = sorted(pokemon_df["name"].unique())
    selected_pokemon = st.selectbox(
        "Choose a Pokemon:", pokemon_names, key="pokemon_select"
    )

    pokemon_row = pokemon_df[pokemon_df["name"] == selected_pokemon].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pokemon Information")
        st.write(f"**Name:** {pokemon_row['name']}")
        st.write(f"**Generation:** {pokemon_row['generation']}")
        st.write(f"**Height:** {pokemon_row['height']} m")
        st.write(f"**Weight:** {pokemon_row['weight']} kg")
        st.write(f"**Types:** {pokemon_row['types']}")
        st.write(f"**Base Experience:** {pokemon_row['base_experience']}")

    with col2:
        st.subheader("Stats")
        stats_data = {
            "HP": pokemon_row["hp"],
            "Attack": pokemon_row["attack"],
            "Defense": pokemon_row["defense"],
            "Sp. Atk": pokemon_row["sp_atk"],
            "Sp. Def": pokemon_row["sp_def"],
            "Speed": pokemon_row["speed"],
        }
        st.bar_chart(pd.Series(stats_data))

    raw_df = pokemon_row.to_frame().T
    prediction, probability, engineered_features = predict_from_raw(raw_df)

    st.markdown("---")
    st.subheader("Prediction Result")
    display_prediction(prediction, probability, prediction_threshold)

    with st.expander("🧩 Engineered Feature Snapshot"):
        show_feature_snapshot(engineered_features)


with tab2:
    st.header("Create and Predict New Pokemon")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Pokemon Name", value="New Pokemon")
        generation = st.number_input("Generation", min_value=1, max_value=9, value=1)
        primary_type = str(
            st.selectbox(
                "Primary Type",
                type_options,
                format_func=lambda x: x.title(),
                key="primary_type",
            )
        )
        secondary_type = str(
            st.selectbox(
                "Secondary Type",
                ["none"] + type_options,
                format_func=lambda x: "None" if x == "none" else x.title(),
                key="secondary_type",
            )
        )
        abilities_input = st.text_input(
            "Abilities (separate with ';')",
            value="pressure",
            help="Example: pressure; telepathy",
        )
        height = st.number_input("Height (m)", min_value=0.0, value=1.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=0.0, value=10.0, step=0.1)
        base_experience = st.number_input(
            "Base Experience", min_value=0, value=50, step=5
        )

    with col2:
        hp = st.number_input("HP", min_value=1, value=50, step=1)
        attack = st.number_input("Attack", min_value=1, value=50, step=1)
        defense = st.number_input("Defense", min_value=1, value=50, step=1)
        sp_atk = st.number_input("Special Attack", min_value=1, value=50, step=1)
        sp_def = st.number_input("Special Defense", min_value=1, value=50, step=1)
        speed = st.number_input("Speed", min_value=1, value=50, step=1)

    base_total = hp + attack + defense + sp_atk + sp_def + speed
    physical_total = attack + defense + hp
    special_total = sp_atk + sp_def + hp

    st.markdown("---")
    st.subheader("Calculated Stats")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Base Total", int(base_total))
    with col_b:
        st.metric("Physical Total", int(physical_total))
    with col_c:
        st.metric("Special Total", int(special_total))

    if st.button("🔮 Predict Legendary Status", key="predict_button"):
        type_components: list[str] = [primary_type]
        if secondary_type != "none":
            type_components.append(secondary_type)
        types_value = ", ".join(type_components)

        input_record = {
            "name": name,
            "generation": int(generation),
            "types": types_value,
            "abilities": abilities_input.strip(),
            "hp": int(hp),
            "attack": int(attack),
            "defense": int(defense),
            "sp_atk": int(sp_atk),
            "sp_def": int(sp_def),
            "speed": int(speed),
            "base_total": int(base_total),
            "height": float(height),
            "weight": float(weight),
            "base_experience": int(base_experience),
            "is_legendary": False,
        }

        raw_df = pd.DataFrame([input_record])
        prediction, probability, engineered_features = predict_from_raw(raw_df)

        st.markdown("---")
        st.subheader("Prediction Result")
        display_prediction(prediction, probability, prediction_threshold)

        with st.expander("🧩 Engineered Feature Snapshot"):
            show_feature_snapshot(engineered_features)

        with st.expander("📊 Why this prediction?"):
            st.write(
                """
                The Random Forest focuses on engineered signals such as:

                - **Ability legendary rate** — how often the supplied abilities appear on legendary Pokémon.
                - **Base experience** (and its scaled variants).
                - **Type combination legendary rate**.
                - **Overall stat mean/base total** as proxies for combat strength.
                """
            )

st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p><small>Unown | Built with Streamlit & Machine Learning</small></p>
</div>
""",
    unsafe_allow_html=True,
)
