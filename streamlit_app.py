import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Pokemon Legendary Predictor", layout="wide")


# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("pokemon_legendary_model.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return model, feature_columns


@st.cache_data
def load_pokemon_data():
    df = pd.read_csv("./pokemon_data.csv")
    return df


# Load data
model, feature_columns = load_model()
pokemon_df = load_pokemon_data()


# Helper function to display prediction results
def display_prediction(prediction, probability):
    """Display prediction results in a consistent format."""
    if prediction == 1:
        st.success("🌟 This Pokemon is LEGENDARY!")
    else:
        st.info("⚪ This Pokemon is NOT Legendary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non-Legendary Probability", f"{probability[0]:.2%}")
    with col2:
        st.metric("Legendary Probability", f"{probability[1]:.2%}")


st.title("🔮 Unown - Pokemon Legendary Status Predictor")
st.markdown("---")

# Create tabs for two different prediction modes
tab1, tab2 = st.tabs(["Predict Existing Pokemon", "Create New Pokemon"])

with tab1:
    st.header("Select an Existing Pokemon")

    # Get unique pokemon names
    pokemon_names = sorted(pokemon_df["name"].unique())
    selected_pokemon = st.selectbox(
        "Choose a Pokemon:", pokemon_names, key="pokemon_select"
    )

    # Get selected pokemon data
    pokemon_data = pokemon_df[pokemon_df["name"] == selected_pokemon].iloc[0]

    # Display pokemon info
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pokemon Information")
        st.write(f"**Name:** {pokemon_data['name']}")
        st.write(f"**Generation:** {pokemon_data['generation']}")
        st.write(f"**Height:** {pokemon_data['height']} m")
        st.write(f"**Weight:** {pokemon_data['weight']} kg")
        st.write(f"**Types:** {pokemon_data['types']}")
        st.write(f"**Base Experience:** {pokemon_data['base_experience']}")

    with col2:
        st.subheader("Stats")
        stats_data = {
            "HP": pokemon_data["hp"],
            "Attack": pokemon_data["attack"],
            "Defense": pokemon_data["defense"],
            "Sp. Atk": pokemon_data["sp_atk"],
            "Sp. Def": pokemon_data["sp_def"],
            "Speed": pokemon_data["speed"],
        }
        st.bar_chart(pd.Series(stats_data))

    # Prepare features for prediction
    features = []
    for col in feature_columns:
        if col in pokemon_data.index:
            features.append(pokemon_data[col])
        else:
            # For engineered features, calculate them
            if col == "attack_defense_ratio":
                features.append(pokemon_data["attack"] / (pokemon_data["defense"] + 1))
            elif col == "attack_speed_ratio":
                features.append(pokemon_data["attack"] / (pokemon_data["speed"] + 1))
            elif col == "special_attack_ratio":
                features.append(pokemon_data["sp_atk"] / (pokemon_data["sp_def"] + 1))
            elif col == "physical_total":
                features.append(
                    pokemon_data["attack"]
                    + pokemon_data["defense"]
                    + pokemon_data["hp"]
                )
            elif col == "special_total":
                features.append(
                    pokemon_data["sp_atk"] + pokemon_data["sp_def"] + pokemon_data["hp"]
                )

    # Make prediction with feature names
    features_df = pd.DataFrame([features], columns=feature_columns)
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")
    display_prediction(prediction, probability)


with tab2:
    st.header("Create and Predict New Pokemon")

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Pokemon Name", value="New Pokemon")
        generation = st.number_input("Generation", min_value=1, max_value=9, value=1)
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

    # Calculate engineered features
    attack_defense_ratio = attack / (defense + 1)
    attack_speed_ratio = attack / (speed + 1)
    special_attack_ratio = sp_atk / (sp_def + 1)
    physical_total = attack + defense + hp
    special_total = sp_atk + sp_def + hp
    base_total = hp + attack + defense + sp_atk + sp_def + speed

    # Display calculated stats
    st.markdown("---")
    st.subheader("Calculated Stats")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base Total", int(base_total))
    with col2:
        st.metric("Physical Total", int(physical_total))
    with col3:
        st.metric("Special Total", int(special_total))

    # Prepare features for prediction
    features = [
        generation,
        hp,
        attack,
        defense,
        sp_atk,
        sp_def,
        speed,
        base_total,
        height,
        weight,
        base_experience,
        attack_defense_ratio,
        attack_speed_ratio,
        special_attack_ratio,
        physical_total,
        special_total,
    ]

    # Make prediction with feature names
    if st.button("🔮 Predict Legendary Status", key="predict_button"):
        features_df = pd.DataFrame([features], columns=feature_columns)
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        display_prediction(prediction, probability)

        # Show feature importance context
        with st.expander("📊 Why this prediction?"):
            st.write("""
            The most important features for determining legendary status are:
            1. **Base Total** - Overall stat total
            2. **Attack** - Physical attack power
            3. **Defense** - Physical defense power
            4. **Special Attack** - Special attack power
            5. **Speed** - Speed stat
            
            Legendary Pokemon typically have significantly higher base stats compared to non-legendary Pokemon.
            """)

st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p><small>Unown | Built with Streamlit & Machine Learning</small></p>
</div>
""",
    unsafe_allow_html=True,
)
