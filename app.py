import streamlit as st
import pandas as pd
import pickle

# ---------------- Load Model ----------------
with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

# ---------------- Teams & Cities ----------------
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Mumbai', 'Chennai', 'Delhi', 'Bangalore',
    'Hyderabad', 'Kolkata', 'Jaipur', 'Mohali'
]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="IPL Win Probability Predictor", page_icon="🏏")
st.title("🏏 IPL Win Probability Predictor")

batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)

if batting_team == bowling_team:
    st.error("Batting and Bowling teams must be different")
    st.stop()

city = st.selectbox("Select City", cities)

target = st.number_input("Target Score", min_value=1, step=1)
score = st.number_input("Current Score", min_value=0, step=1)
overs = st.number_input(
    "Overs Completed (e.g., 10.3)",
    min_value=0.1,
    max_value=20.0,
    step=0.1
)
wickets = st.number_input(
    "Wickets Lost",
    min_value=0,
    max_value=10,
    step=1
)

# ---------------- Prediction ----------------
if st.button("Predict Win Probability"):

    # Convert overs to balls
    over_int = int(overs)
    balls = int(round((overs - over_int) * 10))

    if balls > 5:
        st.error("Invalid over format. Use format like 10.2 or 15.4")
        st.stop()

    balls_bowled = over_int * 6 + balls
    bowls_left = 120 - balls_bowled
    runs_left = target - score

    if bowls_left <= 0 or runs_left < 0:
        st.error("Invalid match situation")
        st.stop()

    # Rates
    crr = score / overs
    rrr = runs_left / (bowls_left / 6)

    # ---------------- Input DataFrame ----------------
    # MUST match training columns EXACTLY
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'bowls_left': [bowls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # ---------------- Prediction ----------------
    result = pipe.predict_proba(input_df)

    st.success(
        f"🏏 {batting_team} Win Probability: "
        f"**{round(result[0][1] * 100, 2)}%**"
    )

    st.error(
        f"🎯 {bowling_team} Win Probability: "
        f"**{round(result[0][0] * 100, 2)}%**"
    )
