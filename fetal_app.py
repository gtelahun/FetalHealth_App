# I used ChatGPT sparingly for: (a) the soft-voting weight
# normalization snippet, (b) the weighted feature-importance idea for Voting,
# (c) a Pandas Styler pattern for coloring predictions, and (d) minor syntax
# reminders.

# Standard imports used in class
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib.pyplot as plt

def exists(path: str) -> bool:
    return Path(path).exists()

# Page header
st.set_page_config(page_title="Fetal Health Classification", layout="wide")

st.title("Fetal Health Classification: A Machine Learning App")
if exists("fetal_health_image.gif"):
    st.image("fetal_health_image.gif", width=600)
st.markdown(
    "<p style='color:#808080;font-size:12px;margin-top:-8px'>Utilize the ML app to predict fetal health classifications.</p>",
    unsafe_allow_html=True,
)

# Load trained models (from notebook pickles)
def load_pickle(fname, alias):
    if not exists(fname):
        st.warning(f"{alias} model file not found: {fname}")
        return None
    with open(fname, "rb") as f:
        return pickle.load(f)

clf_dt  = load_pickle("dt_fetal.pickle", "Decision Tree")
clf_rf  = load_pickle("rf_fetal.pickle", "Random Forest")
clf_ada = load_pickle("ADA_fetal.pickle", "AdaBoost")
clf_vot = load_pickle("Voting_fetal.pickle", "Soft Voting")

# Load default dataset to show sample
if not exists("fetal_health.csv"):
    st.stop()

default_df = pd.read_csv("fetal_health.csv").dropna().reset_index(drop=True)
feature_cols = [c for c in default_df.columns if c != "fetal_health"]
template_df = default_df[feature_cols]

# Sidebar inputs
st.sidebar.header("Fetal Health Features Input")
uploaded = st.sidebar.file_uploader("Upload a fetal health CSV", type=["csv"])
st.sidebar.warning("Ensure your data strictly follows the format outlined below")
st.sidebar.dataframe(template_df.head(5), use_column_width=True)

model_choice = st.sidebar.radio(
    "Choose model for Prediction",
    ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"],
)
st.sidebar.info(f"You selected: {model_choice}")


if uploaded is None:
    st.info("Please upload data to proceed.")
    st.stop()

# Read user data and align column order
user_df = pd.read_csv(uploaded)
# enforce same columns + order
user_df = user_df[feature_cols]

# choose model
model_map = {
    "Random Forest": (clf_rf, "rf"),
    "Decision Tree": (clf_dt, "dt"),
    "AdaBoost": ("ada", "ADA"),  # to keep lowercase/uppercase file prefixes aligned
    "Soft Voting": (clf_vot, "Voting"),
}
clf, prefix = model_map[model_choice]
if model_choice == "AdaBoost":
    clf = clf_ada

if clf is None:
    st.error("Selected model is not available. Make sure you trained and saved pickles in the notebook.")
    st.stop()

# predict
pred = clf.predict(user_df)
proba = clf.predict_proba(user_df)
# find the probability for the predicted class row-wise
class_to_index = {int(c): i for i, c in enumerate(clf.classes_)}
row_probs = []
for i, pcls in enumerate(pred):
    idx = class_to_index[int(pcls)]
    row_probs.append(proba[i, idx])

# format output
out = user_df.copy()
out["Predicted Fetal Health"] = pred
out["Prediction Probability (%)"] = np.round(np.array(row_probs) * 100, 2)

# label mapping + color styling (course colors: lime/yellow/orange tones)
label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
out["Predicted Fetal Health"] = out["Predicted Fetal Health"].map(label_map)

def color_pred(val):
    cmap = {
        "Normal": "background-color: #b6fcb6",      # light green (lime-ish)
        "Suspect": "background-color: #fff8b6",     # light yellow
        "Pathological": "background-color: #ffcc99" # soft orange
    }
    return cmap.get(val, "")

styled = out.style.applymap(color_pred, subset=["Predicted Fetal Health"])

# Display predictions
st.subheader(f"Predicting Fetal Health Using {model_choice} Model")
st.dataframe(styled, use_column_width=True)

# Tabs: Confusion Matrix / Classification Report / Feature Importance
import os

st.subheader("Model Performance and Insights")
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

def file_exists(p): 
    return os.path.exists(p)

# --- Tab 1: Confusion Matrix ---
with tab1:
    st.write("### Confusion Matrix")
    if model_choice == "Random Forest":
        cm_file = "rf_confusion_mat.svg"
    elif model_choice == "Decision Tree":
        cm_file = "dt_confusion_mat.svg"
    elif model_choice == "AdaBoost":
        cm_file = "ADA_confusion_mat.svg"   # NOTE: uppercase ADA in your repo
    else:  # Soft Voting
        cm_file = "Voting_confusion_mat.svg"

    if file_exists(cm_file):
        st.image(cm_file, use_column_width=True)
    else:
        st.warning(f"Confusion matrix image not found: {cm_file}. Re-run the notebook to export it.")

# --- Tab 2: Classification Report ---
with tab2:
    st.write("### Classification Report")
    if model_choice == "Random Forest":
        cr_file = "rf_class_report.csv"
    elif model_choice == "Decision Tree":
        cr_file = "dt_class_report.csv"
    elif model_choice == "AdaBoost":
        cr_file = "ADA_class_report.csv"
    else:
        cr_file = "Voting_class_report.csv"

    if file_exists(cr_file):
        rep_df = pd.read_csv(cr_file)
        st.dataframe(rep_df, use_column_width=True)
        st.caption("Precision, Recall, F1-Score, and Support for each class.")
    else:
        st.warning(f"Classification report file not found: {cr_file}")

# --- Tab 3: Feature Importance ---
with tab3:
    st.write("### Feature Importance")
    if model_choice == "Random Forest":
        fi_file = "rf_feature_imp.svg"
    elif model_choice == "Decision Tree":
        fi_file = "dt_feature_imp.svg"
    elif model_choice == "AdaBoost":
        fi_file = "ADA_feature_imp.svg"     # NOTE: uppercase ADA in your repo
    else:
        fi_file = "voting_feature_imp.svg"

    if file_exists(fi_file):
        st.image(fi_file, use_container_width=True)
        st.caption("Features ranked by relative importance.")
    else:
        # graceful fallback: show a live barplot if available
        if hasattr(clf, "feature_importances_"):
            imp = np.array(clf.feature_importances_, dtype=float)
            s = imp.sum()
            if s > 0: imp = imp / s
            order = np.argsort(imp)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(np.arange(len(feature_cols)), imp[order])
            ax.set_xticks(np.arange(len(feature_cols)))
            ax.set_xticklabels(np.array(feature_cols)[order], rotation=90)
            ax.set_ylabel("Feature Importance")
            ax.set_title(f"{model_choice} — Feature Importance")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"Feature importance image not found: {fi_file}")
