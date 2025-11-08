# I used ChatGPT sparingly for: (a) the soft-voting weight
# normalization snippet, (b) the weighted feature-importance idea for Voting,
# (c) a Pandas Styler pattern for coloring predictions, and (d) minor syntax
# reminders. =

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
st.sidebar.dataframe(template_df.head(5), use_container_width=True)

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
st.dataframe(styled, use_container_width=True)

# Tabs: Confusion Matrix / Classification Report / Feature Importance
st.subheader("Model Performance and Insights")
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

# Tab 1: Confusion Matrix image
with tab1:
    st.write("### Confusion Matrix")
    cm_file = f"{prefix}_confusion_mat.svg" if prefix in ["rf", "dt"] else f"{prefix}_confusion_mat.svg"
    # handle AdaBoost/Voting capitalization
    if model_choice == "AdaBoost":
        cm_file = "ADA_confusion_mat.svg"
    if model_choice == "Soft Voting":
        cm_file = "Voting_confusion_mat.svg"

    if exists(cm_file):
        st.image(cm_file, use_container_width=True)
    else:
        st.caption("Confusion matrix file not found. Re-run the notebook to generate it.")

# Tab 2: Classification Report CSV
with tab2:
    st.write("### Classification Report")
    cr_file = f"{prefix}_class_report.csv" if prefix in ["rf", "dt"] else f"{prefix}_class_report.csv"
    if model_choice == "AdaBoost":
        cr_file = "ADA_class_report.csv"
    if model_choice == "Soft Voting":
        cr_file = "Voting_class_report.csv"

    if exists(cr_file):
        rep_df = pd.read_csv(cr_file)
        st.dataframe(rep_df, use_container_width=True)
        st.caption("Precision, Recall, F1-Score, and Support for each class.")
    else:
        st.caption("Classification report file not found. Re-run the notebook to export it.")

# Tab 3: Feature Importance
with tab3:
    st.write("### Feature Importance")
    # prefer pre-saved SVGs when available; otherwise compute a quick barplot if model exposes importances
    fi_file = None
    if model_choice == "Random Forest" and exists("rf_feature_imp.svg"):
        fi_file = "rf_feature_imp.svg"
    elif model_choice == "Decision Tree" and exists("dt_feature_imp.svg"):
        fi_file = "dt_feature_imp.svg"
    elif model_choice == "AdaBoost" and exists("ADA_feature_imp.svg"):
        fi_file = "ADA_feature_imp.svg"
    elif model_choice == "Soft Voting" and exists("voting_feature_imp.svg"):
        fi_file = "voting_feature_imp.svg"

    if fi_file is not None:
        st.image(fi_file, use_container_width=True)
        st.caption("Features ranked by relative importance.")
    else:
        # fall back: compute simple barplot if model supports attribute
        if hasattr(clf, "feature_importances_"):
            imp = np.array(clf.feature_importances_, dtype=float)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            order = np.argsort(imp)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(np.arange(len(feature_cols)), imp[order])
            ax.set_xticks(np.arange(len(feature_cols)))
            ax.set_xticklabels(np.array(feature_cols)[order], rotation=90)
            ax.set_ylabel("Feature Importance")
            ax.set_title(f"{model_choice} — Feature Importance")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("Computed live from the loaded model.")
        else:
            st.caption("Feature importance graphic not available for this model.")