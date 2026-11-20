"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21323015
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: Claude Code (Anthropic Claude)
#   - How you used them: to explain the requirements, help complete
#     classify_text, compute_accuracy and main(), and fix local package
#     version conflicts. I reviewed and tested all code myself.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER: A bank's fraud / customer-protection analyst who screens
#   inbound SMS so customers can be warned about spam and smishing.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER: My 100-sample run gave 99% accuracy (99/100): 0 false
#   positives and 1 false negative among the 12 spam messages. This is
#   promising as an alert/flagging aid, but not enough to rely on alone,
#   because 100 samples (only 12 spam) is small and the one miss was a spam.
#   A false positive (blocking a bank or one-time-code SMS) breaks customer
#   service, while a false negative can let a scam cause real money loss,
#   so the app should flag messages for human review, not auto-block.
#
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21323015"
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    # The pipeline returns a list with one dict, e.g. [{"label": "LABEL_1", "score": 0.98}]
    result = classifier(text)[0]
    display_label = map_label(result["label"])
    score = float(result["score"])
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    if len(true_labels) == 0:
        return 0.0
    matches = 0
    for pred, truth in zip(pred_labels, true_labels):
        if pred == truth:
            matches += 1
    accuracy = matches / len(true_labels)
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.markdown(f"**Student ID:** {STUDENT_ID}")
    st.markdown(f"**Course secret:** `{COURSE_SECRET}`")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    try:
        if "HF_TOKEN" in st.secrets:
            hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        # No secrets.toml (e.g., running locally) — the public model still loads.
        hf_token = None

    with st.spinner("Loading model..."):
        classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if alert_text.strip() == "":
            st.warning("Please paste an SMS message first.")
        else:
            label, score = classify_text(classifier, alert_text)
            if label == "spam":
                st.error(f"Prediction: **{label}** (confidence {score:.2%})")
            else:
                st.success(f"Prediction: **{label}** (confidence {score:.2%})")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        with st.spinner("Loading dataset and classifying 100 messages..."):
            texts, true_labels = load_test_samples()
            pred_labels = []
            rows = []
            for i, (text, truth) in enumerate(zip(texts, true_labels), start=1):
                pred, score = classify_text(classifier, text)
                pred_labels.append(pred)
                short = text.strip().replace("\n", " ")
                if len(short) > 60:
                    short = short[:60] + "..."
                rows.append(
                    {
                        "#": i,
                        "Text (short)": short,
                        "Ground truth": truth,
                        "Prediction": pred,
                        "Confidence": round(score, 4),
                        "Match": "✅" if pred == truth else "❌",
                    }
                )
            accuracy = compute_accuracy(pred_labels, true_labels)

        n_correct = sum(1 for r in rows if r["Match"] == "✅")
        st.metric("Accuracy on 100 samples", f"{accuracy:.2%}")
        st.write(f"Correct: {n_correct} / {len(rows)}")
        st.dataframe(rows, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
