"""Doctor vs AI — a CT-scan guessing game powered by a fine-tuned ResNet18.

    streamlit run app.py
"""
import json
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from common import ART, CLASSES, build_model, load_split, to_tensor

st.set_page_config(page_title="Doctor vs AI", layout="wide")
ROUNDS = 10
LEADERBOARD = ART / "leaderboard.json"


# ---------- model + data ----------
@st.cache_resource
def get_model():
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(ART / "model.pt", map_location="cpu"))
    return m.eval()


@st.cache_data
def get_test():
    return load_split("test")


@st.cache_data
def get_metrics():
    return json.load(open(ART / "metrics.json")), json.load(open(ART / "history.json"))


def predict(img_u8):
    with torch.no_grad():
        return get_model()(to_tensor(img_u8[None])).softmax(1)[0].numpy()


def grad_cam(img_u8):
    """Grad-CAM on the last ResNet block: which pixels pushed the top prediction up?"""
    model, store = get_model(), {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: store.update(act=o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    out = model(to_tensor(img_u8[None]))
    cls = out.argmax(1).item()
    model.zero_grad()
    out[0, cls].backward()
    h1.remove(); h2.remove()
    weights = store["grad"].mean((2, 3), keepdim=True)            # importance of each channel
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))    # weighted sum of feature maps
    cam = F.interpolate(cam, size=img_u8.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), cls


def big(img_u8, size=320):
    return Image.fromarray(img_u8).resize((size, size), Image.LANCZOS)


def overlay(img_u8, cam, alpha=0.45, size=320):
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.stack([img_u8] * 3, -1)
    mix = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(mix).resize((size, size), Image.LANCZOS)


def label(c):
    return c


def prob_chart(probs, k=5):
    top = np.argsort(probs)[::-1][:k]
    df = pd.DataFrame({"organ": [label(CLASSES[i]) for i in top], "confidence": probs[top]})
    st.bar_chart(df, x="organ", y="confidence", horizontal=True, height=220)


def examples_per_class():
    x, y = get_test()
    rng = np.random.default_rng(7)
    return {c: x[rng.choice(np.where(y == i)[0])] for i, c in enumerate(CLASSES)}


def study_guide():
    with st.expander("Study guide: one example of every organ"):
        st.caption("CT images use the radiology convention: the patient's **left** "
                   "appears on the **right** side of the image, as if you face the patient.")
        ex = examples_per_class()
        cols = st.columns(6)
        for k, c in enumerate(CLASSES):
            cols[k % 6].image(big(ex[c], 120), caption=label(c))


# ---------- page: play ----------
def new_game(difficulty):
    x, y = get_test()
    st.session_state.game = dict(
        difficulty=difficulty, round=1, you=0, ai=0, streak=0, best_streak=0,
        idx=random.sample(range(len(y)), ROUNDS), answered=None, log=[], options=None,
    )


def options_for(true_idx, difficulty):
    if difficulty == "Hard (all 11 organs)":
        return list(range(len(CLASSES)))
    others = random.sample([i for i in range(len(CLASSES)) if i != true_idx], 3)
    opts = others + [true_idx]
    random.shuffle(opts)
    return opts


def page_play():
    st.title("Doctor vs AI")
    st.write("You and a deep-learning model look at the same abdominal CT slice. "
             "Name the organ. **+10** for a correct answer, **+5 bonus** for every 3-in-a-row streak. "
             f"{ROUNDS} rounds. Can you beat the machine?")

    if "game" not in st.session_state:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("Your name", value=st.session_state.get("name", ""), placeholder="Dr. ...")
        diff = c2.radio("Difficulty", ["Easy (4 choices)", "Hard (all 11 organs)"])
        study_guide()
        if st.button("Start game", type="primary", disabled=not name.strip()):
            st.session_state.name = name.strip()
            new_game(diff)
            st.rerun()
        return

    g = st.session_state.game
    x, y = get_test()

    if g["round"] > ROUNDS:
        return game_over(g)

    i = g["idx"][g["round"] - 1]
    img, truth = x[i], int(y[i])
    if g["options"] is None:
        g["options"] = options_for(truth, g["difficulty"])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Round", f"{g['round']} / {ROUNDS}")
    s2.metric(f"{st.session_state.name}", g["you"])
    s3.metric("AI", g["ai"])
    s4.metric("Streak", g["streak"])
    st.progress((g["round"] - 1) / ROUNDS)

    left, right = st.columns([1, 1.3])
    left.image(big(img), caption=f"Scan #{i} (64×64 CT slice, upscaled)")

    with right:
        if g["answered"] is None:
            st.subheader("Which organ is this?")
            cols = st.columns(2 if len(g["options"]) <= 4 else 3)
            for k, o in enumerate(g["options"]):
                if cols[k % len(cols)].button(label(CLASSES[o]), key=f"opt{o}", use_container_width=True):
                    resolve_round(g, o, truth, img, i)
                    st.rerun()
            study_guide()
        else:
            a = g["answered"]
            you_ok, ai_ok = a["you"] == truth, a["ai"] == truth
            st.subheader(f"Answer: {label(CLASSES[truth])}")
            (st.success if you_ok else st.error)(
                f"You said **{CLASSES[a['you']]}**: {'correct! +10' if you_ok else 'wrong'}"
                + (" Streak bonus +5!" if a["bonus"] else ""))
            (st.success if ai_ok else st.error)(
                f"AI said **{CLASSES[a['ai']]}** ({a['conf']:.0%} sure): {'correct +10' if ai_ok else 'wrong!'}")
            st.caption("AI's top guesses")
            prob_chart(a["probs"])
            if st.button("Next round" if g["round"] < ROUNDS else "See results", type="primary"):
                g["round"] += 1
                g["answered"], g["options"] = None, None
                st.rerun()


def resolve_round(g, choice, truth, img, i):
    probs = predict(img)
    ai = int(probs.argmax())
    bonus = False
    if choice == truth:
        g["you"] += 10
        g["streak"] += 1
        g["best_streak"] = max(g["best_streak"], g["streak"])
        if g["streak"] % 3 == 0:
            g["you"] += 5
            bonus = True
    else:
        g["streak"] = 0
    if ai == truth:
        g["ai"] += 10
    g["answered"] = dict(you=choice, ai=ai, conf=float(probs[ai]), probs=probs, bonus=bonus)
    g["log"].append(dict(scan=i, truth=CLASSES[truth], you=CLASSES[choice], ai=CLASSES[ai],
                         you_ok=choice == truth, ai_ok=ai == truth))


def game_over(g):
    name = st.session_state.name
    if g["you"] > g["ai"]:
        st.balloons()
        st.header(f"{name} beats the AI {g['you']} – {g['ai']}!")
    elif g["you"] == g["ai"]:
        st.header(f"Draw! {g['you']} – {g['ai']}")
    else:
        st.header(f"The AI wins {g['ai']} – {g['you']}. Keep studying, {name}!")

    if not g.get("saved"):
        board = json.load(open(LEADERBOARD)) if LEADERBOARD.exists() else []
        board.append(dict(name=name, score=g["you"], ai=g["ai"], mode=g["difficulty"].split()[0],
                          best_streak=g["best_streak"]))
        json.dump(board, open(LEADERBOARD, "w"), indent=2)
        g["saved"] = True

    log = pd.DataFrame(g["log"])
    c1, c2 = st.columns(2)
    c1.metric("Your accuracy", f"{log.you_ok.mean():.0%}")
    c2.metric("AI accuracy", f"{log.ai_ok.mean():.0%}")
    st.dataframe(log.replace({True: "Correct", False: "Wrong"}), hide_index=True, use_container_width=True)

    st.subheader("Leaderboard")
    board = pd.DataFrame(json.load(open(LEADERBOARD))).sort_values("score", ascending=False)
    st.dataframe(board.head(10), hide_index=True, use_container_width=True)

    if st.button("Play again", type="primary"):
        del st.session_state.game
        st.rerun()


# ---------- page: grad-cam ----------
def page_vision():
    st.title("Where is the AI looking?")
    st.write("**Grad-CAM** takes the gradient of the predicted class with respect to the last convolutional "
             "feature maps and highlights the regions that most influenced the decision. Red = important.")
    x, y = get_test()
    c1, c2 = st.columns([1, 3])
    organ = c1.selectbox("Organ", ["any"] + CLASSES)
    if c1.button("Random scan") or "cam_idx" not in st.session_state:
        pool = np.arange(len(y)) if organ == "any" else np.where(y == CLASSES.index(organ))[0]
        st.session_state.cam_idx = int(random.choice(pool))
    alpha = c1.slider("Heatmap strength", 0.0, 0.9, 0.45)
    i = st.session_state.cam_idx
    cam, cls = grad_cam(x[i])
    with c2:
        a, b = st.columns(2)
        a.image(big(x[i], 300), caption=f"Truth: {label(CLASSES[int(y[i])])}")
        b.image(overlay(x[i], cam, alpha, 300), caption=f"AI predicts: {label(CLASSES[cls])}")
    ok = cls == int(y[i])
    (st.success if ok else st.warning)(
        "Correct, and the heatmap shows which region convinced it." if ok
        else "Wrong prediction. The heatmap shows what confused it.")


# ---------- page: upload ----------
def prepare_upload(file):
    """Any image -> 64x64 grayscale uint8 (centre crop to square, then resize)."""
    im = Image.open(file)
    rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255
    saturation = float((rgb.max(-1) - rgb.min(-1)).mean())  # ~0 for grayscale scans
    w, h = im.size
    s = min(w, h)
    im = im.convert("L").crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(im.resize((64, 64), Image.LANCZOS), dtype=np.uint8), saturation, (w, h)


def sample_png(i):
    import io
    buf = io.BytesIO()
    big(get_test()[0][i], 256).save(buf, format="PNG")
    return buf.getvalue()


def page_upload():
    st.title("Upload & Predict")
    st.write("Upload a CT image and the model will tell you which organ it shows. It works best on an "
             "**axial abdominal CT slice cropped around one organ**, the same kind of image it was trained on.")

    x, y = get_test()
    with st.expander("No CT image handy? Download a sample from the test set"):
        cols = st.columns(4)
        for k, c in enumerate(["liver", "kidney-left", "heart", "femur-right"]):
            i = int(np.where(y == CLASSES.index(c))[0][0])
            cols[k].image(big(x[i], 120), caption=label(c))
            cols[k].download_button("Download", sample_png(i), file_name=f"sample_{c}.png",
                                    mime="image/png", key=f"dl{c}")

    file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
    if not file:
        return
    try:
        img, saturation, (w, h) = prepare_upload(file)
    except Exception:
        st.error("Couldn't read that file as an image.")
        return

    probs = predict(img)
    cam, cls = grad_cam(img)
    conf = float(probs[cls])

    a, b, c = st.columns(3)
    a.image(Image.open(file), caption=f"Uploaded ({w}×{h})", use_container_width=True)
    b.image(big(img, 300), caption="What the model sees (64×64 grayscale)")
    c.image(overlay(img, cam, 0.45, 300), caption="Grad-CAM: where it looked")

    if conf < 0.4:
        st.error(f"Sorry, I don't know this. My best guess was only {conf:.0%} confident, so I won't name an "
                 "organ. Try an axial abdominal CT slice cropped around a single organ.")
        with st.expander("Show my uncertain guesses anyway"):
            prob_chart(probs)
        return

    if saturation > 0.08:
        st.warning("This looks like a **colour photo**, not a CT scan. The model only knows 11 organs in CT, "
                   "so it is forced to pick one of them. Treat the answer below as a guess, not a diagnosis.")
    elif conf < 0.6:
        st.warning(f"Low confidence ({conf:.0%}). The image may not be an abdominal CT slice, or it may "
                   "contain more than one organ.")

    st.subheader(f"Prediction: {label(CLASSES[cls])}  ({conf:.0%} confident)")
    prob_chart(probs)
    st.caption("This is a course project, not a medical device. Don't use it for real diagnosis.")


# ---------- page: report card ----------
def page_report():
    st.title("AI Report Card")
    m, hist = get_metrics()
    c = st.columns(4)
    c[0].metric("Test accuracy", f"{m['test_acc']:.1%}")
    c[1].metric("Macro F1", f"{m['macro_f1']:.3f}")
    c[2].metric("Test images", f"{sum(v['support'] for v in m['per_class'].values()):,}")
    b = m["amp_benchmark_ms_per_step"]
    c[3].metric("Mixed-precision speed-up", f"{b['fp32'] / b['amp']:.2f}×",
                help=f"FP32 {b['fp32']} ms/step vs AMP {b['amp']} ms/step")

    h = pd.DataFrame(hist).set_index("epoch")
    l, r = st.columns(2)
    l.subheader("Loss per epoch")
    l.line_chart(h[["train_loss", "val_loss"]])
    r.subheader("Accuracy per epoch")
    r.line_chart(h[["train_acc", "val_acc"]])

    l, r = st.columns([1.2, 1])
    with l:
        st.subheader("Confusion matrix (test set)")
        cm = np.array(m["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(norm, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for a in range(len(CLASSES)):
            for b_ in range(len(CLASSES)):
                if cm[a, b_]:
                    ax.text(b_, a, cm[a, b_], ha="center", va="center", fontsize=7,
                            color="white" if norm[a, b_] > 0.5 else "black")
        fig.tight_layout()
        st.pyplot(fig)
    with r:
        st.subheader("Per-organ scores")
        df = pd.DataFrame(m["per_class"]).T
        df.index = [label(c) for c in df.index]
        df["support"] = df["support"].astype(int)
        st.dataframe(df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                     use_container_width=True)


# ---------- page: how it works ----------
def page_how():
    m, _ = get_metrics()
    cfg = m["config"]
    n_params = sum(p.numel() for p in get_model().parameters())
    hardware = "NVIDIA T4 GPU (Google Colab)" if cfg["device"] == "cuda" else cfg["device"]
    b = m["amp_benchmark_ms_per_step"]

    st.title("How it was built")

    st.subheader("Data")
    st.write("We used OrganAMNIST, one of the datasets in MedMNIST v2 (Yang et al., 2023). The images are axial "
             "slices from abdominal CT scans in the LiTS dataset, each cropped around one of 11 organs. We chose "
             "the 64×64 version because it trains quickly on Colab and still keeps enough detail. We kept the "
             "official split: 34,561 images for training, 6,491 for validation and 17,778 for testing.")

    st.subheader("Model")
    st.write("We used ResNet18 (He et al., 2016) with ImageNet-pretrained weights and fine-tuned the whole network "
             "on the CT images. We also tried training from scratch, but in the same number of epochs it did "
             "noticeably worse.")
    st.write("ResNet18 is designed for 224×224 images and shrinks the input quickly in its first layers. With 64×64 "
             "input, only a 2×2 feature map would be left at the end. So we changed the stride of the first "
             "convolution from 2 to 1 and removed the max-pooling layer. The last layer was replaced with a "
             "512 → 11 fully connected layer, one output per organ.")

    st.subheader("Training")
    st.write("Each grayscale image was copied to 3 channels and normalised with the ImageNet mean and standard "
             "deviation. During training we applied small random rotations, zooms and shifts. We did not use "
             "horizontal flips, because a flip would turn a left kidney into a right kidney.")
    st.write(f"We trained with mixed precision (FP16). On the same GPU, one training step took {b['amp']} ms, "
             f"compared with {b['fp32']} ms in normal FP32. Accuracy stayed the same.")

    rows = [
        ("Architecture", "ResNet18 (18-layer residual CNN, He et al. 2016)"),
        ("Initial weights", "ImageNet-1k pretrained (transfer learning)"),
        ("Changes for 64×64 input", "First conv stride 2 → 1, max-pool removed"),
        ("Output layer", "Fully-connected, 512 → 11 organ classes"),
        ("Parameters", f"{n_params / 1e6:.2f} M"),
        ("Dataset", "OrganAMNIST (MedMNIST v2), 64×64 abdominal CT"),
        ("Training images", "34,561 (validation 6,491, test 17,778)"),
        ("Epochs / batch size / learning rate", f"{cfg['epochs']} / {cfg['bs']} / {cfg['lr']}"),
        ("Optimiser / schedule", "AdamW (weight decay 1e-4) / One-Cycle"),
        ("Loss", "Cross-entropy with label smoothing 0.1"),
        ("Precision / hardware", f"Mixed FP16 (AMP) / {hardware}"),
    ]
    st.table(pd.DataFrame(rows, columns=["Setting", "Value"]).set_index("Setting"))

    st.subheader("Result")
    st.write(f"The final model gets {m['test_acc']:.1%} accuracy on the test set. For comparison, the MedMNIST v2 "
             "paper reports 93.5% for ResNet-18 at 28px and 95.1% at 224px on OrganAMNIST. The most common "
             "mistake is confusing the left kidney with the spleen. Both organs sit on the same side of the "
             "body at a similar height, so this is understandable.")

    st.subheader("References")
    st.markdown("""
1. J. Yang et al., MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification, *Scientific Data*, 2023.
2. K. He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
3. R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks, ICCV 2017.
4. P. Micikevicius et al., Mixed Precision Training, ICLR 2018.
""")


missing = [p for p in (ART / "model.pt", ART / "metrics.json", ART / "history.json") if not p.exists()]
if missing:
    st.title("Doctor vs AI")
    st.error("The trained model isn't here yet: " + ", ".join(p.name for p in missing))
    st.markdown("""
1. Open `OrganAMNIST_ResNet18_Training.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
3. Unzip the downloaded `artifacts.zip` into this project folder, so you have `artifacts/model.pt`.
4. Refresh this page.
""")
    st.stop()

pages = [
    st.Page(page_play, title="Play: Doctor vs AI", default=True),
    st.Page(page_vision, title="Where is the AI looking?"),
    st.Page(page_upload, title="Upload & Predict"),
    st.Page(page_report, title="AI Report Card"),
    st.Page(page_how, title="How it was built"),
]
st.navigation(pages).run()
