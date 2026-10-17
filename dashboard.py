
import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="NEXUS | Road Safety AI",
    page_icon="◈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_ROOT = BASE_DIR / "evidence"

# Dedicated evidence folders (supported if the camera uses them)
HELMET_EVIDENCE_DIR = EVIDENCE_ROOT / "helmet"
TRIPLE_EVIDENCE_DIR = EVIDENCE_ROOT / "triple_riding_clean"
NUMBER_PLATE_EVIDENCE_DIR = EVIDENCE_ROOT / "number_plate"

# Backward compatibility with the current camera script
LEGACY_EVIDENCE_DIR = EVIDENCE_ROOT / "triple_riding"

for folder in (
    HELMET_EVIDENCE_DIR,
    TRIPLE_EVIDENCE_DIR,
    NUMBER_PLATE_EVIDENCE_DIR,
    LEGACY_EVIDENCE_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)

# Model paths
HELMET_MODEL = BASE_DIR / "runs/detect/train5/weights/best.pt"
TRIPLE_MODEL = BASE_DIR / "runs/detect/triple_riding_fresh/weights/best.pt"

PLATE_CANDIDATES = [
    BASE_DIR / "runs/detect/number_plate_fresh/weights/best.pt",
    BASE_DIR / "runs/number_plate_new/train/weights/best.pt",
    BASE_DIR / "runs/detect/runs/number_plate_new/train/weights/best.pt",
]
PLATE_MODEL = next(
    (p for p in PLATE_CANDIDATES if p.exists()),
    PLATE_CANDIDATES[0],
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -------------------- SCI-FI DESIGN --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #050914;
    color: #e5f3ff;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,220,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,220,255,.045) 1px, transparent 1px);
    background-size: 45px 45px;
    mask-image: linear-gradient(to bottom, black, transparent 90%);
}

section[data-testid="stSidebar"] {
    background: rgba(7,14,29,.98);
    border-right: 1px solid rgba(0,220,255,.2);
}

section[data-testid="stSidebar"] * {
    color: #d9edff;
}

h1, h2, h3, p, label {
    color: #e5f3ff;
}

.brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    letter-spacing: 5px;
    color: #00e5ff;
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(28px, 4vw, 45px);
    font-weight: 800;
    letter-spacing: 2px;
    line-height: 1.25;
    color: #f0fbff;
    text-shadow: 0 0 25px rgba(0,220,255,.24);
}

.hero-subtitle {
    color: #91a9c4;
    font-size: 15px;
    margin-top: 12px;
    line-height: 1.7;
}

.hero {
    padding: 32px;
    border-radius: 18px;
    border: 1px solid rgba(0,220,255,.3);
    background: linear-gradient(
        120deg,
        rgba(10,35,58,.96),
        rgba(13,20,43,.96),
        rgba(30,17,58,.94)
    );
    box-shadow: 0 0 30px rgba(0,180,255,.06);
    margin-bottom: 24px;
}

.section-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    letter-spacing: 3px;
    color: #76dfff;
    margin: 28px 0 16px;
}

.feature-card {
    background: linear-gradient(145deg, #112035, #091120);
    border: 1px solid rgba(88,157,194,.28);
    border-radius: 14px;
    padding: 22px;
    min-height: 150px;
    margin-bottom: 14px;
    transition: border-color .2s ease, transform .2s ease;
}

.feature-card:hover {
    border-color: #00d9ff;
    transform: translateY(-3px);
}

.feature-icon {
    font-size: 27px;
    margin-bottom: 12px;
}

.feature-title {
    color: #eafaff;
    font-size: 17px;
    font-weight: 700;
}

.feature-desc {
    color: #8fa9c3;
    font-size: 13px;
    margin-top: 9px;
    line-height: 1.6;
}

.ready {
    color: #54ffb1;
    font-size: 11px;
    letter-spacing: 1px;
    font-weight: 700;
    margin-top: 13px;
}

.pending {
    color: #ffcf70;
    font-size: 11px;
    letter-spacing: 1px;
    font-weight: 700;
    margin-top: 13px;
}

div[data-testid="stMetric"] {
    background: rgba(14,28,47,.85);
    border: 1px solid rgba(0,210,255,.2);
    border-radius: 12px;
    padding: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #8fa9c3;
}

div[data-testid="stMetricValue"] {
    color: #d9faff;
}

.stButton > button, .stDownloadButton > button {
    background: #092b40;
    color: #8cf3ff;
    border: 1px solid #087b9a;
    border-radius: 8px;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #0b4057;
    border-color: #00e5ff;
    color: white;
}

div[data-testid="stImage"] img {
    border: 1px solid rgba(0,229,255,.25);
    border-radius: 12px;
}

hr {
    border-color: rgba(0,229,255,.15);
}

.small-muted {
    color: #69849f;
    font-size: 12px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- EVIDENCE HELPERS --------------------

def list_images(folder):
    if not folder.exists():
        return []

    return sorted(
        [
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def unique_paths(paths):
    seen = set()
    output = []

    for path in paths:
        identity = str(path.resolve())

        if identity not in seen:
            seen.add(identity)
            output.append(path)

    return sorted(
        output,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_helmet_evidence():
    # New dedicated folder
    images = list_images(HELMET_EVIDENCE_DIR)

    # Current camera script saves both categories into one folder.
    # Separate them using the no_helmet filename prefix.
    legacy = list_images(LEGACY_EVIDENCE_DIR)

    images.extend(
        path for path in legacy
        if path.name.lower().startswith(
            ("no_helmet_", "helmet_", "without_helmet_")
        )
    )

    return unique_paths(images)


def get_triple_evidence():
    # New dedicated folder
    images = list_images(TRIPLE_EVIDENCE_DIR)

    # Backward compatibility with the current camera script
    legacy = list_images(LEGACY_EVIDENCE_DIR)

    images.extend(
        path for path in legacy
        if path.name.lower().startswith(
            ("triple_riding_", "triple_seat_")
        )
    )

    return unique_paths(images)

def get_number_plate_evidence():
    return unique_paths(list_images(NUMBER_PLATE_EVIDENCE_DIR))
def render_evidence_section(title, images, key):
    st.markdown(
        f'<div class="section-label">◈ &nbsp; {title}</div>',
        unsafe_allow_html=True,
    )

    if not images:
        st.info("No evidence saved in this section yet.")
        return

    st.metric("Images saved", len(images))

    search = st.text_input(
        "Search filenames",
        placeholder="Enter filename...",
        key=f"search_{key}",
    )

    filtered = [
        path for path in images
        if search.lower() in path.name.lower()
    ]

    if not filtered:
        st.warning("No images match that search.")
        return

    for start in range(0, len(filtered), 3):
        columns = st.columns(3)

        for column, path in zip(columns, filtered[start:start + 3]):
            with column:
                st.image(
                    str(path),
                    caption=path.name,
                    use_container_width=True,
                )

                with open(path, "rb") as file:
                    st.download_button(
                        "↓ Download evidence",
                        data=file.read(),
                        file_name=path.name,
                        mime="image/jpeg",
                        key=f"download_{key}_{path.name}",
                    )


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">◈ N E X U S</div>
    <div style="font-size:20px;font-weight:700;color:#e8faff;">
        ROAD SAFETY AI
    </div>
    <p style="color:#718ba6;font-size:12px;">
        INTELLIGENT MONITORING SYSTEM
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Evidence Gallery", "About"],
    )

    st.divider()

    st.markdown("""
    <div class="small-muted">SYSTEM STATUS</div>
    <p style="color:#54ffb1;font-size:13px;">● WEBSITE ONLINE</p>
    <p style="color:#91a9c4;font-size:12px;">
        Desktop camera runs separately.
    </p>
    """, unsafe_allow_html=True)


# -------------------- MODEL STATUS --------------------

helmet_ready = HELMET_MODEL.exists()
triple_ready = TRIPLE_MODEL.exists()
plate_ready = PLATE_MODEL.exists()

helmet_images = get_helmet_evidence()
triple_images = get_triple_evidence()


# -------------------- HOME --------------------

if page == "Home":

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("""
        <div class="hero">
            <div class="brand">NEXUS // INTELLIGENCE SYSTEM</div>
            <div class="hero-title">
                ROAD SAFETY<br>REIMAGINED.
            </div>
            <div class="hero-subtitle">
                AI-powered traffic monitoring. One smarter road.
            </div>
            <div style="margin-top:24px;color:#76dfff;
                font-family:Orbitron,sans-serif;font-size:11px;
                letter-spacing:2px;">
                VISION SYSTEM / OBJECT DETECTION
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="hero" style="text-align:center;min-height:230px;">
            <div style="font-family:Orbitron,sans-serif;
                color:#00e5ff;letter-spacing:3px;font-size:12px;">
                NEXUS RADAR
            </div>
            <div style="font-size:90px;margin:14px 0;">◉</div>
            <div style="color:#7deeff;letter-spacing:3px;font-size:11px;">
                SYSTEM READY
            </div>
            <div style="color:#607e9b;font-size:11px;margin-top:10px;">
                VISUAL INTERFACE ONLY
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">◈ &nbsp; DETECTION SYSTEMS</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    cards = [
        (c1, "🪖", "Helmet Detection", "Detects helmet-related classes.", helmet_ready),
        (c2, "👥", "Triple Riding", "Dedicated single/double/triple riding model.", triple_ready),
        (c3, "🔢", "Number Plates", "Number-plate detection model.", plate_ready),
    ]

    for col, icon, title, description, ready in cards:
        with col:
            status = "MODEL FOUND" if ready else "MODEL NOT FOUND"
            css_class = "ready" if ready else "pending"

            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{description}</div>
                <div class="{css_class}">● {status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">◈ &nbsp; EVIDENCE OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Models Found",
        sum([helmet_ready, triple_ready, plate_ready]),
        "/ 3",
    )
    m2.metric("Helmet Evidence", len(get_helmet_evidence()))
    m3.metric("Triple Riding Evidence", len(get_triple_evidence()))
    m4.metric("Number Plate Evidence", len(get_number_plate_evidence()))

    st.info(
        "The camera runs in a separate desktop window. "
        "This website only reads saved images; it does not access your "
        "webcam or run object-detection inference."
    )

    st.markdown(
        f'<div class="small-muted" style="text-align:center;margin-top:30px;">'
        f'NEXUS AI &nbsp; • &nbsp; ROAD SAFETY PROJECT &nbsp; • &nbsp; '
        f'{datetime.now().strftime("%d %b %Y")}'
        f'</div>',
        unsafe_allow_html=True,
    )


# -------------------- EVIDENCE GALLERY --------------------

elif page == "Evidence Gallery":

    st.markdown("""
    <div class="brand">NEXUS // ARCHIVE</div>
    <div class="hero-title" style="font-size:34px;">EVIDENCE GALLERY</div>
    <p class="hero-subtitle">
        Review helmet violations and triple-riding detections separately.
    </p>
    """, unsafe_allow_html=True)

    def gallery_content():
        render_evidence_section(
            "HELMET EVIDENCE",
            get_helmet_evidence(),
            "helmet",
        )

        st.divider()

        render_evidence_section(
            "TRIPLE RIDING EVIDENCE",
            get_triple_evidence(),
            "triple",
        )

        st.divider()

        render_evidence_section(
            "NUMBER PLATE EVIDENCE",
            get_number_plate_evidence(),
            "number_plate",
        )

    if hasattr(st, "fragment"):
        st.fragment(run_every="5s")(gallery_content)()
    else:
        if st.button("↻ Refresh evidence"):
            st.rerun()

        gallery_content()



# -------------------- ABOUT --------------------

elif page == "About":

    st.markdown("""
    <div class="brand">NEXUS // INFORMATION</div>
    <div class="hero-title" style="font-size:34px;">ABOUT THE PROJECT</div>
    <p class="hero-subtitle">A simple idea. Safer roads.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🎯 Our Goal</div>
        <div class="feature-desc">
            Use computer vision to help identify helmet violations,
            triple riding and motorcycle number plates.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">⚙️ How It Works</div>
        <div class="feature-desc">
            <b>01 — Capture</b> Camera runs as a separate desktop process.<br><br>
            <b>02 — Detect</b> Local YOLO models analyse camera frames.<br><br>
            <b>03 — Save</b> Candidate evidence images are stored locally.<br><br>
            <b>04 — Review</b> This website displays saved evidence in separate
            helmet and triple-riding sections.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">🧠 Built With</div>
        <div class="feature-desc">Python • OpenCV • YOLO • Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Images are stored locally on your Mac. This dashboard does not "
        "upload evidence to a cloud server or run the webcam."
    )
