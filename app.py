import os
import sys
import warnings

import numpy as np
import librosa

# Allow backend/app.py to import model definitions from the models folder.
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_as_model import ASASTConformer
from train_mr_model import MRASTConformer
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ============================================================
# CARDIOPULMOCARE
# HEART 4-VALVE AST + CONFORMER
# ============================================================

DEVICE = torch.device("cpu")

VALVES = ["AV", "MV", "PV", "TV"]

SAMPLE_RATE = 4000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
TIME_STEPS = 128
FMIN = 20
FMAX = 1800

ALLOWED_EXTENSIONS = {"wav"}


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HEART_MODEL_PATH = os.path.join(
    BASE_DIR, "heart_4valve_ast_conformer.pth"
)

HEART_THRESHOLD_PATH = os.path.join(
    BASE_DIR, "heart_4valve_ast_conformer_threshold.txt"
)

AS_MODEL_PATH = os.path.join(
    BASE_DIR, "as_ast_conformer.pth"
)

AS_THRESHOLD_PATH = os.path.join(
    BASE_DIR, "as_ast_conformer_threshold.txt"
)

MR_MODEL_PATH = os.path.join(
    BASE_DIR, "mr_ast_conformer.pth"
)

MR_THRESHOLD_PATH = os.path.join(
    BASE_DIR, "mr_ast_conformer_threshold.txt"
)
# ============================================================
# THRESHOLD LOADER
# ============================================================

def load_threshold(path, default=0.50):

    if not os.path.exists(path):
        print(
            f"Threshold file not found: {path}"
        )
        return default

    try:

        with open(path, "r") as f:
            value = float(
                f.read().strip()
            )

        if not 0.0 < value < 1.0:
            return default

        return value

    except Exception:
        return default


HEART_THRESHOLD = load_threshold(
    HEART_THRESHOLD_PATH,
    0.50
)


AS_THRESHOLD = load_threshold(
    AS_THRESHOLD_PATH,
    0.50
)

MR_THRESHOLD = load_threshold(
    MR_THRESHOLD_PATH,
    0.50
)


# ============================================================
# ============================================================
# HEART MODEL
# ============================================================
# ============================================================


# ============================================================
# HEART AST PATCH EMBEDDING
# ============================================================

class ASTPatchEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim=128
    ):

        super().__init__()

        self.projection = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=16,
            stride=16
        )

    def forward(self, x):

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(
            1,
            2
        )

        return x


# ============================================================
# HEART AST BRANCH
# ============================================================

class ASTBranch(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.30
    ):

        super().__init__()

        self.patch_embedding = ASTPatchEmbedding(
            embed_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                65,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        nn.init.normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        batch_size = x.size(0)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [
                cls,
                x
            ],
            dim=1
        )

        x = x + self.position_embedding[
            :,
            :x.size(1),
            :
        ]

        x = self.encoder(x)

        x = self.norm(x)

        return x[:, 0]


# ============================================================
# HEART CONFORMER FEED FORWARD
# ============================================================

class ConformerFeedForward(nn.Module):

    def __init__(
        self,
        dim=128,
        expansion=4,
        dropout=0.20
    ):

        super().__init__()

        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(
            dim
        )

        self.linear1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.norm(x)

        x = self.linear1(x)

        x = F.silu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        x = self.dropout(x)

        return x


# ============================================================
# HEART CONFORMER CONVOLUTION
# ============================================================

class ConformerConvModule(nn.Module):

    def __init__(
        self,
        dim=128,
        kernel_size=15,
        dropout=0.20
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.pointwise1 = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=1
        )

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        self.batch_norm = nn.BatchNorm1d(
            dim
        )

        self.pointwise2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.norm(x)

        x = x.transpose(
            1,
            2
        )

        x = self.pointwise1(x)

        x = F.glu(
            x,
            dim=1
        )

        x = self.depthwise(x)

        x = self.batch_norm(x)

        x = F.silu(x)

        x = self.pointwise2(x)

        x = self.dropout(x)

        x = x.transpose(
            1,
            2
        )

        return x


# ============================================================
# HEART CONFORMER BLOCK
# ============================================================

class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim=128,
        num_heads=4,
        dropout=0.20
    ):

        super().__init__()

        self.ff1 = ConformerFeedForward(
            dim=dim,
            expansion=4,
            dropout=dropout
        )

        self.attention_norm = nn.LayerNorm(
            dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.conv = ConformerConvModule(
            dim=dim,
            kernel_size=15,
            dropout=dropout
        )

        self.ff2 = ConformerFeedForward(
            dim=dim,
            expansion=4,
            dropout=dropout
        )

        self.final_norm = nn.LayerNorm(
            dim
        )

    def forward(self, x):

        x = x + (
            0.5 *
            self.ff1(x)
        )

        normalized = self.attention_norm(
            x
        )

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        x = x + self.attention_dropout(
            attention_output
        )

        x = x + self.conv(x)

        x = x + (
            0.5 *
            self.ff2(x)
        )

        x = self.final_norm(x)

        return x


# ============================================================
# HEART CONFORMER BRANCH
# ============================================================

class ConformerBranch(nn.Module):

    def __init__(
        self,
        input_dim=128,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.20
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            model_dim
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                TIME_STEPS,
                model_dim
            )
        )

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(
            model_dim
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        x = x.squeeze(1)

        x = x.transpose(
            1,
            2
        )

        x = self.input_projection(x)

        x = x + self.position_embedding[
            :,
            :x.size(1),
            :
        ]

        for block in self.blocks:

            x = block(x)

        x = self.norm(x)

        x = x.mean(
            dim=1
        )

        return x


# ============================================================
# HEART FOUR VALVE MODEL
# ============================================================

class HeartFourValveASTConformer(nn.Module):

    def __init__(self):

        super().__init__()

        # Shared AST
        self.ast_branch = ASTBranch(
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.30
        )

        # Shared Conformer
        self.conformer_branch = ConformerBranch(
            input_dim=128,
            model_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.20
        )

        # Valve embedding
        self.valve_embedding = nn.Embedding(
            4,
            16
        )

        # AST + Conformer + Valve ID
        self.valve_fusion = nn.Sequential(
            nn.Linear(
                128 + 128 + 16,
                128
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35)
        )

        # Valve evidence
        self.valve_classifier = nn.Linear(
            128,
            1
        )

        # Valve attention
        self.valve_attention = nn.Sequential(
            nn.Linear(
                128,
                32
            ),
            nn.Tanh(),
            nn.Dropout(0.20),
            nn.Linear(
                32,
                1
            )
        )

        # Patient fusion
        self.patient_fusion = nn.Sequential(
            nn.Linear(
                128 * 2,
                128
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.40)
        )

        # Patient classifier
        self.patient_classifier = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        valve_features = []
        valve_logits = []

        for valve_index in range(4):

            valve_x = x[
                :,
                valve_index,
                :,
                :,
                :
            ]

            # AST
            ast_feature = self.ast_branch(
                valve_x
            )

            # Conformer
            conformer_feature = self.conformer_branch(
                valve_x
            )

            # Valve ID
            valve_ids = torch.full(
                (
                    valve_x.size(0),
                ),
                valve_index,
                dtype=torch.long,
                device=valve_x.device
            )

            valve_id_feature = self.valve_embedding(
                valve_ids
            )

            # Fusion
            combined = torch.cat(
                [
                    ast_feature,
                    conformer_feature,
                    valve_id_feature
                ],
                dim=1
            )

            valve_feature = self.valve_fusion(
                combined
            )

            # Valve evidence
            valve_logit = self.valve_classifier(
                valve_feature
            )

            valve_features.append(
                valve_feature
            )

            valve_logits.append(
                valve_logit
            )

        valve_features = torch.stack(
            valve_features,
            dim=1
        )

        valve_logits = torch.cat(
            valve_logits,
            dim=1
        )

        # Valve attention
        attention_scores = self.valve_attention(
            valve_features
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # Weighted feature
        weighted_features = (
            valve_features *
            attention_weights
        )

        weighted_features = weighted_features.sum(
            dim=1
        )

        # Mean feature
        mean_features = valve_features.mean(
            dim=1
        )

        # Patient fusion
        patient_input = torch.cat(
            [
                weighted_features,
                mean_features
            ],
            dim=1
        )

        patient_feature = self.patient_fusion(
            patient_input
        )

        patient_logit = self.patient_classifier(
            patient_feature
        ).squeeze(1)

        return (
            patient_logit,
            valve_logits,
            attention_weights.squeeze(-1)
        )
print("DEBUG BASE_DIR:", BASE_DIR)
print("DEBUG HEART MODEL PATH:", HEART_MODEL_PATH)
print("DEBUG CURRENT DIRECTORY:", os.getcwd())
print("DEBUG FILES IN BASE_DIR:")

try:
    print(os.listdir(BASE_DIR))
except Exception as e:
    print("Directory error:", e)

# ============================================================
# LOAD HEART MODEL
# ============================================================

print()
print("=" * 70)
print("CARDIOPULMOCARE BACKEND")
print("=" * 70)

print()
print("Loading 4-valve Heart AST + Conformer:")
print(HEART_MODEL_PATH)

if not os.path.exists(
    HEART_MODEL_PATH
):

    raise FileNotFoundError(
        f"Heart model not found:\n{HEART_MODEL_PATH}"
    )


heart_model = HeartFourValveASTConformer().to(
    DEVICE
)

heart_checkpoint = torch.load(
    HEART_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(heart_checkpoint, dict)
    and
    "model_state_dict" in heart_checkpoint
):

    heart_model.load_state_dict(
        heart_checkpoint[
            "model_state_dict"
        ]
    )

else:

    heart_model.load_state_dict(
        heart_checkpoint
    )

heart_model.eval()

print(
    "4-valve Heart model loaded successfully."
)


# ============================================================
# LOAD AS MODEL
# ============================================================

print()
print("Loading Aortic Stenosis AST + Conformer:")
print(AS_MODEL_PATH)

if not os.path.exists(AS_MODEL_PATH):
    raise FileNotFoundError(
        f"AS model not found:\n{AS_MODEL_PATH}"
    )

as_model = ASASTConformer().to(DEVICE)

as_checkpoint = torch.load(
    AS_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(as_checkpoint, dict)
    and "model_state_dict" in as_checkpoint
):
    as_model.load_state_dict(
        as_checkpoint["model_state_dict"]
    )
else:
    as_model.load_state_dict(
        as_checkpoint
    )

as_model.eval()

print(
    "Aortic Stenosis model loaded successfully."
)


# ============================================================
# LOAD MR MODEL
# ============================================================

print()
print("Loading Mitral Regurgitation AST + Conformer:")
print(MR_MODEL_PATH)

if not os.path.exists(MR_MODEL_PATH):
    raise FileNotFoundError(
        f"MR model not found:\n{MR_MODEL_PATH}"
    )

mr_model = MRASTConformer().to(DEVICE)

mr_checkpoint = torch.load(
    MR_MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(mr_checkpoint, dict)
    and "model_state_dict" in mr_checkpoint
):
    mr_model.load_state_dict(
        mr_checkpoint["model_state_dict"]
    )
else:
    mr_model.load_state_dict(
        mr_checkpoint
    )

mr_model.eval()

print(
    "Mitral Regurgitation model loaded successfully."
)

print()
print(
    f"Heart threshold: {HEART_THRESHOLD:.4f}"
)

print(
    f"AS threshold: {AS_THRESHOLD:.4f}"
)

print(
    f"MR threshold: {MR_THRESHOLD:.4f}"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://smart-cardio-care.vercel.app"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=False
)

# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HEART AUDIO PREPROCESSING
# IMPORTANT:
# MATCHES HEART TRAINING
# ============================================================

def heart_audio_to_mel(path):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        audio, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

    if len(audio) == 0:

        raise ValueError(
            "Heart audio file is empty."
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mean = mel_db.mean()

    std = mel_db.std()

    mel_db = (
        mel_db - mean
    ) / (
        std + 1e-8
    )

    if mel_db.shape[1] < TIME_STEPS:

        padding = (
            TIME_STEPS -
            mel_db.shape[1]
        )

        mel_db = np.pad(
            mel_db,
            (
                (0, 0),
                (0, padding)
            ),
            mode="constant"
        )

    else:

        mel_db = mel_db[
            :,
            :TIME_STEPS
        ]

    return mel_db.astype(
        np.float32
    )


# ============================================================
# MAKE TENSOR
# ============================================================

def make_tensor(spectrogram):

    tensor = torch.tensor(
        spectrogram,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return tensor.to(
        DEVICE
    )


# ============================================================
# HEART SINGLE VALVE FEATURE
# ============================================================

def get_heart_valve_result(
    audio_path,
    valve_index
):

    spectrogram = heart_audio_to_mel(
        audio_path
    )

    tensor = make_tensor(
        spectrogram
    )

    with torch.no_grad():

        patient_dummy_input = tensor.unsqueeze(
            1
        )

        _, valve_logits, _ = heart_model(
            torch.cat(
                [
                    patient_dummy_input,
                    patient_dummy_input,
                    patient_dummy_input,
                    patient_dummy_input
                ],
                dim=1
            )
        )

        valve_probability = torch.sigmoid(
            valve_logits[0][valve_index]
        ).item()

    return valve_probability


# ============================================================
# HEART FOUR VALVE PREDICTION
# ============================================================

def predict_patient_heart(
    valve_paths
):

    missing = []

    for valve in VALVES:

        if valve not in valve_paths:

            missing.append(
                valve
            )

    if missing:

        raise ValueError(
            "Missing heart valve files: "
            + ", ".join(missing)
        )

    valve_tensors = []

    for valve in VALVES:

        spectrogram = heart_audio_to_mel(
            valve_paths[valve]
        )

        tensor = torch.tensor(
            spectrogram,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)

        valve_tensors.append(
            tensor
        )

    # [4, 1, 128, 128]
    x = torch.stack(
        valve_tensors,
        dim=0
    )

    # [1, 4, 1, 128, 128]
    x = x.unsqueeze(0)

    x = x.to(
        DEVICE
    )

    with torch.no_grad():

        patient_logit, valve_logits, attention_weights = heart_model(
            x
        )

        patient_probability = torch.sigmoid(
            patient_logit
        ).item()

        valve_probabilities = torch.sigmoid(
            valve_logits
        )[0].cpu().numpy()

        attention = attention_weights[
            0
        ].cpu().numpy()

    prediction = int(
        patient_probability >= HEART_THRESHOLD
    )

    result = (
        "Murmur Present"
        if prediction == 1
        else
        "Murmur Absent"
    )

    valve_results = {}

    for index, valve in enumerate(VALVES):

        valve_results[valve] = {

            "murmur_probability": round(
                float(
                    valve_probabilities[index]
                ) * 100,
                2
            ),

            "attention": round(
                float(
                    attention[index]
                ) * 100,
                2
            ),

            "filename": os.path.basename(
                valve_paths[valve]
            )
        }

    # --------------------------------------------------------
    # AS + MR analysis
    # --------------------------------------------------------

    as_mr_result = predict_patient_as_mr(
        valve_paths
    )

    as_probability = as_mr_result[
        "as_probability"
    ]

    mr_probability = as_mr_result[
        "mr_probability"
    ]

    # --------------------------------------------------------
    # Final four-class presentation
    # --------------------------------------------------------

    final_result = build_final_four_class_result(
        patient_probability,
        as_probability,
        mr_probability
    )

    return {

        "prediction":
            prediction,

        "result":
            result,

        "murmur_probability":
            round(
                patient_probability * 100,
                2
            ),

        "absent_probability":
            round(
                (1.0 - patient_probability) * 100,
                2
            ),

        "murmur_threshold":
            round(
                HEART_THRESHOLD,
                4
            ),

        "as_probability":
            round(
                as_probability * 100,
                2
            ),

        "as_threshold":
            round(
                AS_THRESHOLD,
                4
            ),

        "mr_probability":
            round(
                mr_probability * 100,
                2
            ),

        "mr_threshold":
            round(
                MR_THRESHOLD,
                4
            ),

        "as_valve_probabilities":
            as_mr_result[
                "as_valve_probabilities"
            ],

        "mr_valve_probabilities":
            as_mr_result[
                "mr_valve_probabilities"
            ],

        "final_prediction":
            final_result[
                "final_prediction"
            ],

        "final_class_probabilities":
            final_result[
                "final_class_probabilities"
            ],

        "valves":
            valve_results
    }


# ============================================================
# AS + MR FOUR-VALVE ANALYSIS
# IMPORTANT:
# AS/MR models were trained on single PCG recordings.
# For the 4-valve patient UI, we run each model on AV/MV/PV/TV
# separately and aggregate the four outputs by arithmetic mean.
# This is an inference-time fusion/aggregation method, not
# four-valve AS/MR training.
# ============================================================

def predict_patient_as_mr(valve_paths):

    as_probabilities = []
    mr_probabilities = []

    for valve in VALVES:

        spectrogram = heart_audio_to_mel(
            valve_paths[valve]
        )

        tensor = make_tensor(
            spectrogram
        )

        with torch.no_grad():

            as_logit = as_model(
                tensor
            )

            mr_logit = mr_model(
                tensor
            )

            as_probability = torch.sigmoid(
                as_logit
            ).item()

            mr_probability = torch.sigmoid(
                mr_logit
            ).item()

        as_probabilities.append(
            as_probability
        )

        mr_probabilities.append(
            mr_probability
        )

    # Mean of the four valve-specific model outputs.
    as_probability = float(
        np.mean(as_probabilities)
    )

    mr_probability = float(
        np.mean(mr_probabilities)
    )

    # Existing murmur patient-level probability is computed
    # by the completed four-valve murmur model.
    # This function returns the independent AS/MR probabilities
    # and a four-class normalized presentation score.
    return {
        "as_probability": as_probability,
        "mr_probability": mr_probability,
        "as_valve_probabilities": {
            valve: round(
                float(as_probabilities[index]) * 100,
                2
            )
            for index, valve in enumerate(VALVES)
        },
        "mr_valve_probabilities": {
            valve: round(
                float(mr_probabilities[index]) * 100,
                2
            )
            for index, valve in enumerate(VALVES)
        }
    }


def build_final_four_class_result(
    murmur_probability,
    as_probability,
    mr_probability
):
    """
    Build a single four-class presentation from three
    independently trained model probabilities.

    This normalization makes the displayed class values sum
    to 100%, but they should be treated as combined model scores,
    not clinically calibrated probabilities.
    """

    # Murmur is the primary screening gate.
    # If murmur is absent, the final class is Normal.
    if murmur_probability < HEART_THRESHOLD:
        final_prediction = "Normal"

    elif (
        as_probability >= AS_THRESHOLD
        and as_probability > mr_probability
    ):
        final_prediction = "Aortic Stenosis"

    elif (
        mr_probability >= MR_THRESHOLD
        and mr_probability > as_probability
    ):
        final_prediction = "Mitral Regurgitation"

    else:
        final_prediction = "Murmur"

    probabilities = {
        "Normal": 100.0 if final_prediction == "Normal" else 0.0,
        "Murmur": 100.0 if final_prediction == "Murmur" else 0.0,
        "Aortic Stenosis": 100.0 if final_prediction == "Aortic Stenosis" else 0.0,
        "Mitral Regurgitation": 100.0 if final_prediction == "Mitral Regurgitation" else 0.0
    }

    return {
        "final_prediction": final_prediction,
        "final_class_probabilities": probabilities
    }



# ============================================================
# SINGLE AUDIO - AORTIC STENOSIS
# ============================================================

@app.route("/predict-as", methods=["POST"])
def predict_as():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "Please upload an AS WAV file."
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "success": False,
                "error": "No AS audio file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Only WAV files are supported."
            }), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(
            UPLOAD_FOLDER,
            "AS_" + filename
        )
        file.save(file_path)

        spectrogram = heart_audio_to_mel(file_path)
        tensor = make_tensor(spectrogram)

        with torch.inference_mode():
            tensor = tensor.to(DEVICE)
            logit = as_model(tensor)
            as_probability = torch.sigmoid(logit).item()
            
        prediction = (
            "Aortic Stenosis"
            if as_probability >= AS_THRESHOLD
            else "Normal"
        )

        return jsonify({
            "success": True,
            "model": "Aortic Stenosis AST + Conformer",
            "prediction": prediction,
            "message": (
                "Aortic Stenosis Present"
                if prediction == "Aortic Stenosis"
                else "Aortic Stenosis Not Present"
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# SINGLE AUDIO - MITRAL REGURGITATION
# ============================================================

@app.route("/predict-mr", methods=["POST"])
def predict_mr():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "Please upload an MR WAV file."
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "success": False,
                "error": "No MR audio file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Only WAV files are supported."
            }), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(
            UPLOAD_FOLDER,
            "MR_" + filename
        )
        file.save(file_path)

        spectrogram = heart_audio_to_mel(file_path)
        tensor = make_tensor(spectrogram)

        with torch.inference_mode():
            tensor = tensor.to(DEVICE)
            logit = mr_model(tensor)
            mr_probability = torch.sigmoid(logit).item()
        prediction = (
            "Mitral Regurgitation"
            if mr_probability >= MR_THRESHOLD
            else "Normal"
        )

        return jsonify({
            "success": True,
            "model": "Mitral Regurgitation AST + Conformer",
            "prediction": prediction,
            "message": (
                "Mitral Regurgitation Present"
                if prediction == "Mitral Regurgitation"
                else "Mitral Regurgitation Not Present"
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status":
            "success",

        "message":
            "Cardiac backend is running",

        "models": {

            "heart":
                "4-Valve Heart AST + Conformer",

            "aortic_stenosis":
                "Aortic Stenosis AST + Conformer",

            "mitral_regurgitation":
                "Mitral Regurgitation AST + Conformer"
        },

        "heart_valves":
            VALVES,

        "classes": {

            "heart": {

                "0":
                    "Murmur Absent",

                "1":
                    "Murmur Present"
            },

            "aortic_stenosis": {

                "0":
                    "Normal",

                "1":
                    "Aortic Stenosis"
            },

            "mitral_regurgitation": {

                "0":
                    "Normal",

                "1":
                    "Mitral Regurgitation"
            }
        },

        "thresholds": {

            "heart":
                HEART_THRESHOLD,

            "aortic_stenosis":
                AS_THRESHOLD,

            "mitral_regurgitation":
                MR_THRESHOLD
        }
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "heart_model_loaded":
            True,

        "as_model_loaded":
            True,

        "mr_model_loaded":
            True,

        "device":
            str(DEVICE),

        "heart_threshold":
            HEART_THRESHOLD,

        "as_threshold":
            AS_THRESHOLD,

        "mr_threshold":
            MR_THRESHOLD
    })


# ============================================================
# FOUR VALVE HEART ROUTE
# ============================================================

@app.route(
    "/predict-patient-heart",
    methods=["POST"]
)
def predict_patient_heart_route():

    try:

        valve_files = {}

        # --------------------------------------------------------
        # Preferred:
        # av_file
        # mv_file
        # pv_file
        # tv_file
        # --------------------------------------------------------

        for valve in VALVES:

            field_name = (
                valve.lower()
                + "_file"
            )

            if field_name in request.files:

                file = request.files[
                    field_name
                ]

                if file.filename:

                    if not allowed_file(
                        file.filename
                    ):

                        return jsonify({
                            "success": False,
                            "error":
                                "Only WAV files are supported."
                        }), 400

                    filename = secure_filename(
                        file.filename
                    )

                    file_path = os.path.join(
                        UPLOAD_FOLDER,
                        filename
                    )

                    file.save(
                        file_path
                    )

                    valve_files[
                        valve
                    ] = file_path

        # --------------------------------------------------------
        # Fallback:
        # files[]
        # Filename must contain _AV, _MV, _PV, _TV
        # --------------------------------------------------------

        if len(valve_files) != 4:

            generic_files = request.files.getlist(
                "files"
            )

            for file in generic_files:

                if not file.filename:
                    continue

                if not allowed_file(
                    file.filename
                ):

                    return jsonify({
                        "success": False,
                        "error":
                            "Only WAV files are supported."
                    }), 400

                filename = secure_filename(
                    file.filename
                )

                upper_name = filename.upper()

                detected_valve = None

                for valve in VALVES:

                    if (
                        f"_{valve}.WAV"
                        in upper_name
                        or
                        f"_{valve}"
                        in os.path.splitext(
                            upper_name
                        )[0]
                    ):

                        detected_valve = valve
                        break

                if detected_valve is None:
                    continue

                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                file.save(
                    file_path
                )

                valve_files[
                    detected_valve
                ] = file_path

        # --------------------------------------------------------
        # Check all 4
        # --------------------------------------------------------

        missing = [
            valve
            for valve in VALVES
            if valve not in valve_files
        ]

        if missing:

            return jsonify({

                "success":
                    False,

                "error":
                    "All 4 heart valve WAV files are required.",

                "missing_valves":
                    missing,

                "required":
                    VALVES
            }), 400

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        result = predict_patient_heart(
            valve_files
        )

        return jsonify({

            "success":
                True,

            "model":
                "4-Valve Heart AST + Conformer",

            **result
        })

    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# OLD /predict ROUTE
# ============================================================
# Kept only so old frontend gets a clear message.
# Four valve files are required now.
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    return jsonify({

        "success":
            False,

        "error":
            "Heart prediction now requires 4 valve WAV files: AV, MV, PV and TV.",

        "endpoint":
            "/predict-patient-heart",

        "required_fields":
            [
                "av_file",
                "mv_file",
                "pv_file",
                "tv_file"
            ]
    }), 400


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CARDIOPULMOCARE SERVER")
    print("=" * 70)

    print()
    print(
        "Heart model:"
    )

    print(
        "4-Valve AST + Conformer"
    )

    print(
        "Valves: AV, MV, PV, TV"
    )

    print()
    print(
        f"Heart threshold: {HEART_THRESHOLD:.4f}"
    )

    print(
        f"AS threshold: {AS_THRESHOLD:.4f}"
    )

    print(
        f"MR threshold: {MR_THRESHOLD:.4f}"
    )

    print()
    print(
        "URL:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
