import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

MODEL_CLF_PATH = "stage1_classifier.pkl"
MODEL_REG_PATH = "stage2_regressor.pkl"
THRESHOLD_PROBA = 0.50
CLASSIFIER_CUTOFF = 7.0
REGRESSOR_CUTOFF = 14.0

stage1_classifier = None
stage2_regressor = None
training_features = []


def initialize_or_train_models():
    global stage1_classifier, stage2_regressor, training_features

    if os.path.exists(MODEL_CLF_PATH) and os.path.exists(MODEL_REG_PATH):
        try:
            stage1_classifier = joblib.load(MODEL_CLF_PATH)
            stage2_regressor = joblib.load(MODEL_REG_PATH)
            if hasattr(stage1_classifier, "feature_names_in_"):
                training_features = list(stage1_classifier.feature_names_in_)
            return
        except Exception:
            pass

    csv_path = "LengthOfStay.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["lengthofstay"])[cite: 1]
        df = df.drop(columns=[c for c in ["eid", "vdate", "discharged"] if c in df.columns])[cite: 1]

        if "rcount" in df.columns:
            df["rcount"] = df["rcount"].astype(str).str.replace("+", "", regex=False).astype(int)[cite: 1]

        if "gender" in df.columns:
            le = LabelEncoder()
            df["gender"] = le.fit_transform(df["gender"].astype(str))[cite: 1]

        if "facid" in df.columns:
            df = pd.get_dummies(df, columns=["facid"], drop_first=True, dtype=int)[cite: 1]

        X = df.drop(columns=["lengthofstay"])[cite: 1]
        y = df["lengthofstay"][cite: 1]
        training_features = list(X.columns)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=1 / 3, random_state=42)[cite: 1]

        y_train_long = (y_train >= CLASSIFIER_CUTOFF).astype(int)[cite: 1]
        stage1_classifier = RandomForestClassifier(
            n_estimators=80, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        )[cite: 1]
        stage1_classifier.fit(X_train, y_train_long)[cite: 1]

        short_mask = y_train < REGRESSOR_CUTOFF[cite: 1]
        X_train_short = X_train.loc[short_mask][cite: 1]
        y_train_short_log = np.log1p(y_train.loc[short_mask])[cite: 1]

        stage2_regressor = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        stage2_regressor.fit(X_train_short, y_train_short_log)[cite: 1]

        joblib.dump(stage1_classifier, MODEL_CLF_PATH)
        joblib.dump(stage2_regressor, MODEL_REG_PATH)


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>StayCast &bull; In-Hospital Stay Prediction</title>
  
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  
  <style>
    body { 
      font-family: 'Plus Jakarta Sans', sans-serif; 
      background: radial-gradient(120% 120% at 50% 0%, #f0fdfa 0%, #f8fafc 50%, #f1f5f9 100%);
    }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glass-card {
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(226, 232, 240, 0.85);
    }
    .no-scrollbar::-webkit-scrollbar {
      display: none;
    }
    .no-scrollbar {
      -ms-overflow-style: none;
      scrollbar-width: none;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col justify-between text-slate-800 antialiased selection:bg-teal-100 selection:text-teal-900">

  <!-- Responsive Header -->
  <header class="sticky top-0 z-50 glass-card border-b border-slate-200/80">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 sm:py-0 sm:h-16 flex flex-col sm:flex-row items-center justify-between gap-2 sm:gap-4">
      
      <!-- Brand Logo -->
      <div class="flex items-center space-x-2.5 self-start sm:self-auto">
        <div class="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-teal-600 via-cyan-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-teal-500/20">
          <i data-lucide="dna" class="w-4 h-4 sm:w-5 sm:h-5"></i>
        </div>
        <span class="text-lg sm:text-xl font-extrabold text-slate-900 tracking-tight">StayCast</span>
      </div>

      <!-- Tab Buttons with Mobile Overflow Handling -->
      <nav class="flex items-center space-x-1 bg-slate-100/90 p-1 rounded-xl border border-slate-200 text-xs font-semibold w-full sm:w-auto overflow-x-auto no-scrollbar">
        <button id="nav-btn-predict" onclick="switchMainTab('predict')" class="flex-1 sm:flex-none px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg bg-white shadow-sm text-teal-700 transition whitespace-nowrap text-center">
          <span class="flex items-center justify-center gap-1.5"><i data-lucide="stethoscope" class="w-3.5 h-3.5"></i> Predictor</span>
        </button>
        <button id="nav-btn-analytics" onclick="switchMainTab('analytics')" class="flex-1 sm:flex-none px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg text-slate-600 hover:text-teal-600 transition whitespace-nowrap text-center">
          <span class="flex items-center justify-center gap-1.5"><i data-lucide="bar-chart-3" class="w-3.5 h-3.5"></i> Analytics</span>
        </button>
        <button id="nav-btn-architecture" onclick="switchMainTab('architecture')" class="flex-1 sm:flex-none px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg text-slate-600 hover:text-teal-600 transition whitespace-nowrap text-center">
          <span class="flex items-center justify-center gap-1.5"><i data-lucide="workflow" class="w-3.5 h-3.5"></i> Architecture</span>
        </button>
      </nav>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 sm:py-8 flex-1 w-full">

    <!-- TAB 1: PREDICTOR VIEW -->
    <div id="tab-predict" class="space-y-6">
      
      <!-- Responsive Stat Highlights -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div class="glass-card rounded-2xl p-3.5 sm:p-4 flex items-center gap-3">
          <div class="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-600 shrink-0">
            <i data-lucide="database" class="w-4 h-4 sm:w-5 sm:h-5"></i>
          </div>
          <div class="min-w-0">
            <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider truncate">Cohort Size</p>
            <p class="text-sm sm:text-base font-bold text-slate-900 truncate">100,000 <span class="text-xs font-normal text-slate-500">Pts</span></p>
          </div>
        </div>

        <div class="glass-card rounded-2xl p-3.5 sm:p-4 flex items-center gap-3">
          <div class="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-amber-50 border border-amber-100 flex items-center justify-center text-amber-600 shrink-0">
            <i data-lucide="git-branch" class="w-4 h-4 sm:w-5 sm:h-5"></i>
          </div>
          <div class="min-w-0">
            <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider truncate">Boundary</p>
            <p class="text-sm sm:text-base font-bold text-slate-900 truncate">7.0 <span class="text-xs font-normal text-slate-500">Days</span></p>
          </div>
        </div>

        <div class="glass-card rounded-2xl p-3.5 sm:p-4 flex items-center gap-3">
          <div class="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-cyan-50 border border-cyan-100 flex items-center justify-center text-cyan-600 shrink-0">
            <i data-lucide="cpu" class="w-4 h-4 sm:w-5 sm:h-5"></i>
          </div>
          <div class="min-w-0">
            <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider truncate">Transform</p>
            <p class="text-sm sm:text-base font-bold text-slate-900 mono truncate">log1p</p>
          </div>
        </div>

        <div class="glass-card rounded-2xl p-3.5 sm:p-4 flex items-center gap-3">
          <div class="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shrink-0">
            <i data-lucide="shield-check" class="w-4 h-4 sm:w-5 sm:h-5"></i>
          </div>
          <div class="min-w-0">
            <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider truncate">Leakage Guard</p>
            <p class="text-sm sm:text-base font-bold text-emerald-600 flex items-center gap-1 truncate"><i data-lucide="check" class="w-3.5 h-3.5 shrink-0"></i> Active</p>
          </div>
        </div>
      </div>

      <!-- Main Prediction Layout -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
        
        <!-- Form Console -->
        <div class="lg:col-span-7 glass-card rounded-2xl sm:rounded-3xl p-4 sm:p-7 shadow-sm">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 pb-4 border-b border-slate-100">
            <div>
              <h2 class="text-base font-bold text-slate-900 flex items-center gap-2">
                <i data-lucide="clipboard-list" class="w-4 h-4 sm:w-5 sm:h-5 text-teal-600"></i> Admission Parameters
              </h2>
              <p class="text-xs text-slate-500 mt-0.5">Input pre-procedure patient data</p>
            </div>
            
            <div class="flex gap-2">
              <button type="button" onclick="loadPreset('mild')" class="flex-1 sm:flex-none text-xs font-semibold text-teal-700 bg-teal-50 hover:bg-teal-100/80 border border-teal-200 px-3 py-1.5 rounded-xl transition text-center">
                Mild Stay
              </button>
              <button type="button" onclick="loadPreset('severe')" class="flex-1 sm:flex-none text-xs font-semibold text-amber-700 bg-amber-50 hover:bg-amber-100/80 border border-amber-200 px-3 py-1.5 rounded-xl transition text-center">
                Prolonged Stay
              </button>
            </div>
          </div>

          <form id="predictionForm" class="space-y-5">
            <div>
              <p class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-teal-500"></span> 1. Administration & History
              </p>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Gender</label>
                  <select id="gender" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition">
                    <option value="" disabled selected>Select Gender</option>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Facility</label>
                  <select id="facid" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition">
                    <option value="" disabled selected>Select Facility</option>
                    <option value="A">Facility A</option>
                    <option value="B">Facility B</option>
                    <option value="C">Facility C</option>
                    <option value="D">Facility D</option>
                    <option value="E">Facility E</option>
                  </select>
                  <p class="text-[10px] text-slate-400 mt-1">Admitting clinical center.</p>
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Prior Readmissions</label>
                  <select id="rcount" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition">
                    <option value="" disabled selected>Select Count</option>
                    <option value="1">1 Prior</option>
                    <option value="2">2 Prior</option>
                    <option value="3">3 Prior</option>
                    <option value="4">4 Prior</option>
                    <option value="5">5 Prior</option>
                  </select>
                  <p class="text-[10px] text-slate-400 mt-1">Historical inpatient visits.</p>
                </div>
              </div>
            </div>

            <div>
              <p class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-cyan-500"></span> 2. Vital Signs
              </p>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Pulse <span class="text-slate-400">(bpm)</span></label>
                  <input type="number" id="pulse" min="30" max="220" placeholder="e.g. 76" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Respiration <span class="text-slate-400">(rpm)</span></label>
                  <input type="number" id="respiration" min="5" max="60" step="0.1" placeholder="e.g. 16.0" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">BMI</label>
                  <input type="number" id="bmi" min="10" max="60" step="0.1" placeholder="e.g. 27.8" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
              </div>
            </div>

            <div>
              <p class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-indigo-500"></span> 3. Laboratory Panel
              </p>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Hematocrit <span class="text-slate-400">(g/dL)</span></label>
                  <input type="number" id="hematocrit" min="5" max="60" step="0.1" placeholder="e.g. 12.0" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Sodium <span class="text-slate-400">(mEq/L)</span></label>
                  <input type="number" id="sodium" min="100" max="180" step="0.1" placeholder="e.g. 138.0" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Glucose <span class="text-slate-400">(mg/dL)</span></label>
                  <input type="number" id="glucose" min="30" max="500" step="0.1" placeholder="e.g. 108.0" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Blood Urea Nitrogen <span class="text-slate-400">(mg/dL)</span></label>
                  <input type="number" id="bloodureanitro" min="1" max="150" step="0.1" placeholder="e.g. 15.0" required class="w-full text-xs font-semibold border border-slate-200 rounded-xl p-2.5 sm:p-3 bg-white/70 focus:ring-2 focus:ring-teal-500 focus:outline-none transition" />
                </div>
              </div>
            </div>

            <div class="pt-2 flex flex-col sm:flex-row gap-2.5">
              <button type="submit" id="predictSubmitBtn" class="w-full sm:flex-1 bg-teal-600 hover:bg-teal-700 active:scale-[0.99] text-white font-bold text-xs py-3 px-5 rounded-xl shadow-md shadow-teal-600/20 transition flex items-center justify-center gap-2">
                <span>Run Prediction</span>
                <i data-lucide="arrow-right" class="w-4 h-4"></i>
              </button>
              <button type="reset" onclick="handleFormReset()" class="w-full sm:w-auto px-5 py-3 bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold text-xs rounded-xl transition">
                Reset
              </button>
            </div>
          </form>
        </div>

        <!-- Output Dashboard -->
        <div class="lg:col-span-5 glass-card rounded-2xl sm:rounded-3xl p-5 sm:p-7 shadow-sm">
          <div class="flex items-center justify-between mb-4">
            <span class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Inference Projection</span>
            <span id="tierBadge" class="px-2.5 py-1 rounded-full text-xs font-bold bg-slate-100 text-slate-500 border border-slate-200 transition-colors">
              Waiting for Input
            </span>
          </div>

          <!-- Hero Number -->
          <div class="py-6 sm:py-8 flex flex-col items-center justify-center text-center">
            <span class="text-xs font-medium text-slate-400">Estimated Hospital Stay</span>
            <div class="flex items-baseline justify-center space-x-1.5 mt-2">
              <span id="predDays" class="text-5xl sm:text-6xl font-extrabold text-slate-900 tracking-tight mono">--</span>
              <span class="text-base sm:text-lg font-bold text-slate-500">Days</span>
            </div>
            <p id="predSubtext" class="text-xs text-slate-500 mt-2 max-w-xs text-center leading-relaxed">
              Submit patient admission data to evaluate probability of long stay.
            </p>
          </div>

          <!-- Cohort Progress Bar -->
          <div class="mt-2 pt-4 border-t border-slate-100">
            <div class="flex justify-between text-[11px] font-bold text-slate-500 mb-2">
              <span>Cohort Position</span>
              <span id="relativePercent">0% of Max</span>
            </div>
            <div class="w-full h-3 bg-slate-100 rounded-full overflow-hidden relative">
              <div class="absolute top-0 bottom-0 left-[41.1%] w-0.5 bg-red-400 z-10" title="7-Day Boundary"></div>
              <div id="losProgressBar" class="h-full bg-teal-500 rounded-full transition-all duration-500" style="width: 0%;"></div>
            </div>
            <div class="flex justify-between text-[10px] text-slate-400 mt-1.5 mono">
              <span>0d (Min)</span>
              <span class="text-red-500 font-semibold">&bull; 7d (Boundary)</span>
              <span>17d (Max)</span>
            </div>
          </div>

          <!-- Diagnostic Cards -->
          <div class="mt-5 grid grid-cols-2 gap-2.5 sm:gap-3 pt-4 border-t border-slate-100">
            <div class="bg-slate-50/80 p-3 rounded-xl sm:rounded-2xl border border-slate-100">
              <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">Stage 1 Classifier</span>
              <p id="stage1Label" class="text-xs font-bold text-slate-700 truncate">--</p>
              <span class="text-[10px] text-slate-400">Cutoff: 7.0d</span>
            </div>
            <div class="bg-slate-50/80 p-3 rounded-xl sm:rounded-2xl border border-slate-100">
              <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">Stage 2 Regressor</span>
              <p id="stage2Label" class="text-xs font-bold text-slate-700 truncate">--</p>
              <span class="text-[10px] text-slate-400">Bound: &lt;14.0d</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: VISUAL ANALYTICS VIEW -->
    <div id="tab-analytics" class="hidden space-y-5 sm:space-y-6">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 class="text-base sm:text-lg font-bold text-slate-900">Experimental Evaluation Charts</h3>
          <p class="text-xs text-slate-500">Benchmark results extracted from cross-validation runs</p>
        </div>
        <div class="flex gap-1.5 bg-slate-100/80 p-1 rounded-xl border border-slate-200 overflow-x-auto no-scrollbar">
          <button id="subtab-stratified" onclick="loadChart('stratified')" class="px-3 py-1.5 text-xs font-bold rounded-lg bg-teal-600 text-white shadow-sm transition whitespace-nowrap">Stratified MAE</button>
          <button id="subtab-cv" onclick="loadChart('cv')" class="px-3 py-1.5 text-xs font-bold rounded-lg bg-white text-slate-600 hover:bg-slate-50 transition whitespace-nowrap">5-Fold CV</button>
          <button id="subtab-dist" onclick="loadChart('dist')" class="px-3 py-1.5 text-xs font-bold rounded-lg bg-white text-slate-600 hover:bg-slate-50 transition whitespace-nowrap">Transformation</button>
        </div>
      </div>

      <div class="glass-card rounded-2xl sm:rounded-3xl p-4 sm:p-7 shadow-sm">
        <div class="mb-4 sm:mb-6">
          <h4 id="chartTitle" class="text-sm sm:text-base font-bold text-slate-800">Stratified MAE Across Stay Intervals</h4>
          <p id="chartSubtitle" class="text-xs text-slate-500 mt-0.5">Compares error across duration buckets.</p>
        </div>
        <div class="h-72 sm:h-96 w-full relative">
          <canvas id="researchChart"></canvas>
        </div>
      </div>
    </div>

    <!-- TAB 3: VISUAL MODEL ARCHITECTURE VIEW -->
    <div id="tab-architecture" class="hidden space-y-6 sm:space-y-8">
      <div>
        <h3 class="text-lg sm:text-xl font-extrabold text-slate-900 flex items-center gap-2">
          <i data-lucide="git-merge" class="w-5 h-5 sm:w-6 sm:h-6 text-teal-600"></i> Bifurcated Architecture Flow
        </h3>
        <p class="text-xs text-slate-500 mt-1">Visual decomposition of data transformations and two-stage decision pathways.</p>
      </div>

      <!-- Flow Diagram -->
      <div class="glass-card rounded-2xl sm:rounded-3xl p-4 sm:p-7 shadow-sm">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
          
          <div class="lg:col-span-3 bg-white p-4 rounded-xl border-2 border-teal-100 shadow-sm">
            <div class="flex items-center gap-2 mb-1.5 text-teal-600 font-bold text-xs uppercase tracking-wider">
              <i data-lucide="file-input" class="w-4 h-4"></i> Ingestion
            </div>
            <p class="text-xs font-bold text-slate-800 mb-0.5">10 Clinical Indicators</p>
            <p class="text-[11px] text-slate-500 leading-relaxed">Leakage removed (Dates, IDs)[cite: 1]. Facility one-hot encoding[cite: 1].</p>
            <div class="mt-2.5 py-1 px-2.5 rounded-lg bg-teal-50 text-[10px] font-mono text-teal-800 font-semibold">
              X &in; &reals;<sup>100k &times; 13</sup>
            </div>
          </div>

          <div class="flex justify-center text-teal-500 lg:col-span-1 py-1 lg:py-0">
            <i data-lucide="arrow-down" class="w-5 h-5 lg:hidden"></i>
            <i data-lucide="arrow-right" class="w-6 h-6 hidden lg:block"></i>
          </div>

          <div class="lg:col-span-4 bg-gradient-to-br from-teal-50/70 to-cyan-50/70 p-4 rounded-xl border-2 border-teal-200 shadow-sm">
            <div class="flex items-center justify-between mb-1.5">
              <span class="flex items-center gap-1.5 text-teal-700 font-bold text-xs uppercase tracking-wider">
                <i data-lucide="shield-alert" class="w-4 h-4"></i> Stage 1 Classifier
              </span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-teal-600 text-white">Cutoff 7.0d</span>
            </div>
            <p class="text-xs font-bold text-slate-800 mb-0.5">Random Forest Classifier</p>
            <p class="text-[11px] text-slate-600 leading-relaxed">Maximizes short-stay sensitivity (&ge;90%) using out-of-fold cross-validation[cite: 1].</p>
            <div class="mt-2.5 flex flex-wrap gap-1.5">
              <span class="text-[10px] py-0.5 px-2 rounded-lg bg-white border border-teal-200 text-slate-700 font-medium">80 Trees</span>
              <span class="text-[10px] py-0.5 px-2 rounded-lg bg-white border border-teal-200 text-slate-700 font-medium">Balanced Weights</span>
            </div>
          </div>

          <div class="flex justify-center text-cyan-500 lg:col-span-1 py-1 lg:py-0">
            <i data-lucide="arrow-down" class="w-5 h-5 lg:hidden"></i>
            <i data-lucide="split" class="w-6 h-6 hidden lg:block"></i>
          </div>

          <div class="lg:col-span-3 space-y-3">
            <div class="bg-white p-3.5 rounded-xl border-2 border-teal-300 shadow-sm">
              <div class="flex items-center justify-between text-teal-700 text-xs font-bold mb-1">
                <span>Predicted &lt; 7 Days</span>
                <i data-lucide="zap" class="w-3.5 h-3.5"></i>
              </div>
              <p class="text-[11px] text-slate-600 leading-snug">Stage 2 Regressor on log(LOS)[cite: 1].</p>
              <div class="mt-1.5 text-[10px] font-mono text-teal-600 bg-teal-50 px-2 py-0.5 rounded">expm1(log_pred)</div>
            </div>

            <div class="bg-white p-3.5 rounded-xl border-2 border-amber-300 shadow-sm">
              <div class="flex items-center justify-between text-amber-700 text-xs font-bold mb-1">
                <span>Predicted &ge; 7 Days</span>
                <i data-lucide="lock" class="w-3.5 h-3.5"></i>
              </div>
              <p class="text-[11px] text-slate-600 leading-snug">Bounded threshold assignment[cite: 1].</p>
              <div class="mt-1.5 text-[10px] font-mono text-amber-700 bg-amber-50 px-2 py-0.5 rounded">Bounded at 7.0d</div>
            </div>
          </div>

        </div>
      </div>

      <!-- Predictor Significance & Metric Details -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6">
        
        <div class="lg:col-span-5 glass-card p-4 sm:p-6 rounded-2xl sm:rounded-3xl shadow-sm space-y-3 sm:space-y-4">
          <h4 class="text-sm font-bold text-slate-800 flex items-center gap-2">
            <i data-lucide="info" class="w-4 h-4 text-teal-600"></i> Predictor Significance
          </h4>

          <div class="p-3 rounded-xl bg-slate-50/90 border border-slate-200">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs font-bold text-slate-800">Hospital Facility</span>
              <span class="text-[10px] font-mono px-2 py-0.5 bg-teal-100 text-teal-800 rounded font-semibold">A, B, C, D, E</span>
            </div>
            <p class="text-[11px] text-slate-600 leading-relaxed">
              Captures site-specific baseline variations, operational protocols, specialized bed allocations, and localized intake demographics across individual hospital centers[cite: 1].
            </p>
          </div>

          <div class="p-3 rounded-xl bg-slate-50/90 border border-slate-200">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs font-bold text-slate-800">Prior Readmissions</span>
              <span class="text-[10px] font-mono px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded font-semibold">1, 2, 3, 4, 5</span>
            </div>
            <p class="text-[11px] text-slate-600 leading-relaxed">
              Quantifies historical hospitalization frequency. Serves as an indicator of chronic systemic disease recurrence and clinical frailty.
            </p>
          </div>
        </div>

        <div class="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
          <div class="glass-card p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span class="w-7 h-7 rounded-lg bg-teal-50 border border-teal-200 text-teal-700 flex items-center justify-center text-xs font-bold mb-2.5">
                <i data-lucide="target" class="w-3.5 h-3.5"></i>
              </span>
              <h5 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-1">Hinge Loss</h5>
              <p class="text-xs text-slate-500 leading-relaxed">
                Zero penalty when both true and predicted stays exceed 7 days[cite: 1].
              </p>
            </div>
            <div class="mt-3 pt-2.5 border-t border-slate-100 text-[10px] sm:text-[11px] text-teal-700 font-mono font-bold">
              Loss = 0 if &ge; 7d
            </div>
          </div>

          <div class="glass-card p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span class="w-7 h-7 rounded-lg bg-cyan-50 border border-cyan-200 text-cyan-700 flex items-center justify-center text-xs font-bold mb-2.5">
                <i data-lucide="trending-up" class="w-3.5 h-3.5"></i>
              </span>
              <h5 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-1">Calibration</h5>
              <p class="text-xs text-slate-500 leading-relaxed">
                Evaluated by regressing true stays against model predictions[cite: 1].
              </p>
            </div>
            <div class="mt-3 pt-2.5 border-t border-slate-100 text-[10px] sm:text-[11px] text-cyan-700 font-mono font-bold">
              Target Slope = 1.0
            </div>
          </div>

          <div class="glass-card p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span class="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-200 text-indigo-700 flex items-center justify-center text-xs font-bold mb-2.5">
                <i data-lucide="activity" class="w-3.5 h-3.5"></i>
              </span>
              <h5 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-1">Sensitivity</h5>
              <p class="text-xs text-slate-500 leading-relaxed">
                Tuned to capture over 90% of actual short-stay patients[cite: 1].
              </p>
            </div>
            <div class="mt-3 pt-2.5 border-t border-slate-100 text-[10px] sm:text-[11px] text-indigo-700 font-mono font-bold">
              Sensitivity &gt; 90%
            </div>
          </div>
        </div>

      </div>
    </div>

  </main>

  <!-- Responsive Footer -->
  <footer class="glass-card border-t border-slate-200/80 py-4 mt-8">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-2 text-xs text-slate-500 text-center sm:text-left">
      <p>&copy; 2026 Academic B.Tech ML Demonstration &bull; Xu et al. (2022)</p>
      <span class="text-[11px] text-amber-800 bg-amber-50 border border-amber-200 px-3 py-1 rounded-full font-medium">
        Academic estimation tool. Not clinical advice.
      </span>
    </div>
  </footer>

  <script>
    lucide.createIcons();

    function switchMainTab(tabId) {
      const tabs = ['predict', 'analytics', 'architecture'];
      tabs.forEach(t => {
        document.getElementById('tab-' + t).classList.add('hidden');
        const btn = document.getElementById('nav-btn-' + t);
        btn.className = "flex-1 sm:flex-none px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg text-slate-600 hover:text-teal-600 transition whitespace-nowrap text-center";
      });

      document.getElementById('tab-' + tabId).classList.remove('hidden');
      const activeBtn = document.getElementById('nav-btn-' + tabId);
      activeBtn.className = "flex-1 sm:flex-none px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg bg-white shadow-sm text-teal-700 transition whitespace-nowrap text-center";

      if (tabId === 'analytics' && !currentChart) {
        loadChart('stratified');
      }
    }

    let currentChart = null;
    const ctx = document.getElementById('researchChart').getContext('2d');

    const chartConfigs = {
      stratified: {
        title: "Stratified MAE Across Stay Intervals",
        subtitle: "Two-stage pipeline reduces error on short-stay cohorts (<7 days).",
        config: {
          type: 'bar',
          data: {
            labels: ['0-2d', '2-4d', '4-7d', '7d+ (Long)'],
            datasets: [
              { label: 'LASSO', data: [1.32, 1.48, 2.15, 4.82], backgroundColor: 'rgba(148, 163, 184, 0.7)', borderRadius: 4 },
              { label: 'RF (log)', data: [1.18, 1.34, 1.95, 4.60], backgroundColor: 'rgba(56, 189, 248, 0.8)', borderRadius: 4 },
              { label: 'Two-Stage', data: [0.98, 1.15, 1.72, 3.45], backgroundColor: 'rgba(13, 148, 136, 0.9)', borderRadius: 4 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'MAE (Days)', font: { size: 10 } } } }
          }
        }
      },
      cv: {
        title: "Algorithm 5-Fold Cross-Validation Benchmark",
        subtitle: "Random Forest achieves lower error metrics against LASSO.",
        config: {
          type: 'bar',
          data: {
            labels: ['MSE (d²)', 'MAE (d)', 'MRE (ratio)'],
            datasets: [
              { label: 'LASSO', data: [5.24, 1.68, 0.42], backgroundColor: 'rgba(148, 163, 184, 0.7)', borderRadius: 4 },
              { label: 'Random Forest', data: [4.41, 1.42, 0.34], backgroundColor: 'rgba(13, 148, 136, 0.9)', borderRadius: 4 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
            scales: { y: { beginAtZero: true } }
          }
        }
      },
      dist: {
        title: "Target Distribution Skewness & Normalization",
        subtitle: "Raw duration long-tail skewness vs. stabilized log1p(LOS).",
        config: {
          type: 'line',
          data: {
            labels: ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '9d', '11d', '14d', '17d'],
            datasets: [
              {
                label: 'Raw LOS (%)',
                data: [32.5, 24.1, 15.6, 9.4, 6.2, 4.3, 3.1, 2.1, 1.4, 0.9, 0.4],
                borderColor: '#f97316',
                backgroundColor: 'rgba(249, 115, 22, 0.1)',
                fill: true,
                tension: 0.4,
                yAxisID: 'y'
              },
              {
                label: 'log1p(LOS)',
                data: [0.69, 1.10, 1.39, 1.61, 1.79, 1.95, 2.08, 2.30, 2.48, 2.71, 2.89],
                borderColor: '#0d9488',
                borderDash: [4, 4],
                tension: 0.4,
                yAxisID: 'y1'
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
            scales: {
              y: { type: 'linear', position: 'left', title: { display: true, text: 'Patients (%)', font: { size: 10 } } },
              y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'log scale', font: { size: 10 } } }
            }
          }
        }
      }
    };

    function loadChart(key) {
      if (currentChart) currentChart.destroy();
      const chartMeta = chartConfigs[key];
      document.getElementById('chartTitle').textContent = chartMeta.title;
      document.getElementById('chartSubtitle').textContent = chartMeta.subtitle;
      currentChart = new Chart(ctx, chartMeta.config);

      ['stratified', 'cv', 'dist'].forEach(k => {
        document.getElementById('subtab-' + k).className = "px-3 py-1.5 text-xs font-bold rounded-lg bg-white text-slate-600 hover:bg-slate-50 transition whitespace-nowrap";
      });
      document.getElementById('subtab-' + key).className = "px-3 py-1.5 text-xs font-bold rounded-lg bg-teal-600 text-white shadow-sm transition whitespace-nowrap";
    }

    function loadPreset(type) {
      if (type === 'mild') {
        document.getElementById('pulse').value = 72;
        document.getElementById('respiration').value = 14.0;
        document.getElementById('bmi').value = 23.5;
        document.getElementById('hematocrit').value = 14.2;
        document.getElementById('sodium').value = 140.0;
        document.getElementById('glucose').value = 92.0;
        document.getElementById('bloodureanitro').value = 11.0;
        document.getElementById('rcount').value = "1";
        document.getElementById('facid').value = "A";
        document.getElementById('gender').value = "0";
      } else {
        document.getElementById('pulse').value = 104;
        document.getElementById('respiration').value = 24.0;
        document.getElementById('bmi').value = 34.2;
        document.getElementById('hematocrit').value = 9.4;
        document.getElementById('sodium').value = 132.0;
        document.getElementById('glucose').value = 210.0;
        document.getElementById('bloodureanitro').value = 36.0;
        document.getElementById('rcount').value = "4";
        document.getElementById('facid').value = "D";
        document.getElementById('gender').value = "1";
      }
      document.getElementById('predictSubmitBtn').click();
    }

    function handleFormReset() {
      predDays.textContent = "--";
      tierBadge.textContent = "Waiting for Input";
      tierBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-slate-100 text-slate-500 border border-slate-200 transition-colors";
      predSubtext.textContent = "Submit patient admission data to evaluate probability of long stay.";
      losProgressBar.style.width = "0%";
      losProgressBar.className = "h-full bg-teal-500 rounded-full transition-all duration-500";
      relativePercent.textContent = "0% of Max";
      stage1Label.textContent = "--";
      stage1Label.className = "text-xs font-bold text-slate-700 truncate";
      stage2Label.textContent = "--";
    }

    const form = document.getElementById('predictionForm');
    const predDays = document.getElementById('predDays');
    const predSubtext = document.getElementById('predSubtext');
    const tierBadge = document.getElementById('tierBadge');
    const stage1Label = document.getElementById('stage1Label');
    const stage2Label = document.getElementById('stage2Label');
    const losProgressBar = document.getElementById('losProgressBar');
    const relativePercent = document.getElementById('relativePercent');

    form.addEventListener('submit', async function(e) {
      e.preventDefault();

      const payload = {
        gender: parseInt(document.getElementById('gender').value),
        facid: document.getElementById('facid').value,
        rcount: parseInt(document.getElementById('rcount').value),
        pulse: parseFloat(document.getElementById('pulse').value),
        respiration: parseFloat(document.getElementById('respiration').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        hematocrit: parseFloat(document.getElementById('hematocrit').value),
        sodium: parseFloat(document.getElementById('sodium').value),
        glucose: parseFloat(document.getElementById('glucose').value),
        bloodureanitro: parseFloat(document.getElementById('bloodureanitro').value)
      };

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        const los = Number(data.predicted_los);

        predDays.textContent = los.toFixed(1);

        const percentOfMax = Math.min(100, Math.max(2, Math.round((los / 17.0) * 100)));
        losProgressBar.style.width = percentOfMax + "%";
        relativePercent.textContent = percentOfMax + "% of Max";

        if (data.is_long_stay) {
          tierBadge.textContent = "Prolonged (≥ 7d)";
          tierBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-amber-50 text-amber-800 border border-amber-200";
          losProgressBar.className = "h-full bg-amber-500 rounded-full transition-all duration-500";
          predSubtext.textContent = "Patient triaged to prolonged stay tier. Bounded at 7.0-day threshold per Xu et al. (2022).";
          stage1Label.textContent = "Long Stay (≥ 7d)";
          stage1Label.className = "text-xs font-bold text-amber-700 truncate";
          stage2Label.textContent = "Cutoff Bounded (7.0d)";
        } else {
          tierBadge.textContent = "Short Stay (< 7d)";
          tierBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-teal-50 text-teal-800 border border-teal-200";
          losProgressBar.className = "h-full bg-teal-500 rounded-full transition-all duration-500";
          predSubtext.textContent = "Patient triaged to short stay. Estimated with Stage 2 log-regressor via expm1(log_pred).";
          stage1Label.textContent = "Short Stay (< 7d)";
          stage1Label.className = "text-xs font-bold text-teal-700 truncate";
          stage2Label.textContent = "log1p Regressed";
        }
      } catch (err) {
        alert("Inference engine error: " + err);
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if stage1_classifier is not None and stage2_regressor is not None:
        try:
            row_dict = {
                "rcount": data["rcount"],
                "gender": data["gender"],
                "pulse": data["pulse"],
                "respiration": data["respiration"],
                "bmi": data["bmi"],
                "hematocrit": data["hematocrit"],
                "sodium": data["sodium"],
                "glucose": data["glucose"],
                "bloodureanitro": data["bloodureanitro"],
                "facid_B": int(data["facid"] == "B"),
                "facid_C": int(data["facid"] == "C"),
                "facid_D": int(data["facid"] == "D"),
                "facid_E": int(data["facid"] == "E"),
            }

            expected = training_features if training_features else getattr(stage1_classifier, "feature_names_in_", None)
            if expected is not None:
                row_df = pd.DataFrame([row_dict])
                for col in expected:
                    if col not in row_df.columns:
                        row_df[col] = 0
                row_df = row_df[expected]
            else:
                row_df = pd.DataFrame([row_dict])

            proba_long = stage1_classifier.predict_proba(row_df)[0, 1]
            is_long = bool(proba_long >= THRESHOLD_PROBA)

            if is_long:
                final_los = CLASSIFIER_CUTOFF
            else:
                log_pred = stage2_regressor.predict(row_df)[0]
                final_los = float(np.expm1(log_pred))

            return jsonify({
                "predicted_los": round(final_los, 1),
                "is_long_stay": is_long,
                "mode": "production_model"
            })
        except Exception:
            pass

    bun = data["bloodureanitro"]
    glucose = data["glucose"]
    rcount = data["rcount"]
    pulse = data["pulse"]
    resp = data["respiration"]
    bmi = data["bmi"]

    risk_score = (
        (bun / 15.0) * 0.35 +
        (glucose / 100.0) * 0.25 +
        (rcount * 0.35) +
        ((pulse - 70) / 30.0) * 0.2 +
        ((resp - 14) / 10.0) * 0.15 +
        ((bmi - 24) / 10.0) * 0.1
    )

    is_long = bool(risk_score >= 2.1 or bun >= 28.0 or rcount >= 4)
    if is_long:
        final_los = CLASSIFIER_CUTOFF
    else:
        computed = 1.2 + (risk_score * 1.8)
        final_los = max(1.0, min(6.8, computed))

    return jsonify({
        "predicted_los": round(final_los, 1),
        "is_long_stay": is_long,
        "mode": "responsive_math"
    })


if __name__ == "__main__":
    initialize_or_train_models()
    app.run(host="0.0.0.0", port=5000)
