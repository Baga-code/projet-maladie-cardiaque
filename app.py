# ============================================
# APPLICATION STREAMLIT - DETECTION MALADIE CARDIAQUE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================

st.set_page_config(
    page_title="Detection Maladie Cardiaque",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONNALISE
# ============================================

st.markdown("""
<style>
    /* Fond general */
    .main { background-color: #0f1117; }
    
    /* Titre principal */
    .titre-principal {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Sous-titre */
    .sous-titre {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Carte metrique */
    .carte-metrique {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e74c3c33;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .carte-metrique h3 {
        color: #888;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .carte-metrique h2 {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0 0 0;
    }
    
    /* Carte resultat positif */
    .resultat-danger {
        background: linear-gradient(135deg, #c0392b22, #e74c3c11);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .resultat-danger h2 {
        color: #e74c3c;
        font-size: 1.5rem;
        margin: 0;
    }
    
    /* Carte resultat negatif */
    .resultat-ok {
        background: linear-gradient(135deg, #27ae6022, #2ecc7111);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .resultat-ok h2 {
        color: #2ecc71;
        font-size: 1.5rem;
        margin: 0;
    }
    
    /* Section titre */
    .section-titre {
        color: #e74c3c;
        font-size: 1.3rem;
        font-weight: 700;
        border-left: 4px solid #e74c3c;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Badge modele */
    .badge-meilleur {
        background: #e74c3c;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    /* Separateur */
    .separateur {
        border: none;
        border-top: 1px solid #333;
        margin: 1.5rem 0;
    }
    
    /* Sidebar */
    .css-1d391kg { background-color: #1a1a2e; }
    
    /* Bouton */
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px #e74c3c44;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ============================================

@st.cache_data
def charger_et_entrainer():
    colonnes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak',
                'slope', 'ca', 'thal', 'target']
    df = pd.read_csv('processed.cleveland.data',
                     names=colonnes, na_values='?')
    df = df.dropna()
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modeles = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN'                : KNeighborsClassifier(n_neighbors=5),
        'SVM'                : SVC(random_state=42, probability=True),
        'Decision Tree'      : DecisionTreeClassifier(random_state=42),
        'Random Forest'      : RandomForestClassifier(random_state=42, n_estimators=100),
        'AdaBoost'           : AdaBoostClassifier(random_state=42, n_estimators=100)
    }

    resultats = {}
    for nom, modele in modeles.items():
        modele.fit(X_train_s, y_train)
        y_pred = modele.predict(X_test_s)
        y_proba = modele.predict_proba(X_test_s)[:, 1]
        resultats[nom] = {
            'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Rappel'   : round(recall_score(y_test, y_pred), 4),
            'F1-Score' : round(f1_score(y_test, y_pred), 4),
            'AUC-ROC'  : round(roc_auc_score(y_test, y_proba), 4)
        }

    return modeles, scaler, resultats, df

modeles, scaler, resultats, df = charger_et_entrainer()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🫀</div>
        <h2 style='color:#e74c3c; margin:0;'>CardioAI</h2>
        <p style='color:#888; font-size:0.8rem;'>Detection de Maladie Cardiaque</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigation",
        ["Prediction", "Comparaison des modeles", "Donnees", "A propos"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#666; font-size:0.75rem; text-align:center;'>
        <p>Dataset: Heart Disease UCI</p>
        <p>297 patients | 13 features</p>
        <p>Meilleur modele: KNN</p>
        <p>AUC-ROC: 94.92%</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# EN-TETE
# ============================================

st.markdown('<h1 class="titre-principal">Detection de Maladie Cardiaque</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sous-titre">IFOAD | Intelligence Artificielle | Dr Arthur Sawadogo</p>',
            unsafe_allow_html=True)

# ============================================
# PAGE 1 : PREDICTION
# ============================================

if menu == "Prediction":

    st.markdown('<p class="section-titre">Informations du patient</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informations generales**")
        age = st.slider("Age", 20, 80, 54)
        sex = st.selectbox("Sexe",
                           [0, 1],
                           format_func=lambda x: "Femme" if x == 0 else "Homme")
        cp = st.selectbox("Type de douleur thoracique",
                          [1, 2, 3, 4],
                          format_func=lambda x: {
                              1: "Type 1 - Angine typique",
                              2: "Type 2 - Angine atypique",
                              3: "Type 3 - Douleur non-anginale",
                              4: "Type 4 - Asymptomatique"
                          }[x])
        trestbps = st.slider("Pression arterielle au repos (mm Hg)", 80, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 246)

    with col2:
        st.markdown("**Examens cliniques**")
        fbs = st.selectbox("Glycemie a jeun > 120 mg/dl",
                           [0, 1],
                           format_func=lambda x: "Non" if x == 0 else "Oui")
        restecg = st.selectbox("Resultats ECG au repos",
                               [0, 1, 2],
                               format_func=lambda x: {
                                   0: "Normal",
                                   1: "Anomalie ST-T",
                                   2: "Hypertrophie ventriculaire"
                               }[x])
        thalach = st.slider("Frequence cardiaque maximale", 60, 210, 150)
        exang = st.selectbox("Angine induite par exercice",
                             [0, 1],
                             format_func=lambda x: "Non" if x == 0 else "Oui")

    with col3:
        st.markdown("**Parametres avances**")
        oldpeak = st.slider("Depression ST (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Pente segment ST",
                             [1, 2, 3],
                             format_func=lambda x: {
                                 1: "Ascendante",
                                 2: "Plate",
                                 3: "Descendante"
                             }[x])
        ca = st.selectbox("Nombre de vaisseaux majeurs", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemie",
                            [3, 6, 7],
                            format_func=lambda x: {
                                3: "Normal",
                                6: "Defaut fixe",
                                7: "Defaut reversible"
                            }[x])

    st.markdown('<hr class="separateur">', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict = st.button("Lancer la prediction")

    if predict:
        donnees = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])
        donnees_norm = scaler.transform(donnees)
        modele_knn = modeles['KNN']
        prediction = modele_knn.predict(donnees_norm)[0]
        proba = modele_knn.predict_proba(donnees_norm)[0]

        st.markdown('<p class="section-titre">Resultat de la prediction</p>',
                    unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns([1, 1, 1])

        with col_r1:
            if prediction == 1:
                st.markdown(f"""
                <div class="resultat-danger">
                    <div style='font-size:3rem;'>⚠️</div>
                    <h2>Risque detecte</h2>
                    <p style='color:#aaa;'>Maladie cardiaque probable</p>
                    <h1 style='color:#e74c3c; font-size:3rem; margin:0;'>
                        {proba[1]*100:.1f}%
                    </h1>
                    <p style='color:#888;'>de probabilite</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="resultat-ok">
                    <div style='font-size:3rem;'>✅</div>
                    <h2>Pas de risque detecte</h2>
                    <p style='color:#aaa;'>Pas de maladie cardiaque</p>
                    <h1 style='color:#2ecc71; font-size:3rem; margin:0;'>
                        {proba[0]*100:.1f}%
                    </h1>
                    <p style='color:#888;'>de probabilite</p>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            categories = ['Sans maladie', 'Avec maladie']
            valeurs = [proba[0]*100, proba[1]*100]
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(categories, valeurs, color=colors,
                          width=0.5, edgecolor='none')
            ax.set_ylabel('Probabilite (%)', color='#888')
            ax.set_ylim(0, 110)
            ax.tick_params(colors='#888')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, valeurs):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + 2, f'{val:.1f}%',
                        ha='center', color='white', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_r3:
            st.markdown("""
            <div class="carte-metrique">
                <h3>Modele utilise</h3>
                <h2>KNN</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="carte-metrique">
                <h3>Accuracy</h3>
                <h2>88.33%</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="carte-metrique">
                <h3>AUC-ROC</h3>
                <h2>94.92%</h2>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE 2 : COMPARAISON DES MODELES
# ============================================

elif menu == "Comparaison des modeles":

    st.markdown('<p class="section-titre">Performances des 6 algorithmes</p>',
                unsafe_allow_html=True)

    df_res = pd.DataFrame(resultats).T

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    metriques = ['Accuracy', 'Precision', 'Rappel', 'F1-Score', 'AUC-ROC']
    cols = [col_m1, col_m2, col_m3, col_m4, col_m5]

    for col, metrique in zip(cols, metriques):
        meilleur = df_res[metrique].idxmax()
        valeur = df_res[metrique].max()
        with col:
            st.markdown(f"""
            <div class="carte-metrique">
                <h3>{metrique}</h3>
                <h2>{valeur:.4f}</h2>
                <p style='color:#666; font-size:0.8rem; margin:0;'>{meilleur}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="separateur">', unsafe_allow_html=True)
    st.markdown('<p class="section-titre">Tableau comparatif</p>',
                unsafe_allow_html=True)

    st.dataframe(
        df_res.style
        .highlight_max(axis=0, color='#1a4a1a')
        .highlight_min(axis=0, color='#4a1a1a')
        .format("{:.4f}"),
        use_container_width=True,
        height=250
    )

    st.markdown('<p class="section-titre">Visualisation</p>',
                unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        x = np.arange(len(df_res))
        width = 0.15
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (metrique, color) in enumerate(zip(metriques, colors)):
            ax.bar(x + i*width, df_res[metrique],
                   width=width, label=metrique, color=color, alpha=0.85)
        ax.set_xticks(x + width*2)
        ax.set_xticklabels(df_res.index, rotation=30, ha='right', color='#888')
        ax.set_ylabel('Score', color='#888')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.8, color='#666', linestyle='--', alpha=0.5)
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
        ax.set_title('Toutes les metriques', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        auc_sorted = df_res['AUC-ROC'].sort_values()
        colors_auc = ['#e74c3c' if v < 0.9 else '#2ecc71' for v in auc_sorted]
        bars = ax.barh(auc_sorted.index, auc_sorted.values,
                       color=colors_auc, edgecolor='none', height=0.5)
        ax.axvline(x=0.9, color='#f39c12', linestyle='--',
                   alpha=0.7, label='Seuil 0.9')
        ax.set_xlim(0, 1.15)
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('AUC-ROC par modele', color='white', fontweight='bold')
        ax.legend(facecolor='#16213e', labelcolor='white')
        for bar, val in zip(bars, auc_sorted):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', color='white',
                    fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================
# PAGE 3 : DONNEES
# ============================================

elif menu == "Donnees":

    st.markdown('<p class="section-titre">Apercu du dataset</p>',
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown("""
        <div class="carte-metrique">
            <h3>Total patients</h3>
            <h2>297</h2>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="carte-metrique">
            <h3>Variables</h3>
            <h2>13</h2>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        malades = int(df['target'].sum())
        st.markdown(f"""
        <div class="carte-metrique">
            <h3>Malades</h3>
            <h2>{malades}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        sains = int((df['target'] == 0).sum())
        st.markdown(f"""
        <div class="carte-metrique">
            <h3>Sains</h3>
            <h2>{sains}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="separateur">', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

# ============================================
# PAGE 4 : A PROPOS
# ============================================

elif menu == "A propos":

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                    border: 1px solid #e74c3c33;
                    border-radius: 12px; padding: 2rem;'>
            <h2 style='color:#e74c3c;'>A propos du projet</h2>
            <p style='color:#aaa;'>
                Ce projet a ete realise dans le cadre du cours d Intelligence 
                Artificielle de l IFOAD, sous la direction du Dr Arthur Sawadogo.
            </p>
            <hr style='border-color:#333;'>
            <h3 style='color:#fff;'>Dataset</h3>
            <p style='color:#aaa;'>
                Heart Disease UCI - Cleveland Database<br>
                303 patients originaux, 297 apres nettoyage<br>
                14 variables dont 1 variable cible
            </p>
            <hr style='border-color:#333;'>
            <h3 style='color:#fff;'>Algorithmes</h3>
            <ul style='color:#aaa;'>
                <li>Logistic Regression</li>
                <li>K-Nearest Neighbors (KNN)</li>
                <li>Support Vector Machine (SVM)</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>AdaBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                    border: 1px solid #e74c3c33;
                    border-radius: 12px; padding: 2rem;'>
            <h2 style='color:#e74c3c;'>Resultats</h2>
            <hr style='border-color:#333;'>
            <h3 style='color:#2ecc71;'>Meilleur modele : KNN</h3>
            <p style='color:#aaa;'>
                KNN est le modele le plus adapte pour ce contexte medical 
                car il combine le meilleur Rappel et la meilleure Precision,
                minimisant ainsi le risque de laisser partir un patient 
                malade sans traitement.
            </p>
            <hr style='border-color:#333;'>
            <table style='width:100%; color:#aaa;'>
                <tr>
                    <td style='padding:0.4rem 0;'>Accuracy</td>
                    <td style='color:#e74c3c; font-weight:bold;'>88.33%</td>
                </tr>
                <tr>
                    <td style='padding:0.4rem 0;'>Precision</td>
                    <td style='color:#e74c3c; font-weight:bold;'>92.00%</td>
                </tr>
                <tr>
                    <td style='padding:0.4rem 0;'>Rappel</td>
                    <td style='color:#e74c3c; font-weight:bold;'>82.14%</td>
                </tr>
                <tr>
                    <td style='padding:0.4rem 0;'>F1-Score</td>
                    <td style='color:#e74c3c; font-weight:bold;'>86.79%</td>
                </tr>
                <tr>
                    <td style='padding:0.4rem 0;'>AUC-ROC</td>
                    <td style='color:#e74c3c; font-weight:bold;'>94.92%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)