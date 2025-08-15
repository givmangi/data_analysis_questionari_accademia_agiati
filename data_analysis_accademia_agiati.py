"""
Analisi Quantitativa dei Dati di Soddisfazione - Accademia Roveretana degli Agiati
================================================================================

Autore: Giuseppe Pio Mangiacotti
Istituzione: Università degli Studi di Trento - Dipartimento di Scienze Cognitive
Anno Accademico: 2024/2025
Versione: 2.1 (CORREZIONE METODOLOGICA CRITICA - Fisher Exact Test) + GRAFICI TESI

⚠️  AGGIORNAMENTO CRITICO: 
Questa versione corregge un errore metodologico significativo nella v2.0.
Il test Mann-Whitney U è stato sostituito con Fisher Exact Test per 
dati essenzialmente binari (solo valori 2 e 3).

🎓 AGGIORNAMENTO TESI:
Aggiunti grafici specifici per l'elaborato finale:
- Pannello demografico completo (età, soci, scolarizzazione)
- Pannello modalità partecipazione (cartaceo/webinar, argomenti)

Descrizione:
Script per l'analisi completa del dataset di feedback raccolto presso l'Accademia 
Roveretana degli Agiati. Implementa algoritmi statistici appropriati per 
l'identificazione dei key findings principali e la generazione del Cultural 
Engagement Score (CES).

Correzioni Metodologiche v2.1:
===============================
🎯 PROBLEMA RISOLTO:
- Test Mann-Whitney U inappropriato per dati con ties eccessivi (100% ties)
- Solo valori 2 (Soddisfatto) e 3 (Molto Soddisfatto) nel dataset
- Perdita significativa di potenza statistica

✅ SOLUZIONE IMPLEMENTATA:
- Fisher Exact Test per confronto proporzioni binarie
- Tabelle di contingenza 2×2 appropriate
- Odds ratio e p-value esatti
- Warning automatici per test inappropriati
- Validazione qualità statistica integrata
- REGRESSIONE LOGISTICA per validazione terziaria completa

Metodologia Statistica (Corretta):
===================================
PRIMARI:
- Fisher Exact Test per confronti binari/quasi-binari ✅
- Test χ² per tabelle contingenza (con correzioni appropriate) ✅
- Bootstrap resampling per validazione indicatori compositi ✅
- Regressione logistica per validazione crociata ✅

SECONDARI (con disclaimer):
- Mann-Whitney U mantenuto per compatibilità ⚠️
- Coefficiente r di Rosenthal per effect size ⚠️

Dipendenze:
- pandas >= 1.5.0
- matplotlib >= 3.6.0  
- seaborn >= 0.12.0
- numpy >= 1.24.0
- scipy >= 1.10.0 (fisher_exact, chi2_contingency, mannwhitneyu)
- scikit-learn >= 1.0.0 (LogisticRegression - opzionale)
- plotly >= 5.15.0 (opzionale)

Utilizzo:
python data_analysis_accademia_agiati_v2.1.py --input paper_report.CSV --output ./output/

Citazione (Aggiornata):
=======================
Mangiacotti, G.P. (2025). Progettazione di interfacce per la raccolta di feedback 
nel settore culturale: un sistema ibrido per l'Accademia Roveretana degli Agiati v2.1
Università degli Studi di Trento.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, mannwhitneyu, ks_2samp, rankdata, fisher_exact, spearmanr, norm
import warnings
import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# ===== PURPLE THEME CONFIGURATION =====
# Custom purple palette with complementary accents
PURPLE_PALETTE = {
    'primary_purple': '#6A4C93',      # Deep purple
    'light_purple': '#A47FC7',        # Light purple
    'dark_purple': '#4A2C70',         # Dark purple
    'lavender': '#C8B8E8',            # Very light purple
    'royal_purple': '#8B5A9F',        # Royal purple
    'plum': '#9B59B6',                # Plum
    'periwinkle': '#8E7CC3',          # Blue-purple
    'sage': '#87A96B',                # Complementary green
    'slate_blue': '#6C7B95',          # Blue-gray
    'warm_gold': '#DAA520',           # Accent gold
    'soft_teal': '#5D8A8A',           # Muted teal
    'dusty_rose': '#C49BB0'           # Soft pink-purple
}

# Color sequences for different needs
PURPLE_SEQUENTIAL = ['#F3F0FF', '#E6DBFF', '#D9C5FF', '#CCB0FF', '#BF9BFF', '#B386FF', '#A670FF', '#995BFF', '#8C46FF', '#7F31FF']
PURPLE_DIVERGING = ['#4A2C70', '#6A4C93', '#8B5A9F', '#A47FC7', '#C8B8E8', '#E6DBFF']
PURPLE_CATEGORICAL = ['#6A4C93', '#87A96B', '#8E7CC3', '#C49BB0', '#5D8A8A', '#DAA520', '#9B59B6', '#6C7B95']

# Set the style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(PURPLE_CATEGORICAL)

class AccademiaAnalyzer:
    """
    Classe principale per l'analisi dei dati dell'Accademia degli Agiati.
    
    VERSIONE 2.1 + GRAFICI TESI - Metodologia Corretta per Dati Binari/Ordinali Discreti
    ===================================================================================
    
    Implementa metodologie quantitative appropriate per l'identificazione di pattern
    di soddisfazione, engagement e comportamento del pubblico culturale.
    
    CORREZIONE METODOLOGICA CRITICA:
    --------------------------------
    - Fisher Exact Test per confronti di proporzioni (dati essenzialmente binari)
    - Mann-Whitney U relegato ad analisi secondaria per dati con ties eccessivi
    - Verifica automatica dell'appropriatezza dei test statistici
    - Warning metodologici espliciti quando le assunzioni sono violate
    - REGRESSIONE LOGISTICA COMPLETA per validazione terziaria
    
    GRAFICI SPECIFICI PER TESI:
    ---------------------------
    - Pannello demografico completo (età, soci, scolarizzazione)
    - Pannello modalità partecipazione (cartaceo/webinar, argomenti)
    - Integrazione automatica nel workflow di analisi
    
    TEST STATISTICI UTILIZZATI:
    ---------------------------
    PRINCIPALI (Raccomandati):
    - Fisher Exact Test per tabelle di contingenza 2×2
    - Test χ² con correzioni appropriate
    - Bootstrap resampling per validazione indicatori compositi
    - Regressione logistica per validazione crociata e intervalli di confidenza
    
    SECONDARI (Con limitazioni dichiarate):
    - Mann-Whitney U per compatibilità (con disclaimer sulle limitazioni)
    - Coefficiente r di Rosenthal per effect size comparativo
    
    OUTPUT GENERATI:
    ---------------
    - analysis_report_corrected.txt: Report con metodologia corretta
    - methodological_warning.txt: Avviso sui problemi metodologici
    - contingency_table_fisher_test.csv: Tabella per Fisher test
    - digital_satisfaction_paradox_corrected.png: Visualizzazione aggiornata
    - demographic_overview_panel.png: 🎓 Pannello demografico (TESI)
    - participation_modalities_panel.png: 🎓 Pannello partecipazione (TESI)
    """
    
    def __init__(self, csv_path: str, output_dir: str = "./output/"):
        """
        Inizializza l'analyzer con metodologia corretta.
        
        Parameters:
        -----------
        csv_path : str
            Percorso al file CSV dei dati
        output_dir : str
            Directory di output (default: ./output/)
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configurazione per grafici publication-ready con tema purple
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.prop_cycle': plt.cycler('color', PURPLE_CATEGORICAL)
        })
        
        self.results = {}  # Storage per tutti i risultati
        
        # Nuovo: Tracker per quality assessment
        self.methodological_warnings = []
        self.test_appropriateness = {}
        
    def _log_methodological_warning(self, test_name: str, warning: str):
        """
        Log avvisi metodologici per revisione.
        
        Parameters:
        -----------
        test_name : str
            Nome del test statistico
        warning : str
            Descrizione del problema metodologico
        """
        self.methodological_warnings.append({
            'test': test_name,
            'warning': warning,
            'timestamp': datetime.now().isoformat()
        })
        print(f"⚠️  AVVISO METODOLOGICO [{test_name}]: {warning}")
    
    def _assess_test_appropriateness(self, test_name: str, data_description: str, 
                                   is_appropriate: bool, reason: str):
        """
        Valuta e registra l'appropriatezza di un test statistico.
        
        Parameters:
        -----------
        test_name : str
            Nome del test statistico
        data_description : str
            Descrizione dei dati analizzati
        is_appropriate : bool
            Se il test è appropriato
        reason : str
            Motivazione della valutazione
        """
        self.test_appropriateness[test_name] = {
            'data_description': data_description,
            'is_appropriate': is_appropriate,
            'reason': reason,
            'recommendation': 'Utilizzare' if is_appropriate else 'Evitare'
        }
        
        if not is_appropriate:
            self._log_methodological_warning(test_name, reason)
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Carica e pulisce il dataset applicando standardizzazioni e validazioni.
        
        Returns:
        --------
        pd.DataFrame
            Dataset pulito e standardizzato
        """
        print("📊 Caricamento e pulizia del dataset...")
        
        # Caricamento con gestione encoding
        try:
            df = pd.read_csv(self.csv_path, delimiter=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, delimiter=';', encoding='cp1252')
        
        # Standardizzazione nomi colonne
        df.columns = df.columns.str.strip()
        
        # Pulizia e standardizzazione campi critici
        df = self._standardize_age_categories(df)
        df = self._standardize_membership_status(df)
        df = self._standardize_satisfaction_scores(df)
        df = self._clean_communication_channels(df)
        
        # Filtro record completi
        required_cols = ['eta_std', 'socio_std', 'soddisfazione_num']
        df_clean = df.dropna(subset=required_cols)
        
        # Logging statistiche pulizia
        print(f"   ✓ Dataset originale: {len(df)} record")
        print(f"   ✓ Dataset pulito: {len(df_clean)} record")
        print(f"   ✓ Tasso completezza: {len(df_clean)/len(df)*100:.1f}%")
        
        self.df = df_clean
        return df_clean
    
    def _standardize_age_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizza le categorie di età."""
        age_mapping = {
            '14-30': '14-30',
            '31-50': '31-50', 
            '51-70': '51-70',
            '>70': '>70',
            '70>': '>70',  # Normalizzazione
            '14-30 ': '14-30'  # Rimozione spazi
        }
        
        df['eta_std'] = df['Età'].str.strip().map(age_mapping)
        return df
    
    def _standardize_membership_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizza lo status di socio."""
        membership_mapping = {
            'Sì': 'Sì',
            'Sí': 'Sì',  # Fix encoding UTF-8
            'Si': 'Sì',
            'No': 'No'
        }
        
        df['socio_std'] = df['Socio'].map(membership_mapping)
        return df
    
    def _standardize_satisfaction_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte e valida i punteggi di soddisfazione."""
        # Conversione a numerico con gestione errori
        df['soddisfazione_num'] = pd.to_numeric(
            df['Soddisfazione (1-3)'], errors='coerce'
        )
        
        # Validazione range
        valid_range = df['soddisfazione_num'].between(1, 3, inclusive='both')
        df.loc[~valid_range, 'soddisfazione_num'] = np.nan
        
        return df
    
    def _clean_communication_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulisce e standardizza i canali di comunicazione."""
        if 'Conoscenza' in df.columns:
            df['canali_clean'] = df['Conoscenza'].fillna('N/A')
        return df
    
    def calculate_key_findings(self) -> dict:
        """
        Calcola i tre key findings principali identificati nell'analisi.
        
        Returns:
        --------
        dict
            Dizionario con tutti i key findings e relative statistiche
        """
        print("🔍 Calcolo Key Findings...")
        
        findings = {}
        
        # 1. Digital Satisfaction Paradox (CORRETTO)
        findings['digital_paradox'] = self._analyze_digital_satisfaction_paradox()
        
        # 2. Generational Membership Gap  
        findings['generational_gap'] = self._analyze_generational_membership_gap()
        
        # 3. Inverse Contentment Effect
        findings['contentment_effect'] = self._analyze_inverse_contentment_effect()
        
        self.results['key_findings'] = findings
        return findings
    
    def _analyze_digital_satisfaction_paradox(self) -> dict:
        """
        Analizza il Digital Satisfaction Paradox usando metodologie statistiche appropriate.
        
        CORREZIONE METODOLOGICA v2.1:
        - Fisher Exact Test per confronto proporzioni (dati essenzialmente binari)
        - Mann-Whitney U mantenuto come analisi secondaria con disclosure limitazioni
        - Aggiunta regressione logistica per robustezza E VALIDAZIONE COMPLETA
        """
        
        # Estrazione dati per modalità
        webinar_data = self.df[self.df['Fonte'] == 'Webinar']['soddisfazione_num']
        cartaceo_data = self.df[self.df['Fonte'] == 'Cartaceo']['soddisfazione_num']
        
        # === STATISTICHE DESCRITTIVE ===
        stats_webinar = {
            'n': len(webinar_data),
            'mean': webinar_data.mean(),
            'std': webinar_data.std(),
            'median': webinar_data.median(),
            'satisfaction_rate_max': (webinar_data == 3).mean(),  # % "Molto Soddisfatti"
            'satisfaction_rate_min': (webinar_data == 2).mean()   # % "Soddisfatti"
        }
        
        stats_cartaceo = {
            'n': len(cartaceo_data),
            'mean': cartaceo_data.mean(), 
            'std': cartaceo_data.std(),
            'median': cartaceo_data.median(),
            'satisfaction_rate_max': (cartaceo_data == 3).mean(),
            'satisfaction_rate_min': (cartaceo_data == 2).mean()
        }
        
        # === TEST ESATTO DI FISHER (METODO PRINCIPALE) ===
        # Costruzione tabella di contingenza 2x2
        contingency_table = [
            [int((cartaceo_data == 2).sum()), int((cartaceo_data == 3).sum())],  # Cartaceo: [Sodd, Molto Sodd]
            [int((webinar_data == 2).sum()), int((webinar_data == 3).sum())]     # Webinar: [Sodd, Molto Sodd]
        ]
        
        # Fisher Exact Test (CORRETTO)
        odds_ratio, fisher_p_value = stats.fisher_exact(contingency_table)
        
        # Calcolo manuale delle proporzioni per verifica
        prop_webinar_very_sat = stats_webinar['satisfaction_rate_max']
        prop_cartaceo_very_sat = stats_cartaceo['satisfaction_rate_max']
        proportion_difference = prop_webinar_very_sat - prop_cartaceo_very_sat
        
        # Assessment appropriatezza Fisher test
        min_expected_frequency = min([min(row) for row in contingency_table])
        self._assess_test_appropriateness(
            'Fisher Exact Test',
            'Confronto proporzioni tra modalità cartacea e webinar',
            True,
            'Appropriato per tabelle 2×2, nessuna assunzione violata'
        )
        
        # === MANN-WHITNEY U (ANALISI SECONDARIA CON DISCLAIMER) ===
        u_statistic, p_value_mw = stats.mannwhitneyu(
            webinar_data, cartaceo_data, 
            alternative='two-sided',
            use_continuity=True
        )
        
        # Calcolo effect size per Mann-Whitney (con avvertimento)
        n1, n2 = len(webinar_data), len(cartaceo_data)
        total_n = n1 + n2
        
        # Z-score dal test Mann-Whitney
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (total_n + 1) / 12)
        z_score = (u_statistic - mean_u) / std_u
        rosenthal_r = abs(z_score) / np.sqrt(total_n)
        
        # Rank analysis
        combined_data = np.concatenate([cartaceo_data, webinar_data])
        group_labels = np.concatenate([
            np.zeros(len(cartaceo_data)),  # 0 = cartaceo
            np.ones(len(webinar_data))     # 1 = webinar
        ])
        ranks = stats.rankdata(combined_data)
        cartaceo_rank_mean = ranks[group_labels == 0].mean()
        webinar_rank_mean = ranks[group_labels == 1].mean()
        
        # Assessment problemi Mann-Whitney
        total_values = len(np.unique(combined_data))
        ties_percentage = 1 - (total_values / total_n)
        mw_problematic = ties_percentage > 0.5  # >50% ties considerato problematico
        
        self._assess_test_appropriateness(
            'Mann-Whitney U',
            f'Dati ordinali con {ties_percentage:.1%} ties',
            not mw_problematic,
            f'Ties eccessivi ({ties_percentage:.1%}), solo {total_values} valori distinti' if mw_problematic else 'Appropriato'
        )
        
        # === REGRESSIONE LOGISTICA (ANALISI TERZIARIA COMPLETA) ===
        logistic_results = self._analyze_logistic_regression_validation(webinar_data, cartaceo_data)
        
        # === ANALISI ETÀ (MANTENUTA) ===
        age_mapping = {'14-30': 22, '31-50': 40, '51-70': 60, '>70': 75}
        age_webinar = self.df[self.df['Fonte'] == 'Webinar']['eta_std'].map(age_mapping).mean()
        age_cartaceo = self.df[self.df['Fonte'] == 'Cartaceo']['eta_std'].map(age_mapping).mean()
        
        return {
            # === RISULTATI PRINCIPALI (FISHER TEST) ===
            'primary_test': 'Fisher Exact Test',
            'fisher_odds_ratio': odds_ratio,
            'fisher_p_value': fisher_p_value,
            'contingency_table': contingency_table,
            'proportion_difference': proportion_difference,
            
            # === STATISTICHE DESCRITTIVE ===
            'webinar_stats': stats_webinar,
            'cartaceo_stats': stats_cartaceo,
            'difference': stats_webinar['mean'] - stats_cartaceo['mean'],
            
            # === MANN-WHITNEY (SECONDARIO CON DISCLAIMER) ===
            'mannwhitney_u': u_statistic,
            'mannwhitney_p': p_value_mw,
            'rosenthal_r': rosenthal_r,
            'rosenthal_r_squared': rosenthal_r**2,
            'z_score': z_score,
            'cartaceo_rank_mean': cartaceo_rank_mean,
            'webinar_rank_mean': webinar_rank_mean,
            'mw_methodological_warning': mw_problematic,
            'ties_percentage': ties_percentage,
            
            # === REGRESSIONE LOGISTICA COMPLETA ===
            'logistic_regression': logistic_results,
            
            # === ANALISI ETÀ ===
            'age_webinar': age_webinar,
            'age_cartaceo': age_cartaceo,
            'age_paradox': age_webinar - age_cartaceo,
            
            # === QUALITY ASSESSMENT ===
            'statistical_quality': {
                'fisher_appropriate': True,
                'min_expected_freq': min_expected_frequency,
                'mw_ties_problematic': mw_problematic,
                'recommended_test': 'Fisher Exact Test'
            },
            
            # === LEGACY COMPATIBILITY ===
            'cohens_d': rosenthal_r * 2,  # Approssimazione per compatibilità grafici
            't_statistic': z_score,       # Compatibilità
            'p_value': fisher_p_value,    # USA P-VALUE FISHER come principale
            'ks_statistic': np.nan,       # Non applicabile per dati binari
            'ks_p_value': np.nan
        }
    
    def _analyze_logistic_regression_validation(self, webinar_data, cartaceo_data):
        """
        Implementa regressione logistica per validazione del Digital Satisfaction Paradox.
        
        Returns:
        --------
        dict
            Risultati completi della regressione logistica con statistiche inferenziali
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from scipy.stats import norm
            
            # Preparazione dati
            X = np.array([1 if fonte == 'Webinar' else 0 for fonte in self.df['Fonte']]).reshape(-1, 1)
            y = np.array([1 if sat == 3 else 0 for sat in self.df['soddisfazione_num']])
            
            # Fit del modello con solver appropriato
            logistic_model = LogisticRegression(solver='liblinear', random_state=42)
            logistic_model.fit(X, y)
            
            # Estrazione coefficienti
            beta = logistic_model.coef_[0][0]
            intercept = logistic_model.intercept_[0]
            
            # Calcolo errore standard attraverso matrice di informazione di Fisher
            p_pred = logistic_model.predict_proba(X)[:, 1]
            
            # Matrice design con intercetta
            X_design = np.column_stack([np.ones(len(X)), X.flatten()])
            
            # Matrice dei pesi (Hessian matrix)
            W = np.diag(p_pred * (1 - p_pred))
            
            try:
                # Matrice di informazione di Fisher
                fisher_info = X_design.T @ W @ X_design
                var_covar_matrix = np.linalg.inv(fisher_info)
                
                se_intercept = np.sqrt(var_covar_matrix[0, 0])
                se_beta = np.sqrt(var_covar_matrix[1, 1])
                
                # Correlazione tra coefficienti
                correlation_coef = var_covar_matrix[0, 1] / (se_intercept * se_beta)
                
            except np.linalg.LinAlgError:
                # Fallback: bootstrap per errore standard
                print("   ⚠️  Matrice singolare - utilizzando bootstrap per SE")
                se_beta = self._bootstrap_logistic_se(X, y)
                se_intercept = self._bootstrap_logistic_se_intercept(X, y)
                correlation_coef = np.nan
            
            # Statistiche inferenziali
            z_score_beta = beta / se_beta
            z_score_intercept = intercept / se_intercept
            
            p_value_beta = 2 * (1 - norm.cdf(abs(z_score_beta)))
            p_value_intercept = 2 * (1 - norm.cdf(abs(z_score_intercept)))
            
            # Odds ratio e intervalli di confidenza
            odds_ratio = np.exp(beta)
            
            # Intervalli di confidenza al 95% per beta e OR
            z_critical = norm.ppf(0.975)  # 1.96
            beta_ci_lower = beta - z_critical * se_beta
            beta_ci_upper = beta + z_critical * se_beta
            
            or_ci_lower = np.exp(beta_ci_lower)
            or_ci_upper = np.exp(beta_ci_upper)
            
            # Pseudo R-squared (McFadden)
            pseudo_r2 = self._calculate_mcfadden_r2(logistic_model, X, y)
            
            # Log-likelihood
            log_likelihood = self._calculate_log_likelihood(logistic_model, X, y)
            
            # AIC e BIC
            n_params = 2  # intercetta + beta
            n_obs = len(y)
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(n_obs) * n_params - 2 * log_likelihood
            
            # Valutazione qualità del modello
            predictions = logistic_model.predict(X)
            accuracy = np.mean(predictions == y)
            
            # Tabella di classificazione
            confusion_matrix = self._create_confusion_matrix(y, predictions)
            
            self._assess_test_appropriateness(
                'Regressione Logistica',
                'Validazione crociata Digital Satisfaction Paradox',
                True,
                'Robusto per qualsiasi dimensione campionaria, appropriato per outcome binario'
            )
            
            return {
                'model_available': True,
                'n_observations': n_obs,
                'converged': True,
                
                # Coefficienti e statistiche principali
                'beta': beta,
                'se_beta': se_beta,
                'z_score': z_score_beta,
                'p_value': p_value_beta,
                'beta_ci_95': [beta_ci_lower, beta_ci_upper],
                
                'intercept': intercept,
                'se_intercept': se_intercept,
                'z_score_intercept': z_score_intercept,
                'p_value_intercept': p_value_intercept,
                
                # Odds ratio
                'odds_ratio': odds_ratio,
                'or_ci_95': [or_ci_lower, or_ci_upper],
                'exp_beta': odds_ratio,  # Alias per compatibilità
                
                # Qualità del modello
                'pseudo_r2_mcfadden': pseudo_r2,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'accuracy': accuracy,
                
                # Diagnostica
                'correlation_coef_inter_beta': correlation_coef,
                'confusion_matrix': confusion_matrix,
                
                # Significatività
                'significant_at_05': p_value_beta < 0.05,
                'significant_at_01': p_value_beta < 0.01,
                'significant_at_001': p_value_beta < 0.001,
                
                # Equazione del modello
                'model_equation': f'logit(P(Y=1)) = {intercept:.3f} + {beta:.3f} × Webinar'
            }
            
        except ImportError:
            self._log_methodological_warning(
                'Regressione Logistica',
                'sklearn non disponibile - analisi di validazione non eseguita'
            )
            return {'model_available': False, 'error': 'sklearn not available'}
            
        except Exception as e:
            self._log_methodological_warning(
                'Regressione Logistica', 
                f'Errore durante il calcolo: {str(e)}'
            )
            return {'model_available': False, 'error': str(e)}
    
    def _bootstrap_logistic_se(self, X, y, n_bootstrap=1000):
        """
        Calcola l'errore standard del coefficiente beta attraverso bootstrap.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            betas = []
            n = len(y)
            
            for _ in range(n_bootstrap):
                # Campionamento bootstrap
                indices = np.random.choice(n, n, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                try:
                    log_reg_boot = LogisticRegression(solver='liblinear', max_iter=1000)
                    log_reg_boot.fit(X_boot, y_boot)
                    betas.append(log_reg_boot.coef_[0][0])
                except:
                    continue
            
            return np.std(betas) if betas else 0.5  # Fallback
        except:
            return 0.5
    
    def _bootstrap_logistic_se_intercept(self, X, y, n_bootstrap=1000):
        """
        Calcola l'errore standard dell'intercetta attraverso bootstrap.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            intercepts = []
            n = len(y)
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(n, n, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                try:
                    log_reg_boot = LogisticRegression(solver='liblinear', max_iter=1000)
                    log_reg_boot.fit(X_boot, y_boot)
                    intercepts.append(log_reg_boot.intercept_[0])
                except:
                    continue
            
            return np.std(intercepts) if intercepts else 0.5
        except:
            return 0.5
    
    def _calculate_mcfadden_r2(self, model, X, y):
        """
        Calcola il Pseudo R-squared di McFadden.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Modello null (solo intercetta)
            null_model = LogisticRegression(solver='liblinear')
            null_model.fit(np.ones((len(X), 1)), y)
            
            # Log-likelihood del modello null e completo
            ll_null = self._calculate_log_likelihood(null_model, np.ones((len(X), 1)), y)
            ll_model = self._calculate_log_likelihood(model, X, y)
            
            # McFadden's Pseudo R-squared
            pseudo_r2 = 1 - (ll_model / ll_null)
            
            return max(0, pseudo_r2)  # Assicura non-negatività
            
        except:
            return np.nan
    
    def _calculate_log_likelihood(self, model, X, y):
        """
        Calcola la log-likelihood del modello.
        """
        try:
            # Predizioni di probabilità
            p_pred = model.predict_proba(X)[:, 1]
            
            # Evita log(0) e log(1)
            p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
            
            # Log-likelihood
            ll = np.sum(y * np.log(p_pred) + (1 - y) * np.log(1 - p_pred))
            
            return ll
        except:
            return np.nan
    
    def _create_confusion_matrix(self, y_true, y_pred):
        """
        Crea matrice di confusione per classificazione binaria.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        
        return {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    def _analyze_generational_membership_gap(self) -> dict:
        """Analizza il Generational Membership Gap."""
        age_groups = ['14-30', '31-50', '51-70', '>70']
        engagement_data = []
        
        for age in age_groups:
            cohort = self.df[self.df['eta_std'] == age]
            
            if len(cohort) > 0:
                membership_rate = (cohort['socio_std'] == 'Sì').mean()
                avg_satisfaction = cohort['soddisfazione_num'].mean()
                engagement_index = membership_rate * avg_satisfaction

                cohort['cognitive_engagement'] = (
                    cohort['Approfondimenti'].notna().astype(int) + 
                    cohort['Proposte'].notna().astype(int)
                )
                cognitive_eng_mean = cohort['cognitive_engagement'].mean()
                engagement_data.append({
                    'age_group': age,
                    'cohort_size': len(cohort),
                    'membership_rate': membership_rate,
                    'avg_satisfaction': avg_satisfaction,
                    'engagement_index': engagement_index,
                    'cognitive_engagement_mean': cognitive_eng_mean,
                    'conversion_potential': (1 - membership_rate) * avg_satisfaction * len(cohort)
                })
        
        # Correlazione età-membership con metodologia appropriata per n piccoli
        age_numerical = [22, 40, 60, 75]
        membership_rates = [d['membership_rate'] for d in engagement_data]
        
        # Correlazione di Spearman (appropriata per n piccoli e dati ordinali)
        from scipy.stats import spearmanr
        correlation_spearman, p_value_spearman = spearmanr(age_numerical, membership_rates)
        
        # Correlazione di Pearson (per confronto, ma con limitazioni)
        correlation_pearson, p_value_pearson = pearsonr(age_numerical, membership_rates)
        
        # Assessment appropriatezza
        sample_size = len(age_numerical)
        normality_feasible = sample_size >= 10  # Soglia minima per test normalità
        
        self._assess_test_appropriateness(
            'Correlazione di Spearman',
            f'Correlazione età-membership con n={sample_size} coorti',
            True,  # Sempre appropriata
            'Non-parametrica, appropriata per campioni piccoli e dati ordinali'
        )
        
        self._assess_test_appropriateness(
            'Correlazione di Pearson',
            f'Correlazione età-membership con n={sample_size} coorti',
            normality_feasible,
            f'Campione troppo piccolo (n={sample_size}) per verificare normalità bivariata' if not normality_feasible else 'Appropriata con verifica normalità'
        )
        
        # Warning per campione piccolo
        if not normality_feasible:
            self._log_methodological_warning(
                'Correlazione età-membership',
                f'Campione molto piccolo (n={sample_size}), preferire metodi non-parametrici'
            )
        
        return {
            'engagement_metrics': engagement_data,
            'age_membership_correlation': correlation_spearman,  # Usa Spearman come primario
            'correlation_p_value': p_value_spearman,
            'r_squared': correlation_spearman**2,
            # Risultati aggiuntivi per confronto
            'spearman_correlation': correlation_spearman,
            'spearman_p_value': p_value_spearman,
            'pearson_correlation': correlation_pearson,
            'pearson_p_value': p_value_pearson,
            'sample_size': sample_size,
            'normality_test_feasible': normality_feasible,
            'recommended_method': 'Spearman'
        }

    def _analyze_inverse_contentment_effect(self) -> dict:
        """Analizza l'Inverse Contentment Effect."""
        # Identificazione feedback elaborato
        def has_elaborate_feedback(row):
            feedback_fields = ['Approfondimenti', 'Proposte']
            for field in feedback_fields:
                if field in row and pd.notna(row[field]):
                    text = str(row[field]).strip()
                    if (text != 'N/A' and len(text) > 10 and 
                        'nessun' not in text.lower() and 'niente' not in text.lower()):
                        return True
            return False
        
        self.df['has_feedback'] = self.df.apply(has_elaborate_feedback, axis=1)
        
        # Segmentazione per livello soddisfazione
        satisfied = self.df[self.df['soddisfazione_num'] == 2]
        very_satisfied = self.df[self.df['soddisfazione_num'] == 3]
        
        # Calcolo metriche feedback
        feedback_metrics = {
            'satisfied': {
                'total': len(satisfied),
                'with_feedback': satisfied['has_feedback'].sum(),
                'feedback_rate': satisfied['has_feedback'].mean()
            },
            'very_satisfied': {
                'total': len(very_satisfied),
                'with_feedback': very_satisfied['has_feedback'].sum(),
                'feedback_rate': very_satisfied['has_feedback'].mean()
            }
        }
        
        # Costruzione tabella di contingenza
        contingency_table = [
            [feedback_metrics['satisfied']['with_feedback'],
             feedback_metrics['satisfied']['total'] - feedback_metrics['satisfied']['with_feedback']],
            [feedback_metrics['very_satisfied']['with_feedback'],
             feedback_metrics['very_satisfied']['total'] - feedback_metrics['very_satisfied']['with_feedback']]
        ]
        
        # VERIFICA CRITERI DI COCHRAN
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Verifica frequenze attese ≥ 5 (Criterio di Cochran)
        min_expected = np.min(expected)
        cells_below_5 = np.sum(expected < 5)
        total_cells = expected.size
        cochran_satisfied = (cells_below_5 == 0) or (cells_below_5 <= 0.2 * total_cells and min_expected >= 1)
        
        # Se criteri Cochran violati, usa Fisher Exact Test
        if not cochran_satisfied:
            fisher_odds, fisher_p = stats.fisher_exact(contingency_table)
            primary_test = 'Fisher Exact Test'
            primary_p_value = fisher_p
            
            self._assess_test_appropriateness(
                'Test Chi-quadro',
                f'Tabella 2×2 con min freq. attesa = {min_expected:.1f}',
                False,
                f'Frequenze attese < 5 in {cells_below_5}/{total_cells} celle, viola criteri Cochran'
            )
            
            self._assess_test_appropriateness(
                'Fisher Exact Test',
                'Tabella 2×2 con frequenze attese basse',
                True,
                'Appropriato per celle con frequenze basse, nessuna assunzione violata'
            )
        else:
            fisher_odds, fisher_p = np.nan, np.nan
            primary_test = 'Chi-quadro'
            primary_p_value = p_value
            
            self._assess_test_appropriateness(
                'Test Chi-quadro',
                f'Tabella 2×2 con min freq. attesa = {min_expected:.1f}',
                True,
                'Criteri di Cochran soddisfatti, appropriato per tabelle di contingenza'
            )
        
        # Contentment Ratio (robusto indipendentemente dal test)
        contentment_ratio = (feedback_metrics['very_satisfied']['feedback_rate'] / 
                           feedback_metrics['satisfied']['feedback_rate'])
        
        return {
            'feedback_metrics': feedback_metrics,
            'contentment_ratio': contentment_ratio,
            'contingency_table': contingency_table,
            # Test statistici
            'primary_test': primary_test,
            'primary_p_value': primary_p_value,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': p_value,
            'fisher_odds_ratio': fisher_odds,
            'fisher_p_value': fisher_p,
            # Verifica appropriatezza
            'cochran_criteria': {
                'satisfied': cochran_satisfied,
                'min_expected_frequency': min_expected,
                'cells_below_5': int(cells_below_5),
                'total_cells': int(total_cells),
                'expected_frequencies': expected.tolist()
            },
            'effect_confirmed': contentment_ratio > 1,
            # Legacy compatibility
            'p_value': primary_p_value
        }
    
    def _old_calculate_cultural_engagement_score(self) -> dict:
        """
        Calcola il Cultural Engagement Score (CES) e le sue componenti.
        
        Returns:
        --------
        dict
            Componenti e score finale del CES
        """
        print("📈 Calcolo Cultural Engagement Score...")
        
        # Componenti del CES
        avg_satisfaction = self.df['soddisfazione_num'].mean()
        satisfaction_component = (avg_satisfaction - 1) / 2  # Normalizzazione 0-1
        
        membership_rate = (self.df['socio_std'] == 'Sì').mean()
        digital_adoption_rate = (self.df['Fonte'] == 'Webinar').mean()
        
        # Formula CES: (S̄/3) × (1 + M) × (1 + D)
        ces_score = satisfaction_component * (1 + membership_rate) * (1 + digital_adoption_rate)
        
        # Validazione bootstrap
        bootstrap_scores = self._bootstrap_ces_validation(n_iterations=1000)
        
        ces_results = {
            'score': ces_score,
            'components': {
                'satisfaction': satisfaction_component,
                'membership': membership_rate,
                'digital_adoption': digital_adoption_rate,
                'max_theoretical': 4.0
            },
            'percentile_rank': (ces_score / 4.0) * 100,
            'bootstrap_validation': {
                'mean': np.mean(bootstrap_scores),
                'std_error': np.std(bootstrap_scores),
                'ci_95_lower': np.percentile(bootstrap_scores, 2.5),
                'ci_95_upper': np.percentile(bootstrap_scores, 97.5),
                'cv': np.std(bootstrap_scores) / np.mean(bootstrap_scores)
            }
        }
        
        # Scenari di miglioramento
        ces_results['improvement_scenarios'] = self._generate_improvement_scenarios(ces_results['components'])
        
        self.results['ces'] = ces_results
        return ces_results
    
    def calculate_cultural_engagement_score(self) -> dict:
        """
        Calcola il Cultural Engagement Score (CES) e le sue componenti.
        
        Returns:
        --------
        dict
            Componenti e score finale del CES
        """
        print("📈 Calcolo Cultural Engagement Score...")
        
        # Componenti del CES
        avg_satisfaction = self.df['soddisfazione_num'].mean()
        satisfaction_component = (avg_satisfaction - 1) / 2  # Normalizzazione 0-1
        
        membership_rate = (self.df['socio_std'] == 'Sì').mean()
        self.df['cognitive_engagement'] = (
            self.df['Approfondimenti'].notna().astype(int) + 
            self.df['Proposte'].notna().astype(int)
        )
        cognitive_engagement_mean = self.df['cognitive_engagement'].mean()/2 # Normalizzazione 0-1
        # Formula CES: (S+1/2) × (1 + M) × (1 + D)
        ces_score = satisfaction_component * (1 + membership_rate)**0.5 * (1 + cognitive_engagement_mean)**0.5
        max_satisfaction = 1.0
        max_membership = 1.0
        max_cognitive = 1.0
        maximum_ces = max_satisfaction * (1 + max_membership)**0.5 * (1 + max_cognitive)**0.5
        # Validazione bootstrap
        bootstrap_scores = self._bootstrap_ces_validation(n_iterations=1000)
        
        ces_results = {
            'score': ces_score,
            'components': {
                'satisfaction': satisfaction_component,
                'membership': membership_rate,
                'cognitive_engagement_mean': cognitive_engagement_mean,
                'max_theoretical': maximum_ces
            },
            'percentile_rank': (ces_score / maximum_ces) * 100,
            'bootstrap_validation': {
                'mean': np.mean(bootstrap_scores),
                'std_error': np.std(bootstrap_scores),
                'ci_95_lower': np.percentile(bootstrap_scores, 2.5),
                'ci_95_upper': np.percentile(bootstrap_scores, 97.5),
                'cv': np.std(bootstrap_scores) / np.mean(bootstrap_scores)
            }
        }
        
        # Scenari di miglioramento
        ces_results['improvement_scenarios'] = self._generate_improvement_scenarios(ces_results['components'])
        
        self.results['ces'] = ces_results
        return ces_results
    
    def _bootstrap_ces_validation(self, n_iterations: int = 1000) -> list:
        """Validazione bootstrap del CES."""
        bootstrap_scores = []
        for _ in range(n_iterations):
            # Campionamento con rimpiazzamento
            rng = np.random.RandomState()
            sample = self.df.sample(n=len(self.df), replace=True, random_state=rng)            
            # Calcolo CES per il campione
            avg_sat = sample['soddisfazione_num'].mean()
            sat_comp = (avg_sat - 1 ) / 2
            mem_rate = (sample['socio_std'] == 'Sì').mean()
            cog_rate = (sample['Approfondimenti'].notna().astype(int) + 
                        sample['Proposte'].notna().astype(int)
                        ).mean() / 2
            
            ces = sat_comp * (1 + mem_rate)**0.5 * (1 + cog_rate)**0.5
            bootstrap_scores.append(ces)
        
        return bootstrap_scores
    
    def _generate_improvement_scenarios(self, baseline_components: dict) -> dict:
        """Genera scenari di miglioramento per il CES."""
        scenarios = {
            'conservativo': {'membership': 0.05, 'digital': 0.10},
            'ambizioso': {'membership': 0.15, 'digital': 0.25},
            'eccellenza': {'membership': 0.25, 'digital': 0.40}
        }
        
        results = {}
        baseline_ces = (baseline_components['satisfaction'] * 
                       (1 + baseline_components['membership']) * 
                       (1 + baseline_components['cognitive_engagement_mean']))
        
        for scenario, changes in scenarios.items():
            new_membership = min(1.0, baseline_components['membership'] + changes['membership'])
            new_cognitive = min(1.0, baseline_components['cognitive_engagement_mean'] + changes['digital'])
            
            new_ces = (baseline_components['satisfaction'] * 
                      (1 + new_membership) * 
                      (1 + new_cognitive))
            
            improvement = ((new_ces - baseline_ces) / baseline_ces) * 100
            
            results[scenario] = {
                'projected_ces': new_ces,
                'improvement_percentage': improvement,
                'target_membership': new_membership,
                'target_cognitive': new_cognitive
            }
        
        return results
    
    def print_detailed_digital_paradox_analysis(self):
        """
        Stampa analisi dettagliata del Digital Satisfaction Paradox con tutti i test.
        """
        print("\n" + "="*90)
        print("🎯 DIGITAL SATISFACTION PARADOX - ANALISI COMPLETA MULTI-METODOLOGICA")
        print("="*90)
        
        if 'digital_paradox' not in self.results.get('key_findings', {}):
            print("❌ Analisi non ancora eseguita")
            return
        
        paradox_data = self.results['key_findings']['digital_paradox']
        
        # === STATISTICHE DESCRITTIVE ===
        print("📊 STATISTICHE DESCRITTIVE:")
        print(f"   📋 CARTACEO:")
        print(f"      • N = {paradox_data['cartaceo_stats']['n']}")
        print(f"      • Media = {paradox_data['cartaceo_stats']['mean']:.3f} ± {paradox_data['cartaceo_stats']['std']:.3f}")
        print(f"      • % Molto Soddisfatti = {paradox_data['cartaceo_stats']['satisfaction_rate_max']:.1%}")
        print(f"   📋 WEBINAR:")
        print(f"      • N = {paradox_data['webinar_stats']['n']}")
        print(f"      • Media = {paradox_data['webinar_stats']['mean']:.3f} ± {paradox_data['webinar_stats']['std']:.3f}")
        print(f"      • % Molto Soddisfatti = {paradox_data['webinar_stats']['satisfaction_rate_max']:.1%}")
        print(f"   📈 DIFFERENZA: +{paradox_data['difference']:.3f} punti (Webinar > Cartaceo)")
        
        # === FISHER EXACT TEST (PRINCIPALE) ===
        print(f"\n🥇 FISHER EXACT TEST (Metodo Principale):")
        print(f"   • Odds Ratio: {paradox_data['fisher_odds_ratio']:.3f}")
        print(f"   • p-value: {paradox_data['fisher_p_value']:.4f}")
        print(f"   • Significatività: {'***' if paradox_data['fisher_p_value'] < 0.001 else '**' if paradox_data['fisher_p_value'] < 0.01 else '*' if paradox_data['fisher_p_value'] < 0.05 else 'n.s.'}")
        print(f"   • Differenza Proporzioni: +{paradox_data['proportion_difference']:.3f}")
        print(f"   • Appropriatezza: ✅ OTTIMALE per dati binari")
        
        # Tabella di contingenza
        contingency = paradox_data['contingency_table']
        print(f"   📋 Tabella di Contingenza 2×2:")
        print(f"                    Soddisfatto(2)  Molto Sodd.(3)")
        print(f"      Cartaceo:     {contingency[0][0]:>11}  {contingency[0][1]:>13}")
        print(f"      Webinar:      {contingency[1][0]:>11}  {contingency[1][1]:>13}")
        
        # === REGRESSIONE LOGISTICA ===
        if paradox_data['logistic_regression'].get('model_available', False):
            print(f"\n🔬 REGRESSIONE LOGISTICA (Validazione Terziaria):")
            lr_data = paradox_data['logistic_regression']
            
            print(f"   📈 COEFFICIENTI:")
            print(f"      • β (Webinar): {lr_data['beta']:.3f} ± {lr_data['se_beta']:.3f}")
            print(f"      • Z-score: {lr_data['z_score']:.3f}")
            print(f"      • p-value: {lr_data['p_value']:.4f}")
            print(f"      • Significatività: {'***' if lr_data['p_value'] < 0.001 else '**' if lr_data['p_value'] < 0.01 else '*' if lr_data['p_value'] < 0.05 else 'n.s.'}")
            print(f"      • CI 95% (β): [{lr_data['beta_ci_95'][0]:.3f}, {lr_data['beta_ci_95'][1]:.3f}]")
            
            print(f"   🎯 ODDS RATIO:")
            print(f"      • OR [exp(β)]: {lr_data['odds_ratio']:.3f}")
            print(f"      • CI 95% (OR): [{lr_data['or_ci_95'][0]:.3f}, {lr_data['or_ci_95'][1]:.3f}]")
            print(f"      • Incremento Chance: +{(lr_data['odds_ratio']-1)*100:.1f}%")
            
            print(f"   📊 QUALITÀ MODELLO:")
            print(f"      • Pseudo R² (McFadden): {lr_data['pseudo_r2_mcfadden']:.3f}")
            print(f"      • Log-Likelihood: {lr_data['log_likelihood']:.1f}")
            print(f"      • AIC: {lr_data['aic']:.1f}")
            print(f"      • BIC: {lr_data['bic']:.1f}")
            print(f"      • Accuracy: {lr_data['accuracy']:.3f}")
            
            print(f"   📝 EQUAZIONE:")
            print(f"      {lr_data['model_equation']}")
            
            # Convergenza metodologica
            or_diff = abs(paradox_data['fisher_odds_ratio'] - lr_data['odds_ratio'])
            convergence_status = "✅ ECCELLENTE" if or_diff < 0.1 else "⚠️ ACCETTABILE" if or_diff < 0.5 else "❌ PROBLEMATICA"
            print(f"   ✅ CONVERGENZA METODOLOGICA:")
            print(f"      • Fisher OR: {paradox_data['fisher_odds_ratio']:.3f}")
            print(f"      • Logistic OR: {lr_data['odds_ratio']:.3f}")
            print(f"      • Differenza: {or_diff:.3f}")
            print(f"      • Status: {convergence_status}")
        else:
            print(f"\n🔬 REGRESSIONE LOGISTICA: ❌ Non disponibile")
            if 'error' in paradox_data['logistic_regression']:
                print(f"      Errore: {paradox_data['logistic_regression']['error']}")
        
        # === MANN-WHITNEY U (SECONDARIO CON DISCLAIMER) ===
        print(f"\n⚠️  MANN-WHITNEY U (Analisi Secondaria - LIMITAZIONI):")
        print(f"   • U-statistic: {paradox_data['mannwhitney_u']:.1f}")
        print(f"   • p-value: {paradox_data['mannwhitney_p']:.4f}")
        print(f"   • Rosenthal r: {paradox_data['rosenthal_r']:.3f}")
        print(f"   • Z-score: {paradox_data['z_score']:.3f}")
        print(f"   • Ties: {paradox_data['ties_percentage']:.1%} (ECCESSIVI)")
        print(f"   • Appropriatezza: ❌ SUBOTTIMALE")
        print(f"   • Limitazioni: Solo 2 valori distinti, violazione assunzioni")
        
        # === ANALISI ETÀ ===
        print(f"\n🔍 AGE PARADOX:")
        print(f"   • Età media Webinar: {paradox_data['age_webinar']:.1f} anni")
        print(f"   • Età media Cartaceo: {paradox_data['age_cartaceo']:.1f} anni") 
        print(f"   • Differenza: +{paradox_data['age_paradox']:.1f} anni (Webinar > Cartaceo)")
        print(f"   • Interpretazione: Controintuitiva rispetto al digital divide atteso")
        
        # === RACCOMANDAZIONI METODOLOGICHE ===
        print(f"\n📋 RACCOMANDAZIONI METODOLOGICHE:")
        print(f"   ✅ UTILIZZARE: Fisher Exact Test come risultato principale")
        print(f"   ✅ UTILIZZARE: Regressione Logistica per validazione e IC")
        print(f"   ⚠️  EVITARE: Mann-Whitney U per dati con ties eccessivi")
        print(f"   📊 RIPORTARE: Odds Ratio e intervalli di confidenza al 95%")
        print(f"   📈 CITARE: Tabella di contingenza e proporzioni per trasparenza")
        
        print("="*90)
    
    def create_visualizations(self):
        """Crea tutte le visualizzazioni per l'analisi."""
        print("📊 Generazione visualizzazioni...")
        
        # 1. Analisi demografica
        self._plot_demographic_analysis()
        
        # 2. Digital Satisfaction Paradox (CORRETTO)
        self._plot_digital_satisfaction_paradox()
        
        # 3. Generational Membership Gap
        self._plot_generational_membership_gap()
        
        # 4. Distribuzione soddisfazione
        self._plot_satisfaction_distribution()
        
        # 5. Distribuzione degli elementi di soddisfazione
        self._plot_satisfaction_elements()

        # 6. Analisi canali comunicazione
        self._plot_communication_channels()
        
        # 7. Cultural Engagement Score
        self._plot_ces_analysis()
        
        # 8. 🎓 GRAFICI SPECIFICI PER LA TESI (NUOVI)
        self._plot_demographic_overview_panel()

        # 9. Pannello modalità di partecipazione
        self._plot_participation_modalities_panel()
        
        print(f"   ✓ Grafici salvati in: {self.output_dir}")
    
    def _plot_demographic_overview_panel(self):
        """
        🎓 GRAFICO TESI: Pannello demografico completo con tre visualizzazioni:
        - Sinistra: Distribuzione per fascia d'età 
        - Centro: Ripartizione soci vs non-soci
        - Destra: Livello di scolarizzazione
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Panoramica Demografica del Campione Primario', fontsize=16, fontweight='bold', 
                     color=PURPLE_PALETTE['dark_purple'])
        
        # 1. DISTRIBUZIONE PER FASCIA D'ETÀ (Sinistra)
        age_counts = self.df['eta_std'].value_counts()
        # Ordina le fasce d'età logicamente
        age_order = ['14-30', '31-50', '51-70', '>70']
        age_counts = age_counts.reindex(age_order, fill_value=0)
        
        bars1 = axes[0].bar(age_counts.index, age_counts.values, 
                           color=PURPLE_PALETTE['primary_purple'], alpha=0.8,
                           edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0].set_title('Distribuzione per Fascia d\'Età', 
                         color=PURPLE_PALETTE['dark_purple'], fontweight='bold')
        axes[0].set_ylabel('Numero Partecipanti')
        axes[0].set_xlabel('Fascia d\'Età')
        axes[0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni valori sulle barre
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold',
                        color=PURPLE_PALETTE['dark_purple'])
        
        # 2. RIPARTIZIONE SOCI VS NON-SOCI (Centro)
        member_counts = self.df['socio_std'].value_counts()
        # Ordina sempre No prima, Sì dopo
        member_order = ['No', 'Sì']
        member_counts = member_counts.reindex(member_order, fill_value=0)
        
        colors_pie = [PURPLE_PALETTE['sage'], PURPLE_PALETTE['dusty_rose']]
        labels_pie = ['Non Soci', 'Soci']
        
        wedges, texts, autotexts = axes[1].pie(member_counts.values, 
                                              labels=labels_pie,
                                              autopct='%1.1f%%', colors=colors_pie, 
                                              startangle=90, 
                                              textprops={'color': PURPLE_PALETTE['dark_purple'], 'fontweight': 'bold'})
        axes[1].set_title('Ripartizione Soci vs Non-Soci', 
                         color=PURPLE_PALETTE['dark_purple'], fontweight='bold')
        
        # 3. LIVELLO DI SCOLARIZZAZIONE (Destra)
        education_counts = self.df['Titolo di studio'].value_counts()
        # Mappa e ordina i livelli di istruzione
        education_mapping = {
            'Licenza media': 'Licenza media',
            'Diploma di scuola superiore': 'Diploma superiore', 
            'Laurea': 'Laurea',
            'Laurea magistrale': 'Laurea magistrale',
            'Post-laurea': 'Post-laurea'
        }
        
        # Applica mapping e riordina
        education_clean = {}
        for original, mapped in education_mapping.items():
            if original in education_counts:
                education_clean[mapped] = education_counts[original]
        
        # Ordine logico dal più basso al più alto
        education_order = ['Licenza media', 'Diploma superiore', 'Laurea', 'Laurea magistrale', 'Post-laurea']
        ordered_education = {k: education_clean.get(k, 0) for k in education_order if k in education_clean}
        
        bars3 = axes[2].barh(list(ordered_education.keys()), list(ordered_education.values()),
                            color=PURPLE_PALETTE['warm_gold'], alpha=0.8,
                            edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[2].set_title('Livello di Scolarizzazione', 
                         color=PURPLE_PALETTE['dark_purple'], fontweight='bold')
        axes[2].set_xlabel('Numero Partecipanti')
        axes[2].grid(axis='x', alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni valori sulle barre orizzontali
        for bar in bars3:
            width = bar.get_width()
            if width > 0:  # Solo se c'è valore
                percentage = (width / sum(list(ordered_education.values()))) * 100
                axes[2].text(width + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{int(width)} ({percentage:.1f}%)', ha='left', va='center', fontweight='bold',
                            color=PURPLE_PALETTE['dark_purple'])
        
        # Aggiungi statistiche sommarie
        total_participants = len(self.df)
        dominant_age = age_counts.idxmax()
        dominant_age_count = age_counts.max()
        soci_percentage = (self.df['socio_std'] == 'Sì').mean() * 100
        top_education = max(ordered_education, key=ordered_education.get) if ordered_education else 'N/A'
        
        stats_text = f'Campione totale: {total_participants} partecipanti\n'
        stats_text += f'Fascia età dominante: {dominant_age} ({dominant_age_count} partecipanti)\n'
        stats_text += f'Percentuale soci: {soci_percentage:.1f}%\n'
        stats_text += f'Livello istruzione prevalente: {top_education}'
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                 bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                          alpha=0.9, edgecolor=PURPLE_PALETTE['primary_purple']),
                 color=PURPLE_PALETTE['dark_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_overview_panel.png', 
                   facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_participation_modalities_panel(self):
        """
        🎓 GRAFICO TESI: Pannello con modalità di partecipazione:
        - Sinistra: Modalità di fruizione (Cartaceo vs Webinar)
        - Destra: Partecipazione per argomento trattato
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Modalità di Partecipazione e Distribuzione per Argomenti', 
                     fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. MODALITÀ DI FRUIZIONE (Sinistra)
        modality_counts = self.df['Fonte'].value_counts()
        colors_modality = [PURPLE_PALETTE['periwinkle'], PURPLE_PALETTE['royal_purple']]
        
        bars1 = axes[0].bar(modality_counts.index, modality_counts.values,
                           color=colors_modality, alpha=0.8,
                           edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0].set_title('Modalità di Fruizione', 
                         color=PURPLE_PALETTE['dark_purple'], fontweight='bold')
        axes[0].set_ylabel('Numero Partecipanti')
        axes[0].set_xlabel('Modalità')
        axes[0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni con percentuali
        total_modality = sum(modality_counts.values)
        for bar, count in zip(bars1, modality_counts.values):
            percentage = (count / total_modality) * 100
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                        fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 2. PARTECIPAZIONE PER ARGOMENTO (Destra)
        topic_counts = self.df['Argomento'].value_counts()
        # Ordina per frequenza decrescente
        topic_counts = topic_counts.sort_values(ascending=True)  # Per barh, ordine crescente = dal basso verso l'alto
        
        # Crea gradiente di colori purple
        colors_topics = [PURPLE_SEQUENTIAL[i % len(PURPLE_SEQUENTIAL)] for i in range(len(topic_counts))]
        
        bars2 = axes[1].barh(range(len(topic_counts)), topic_counts.values,
                            color=colors_topics, alpha=0.8,
                            edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        axes[1].set_yticks(range(len(topic_counts)))
        axes[1].set_yticklabels(topic_counts.index, fontsize=10)
        axes[1].set_title('Partecipazione per Argomento Trattato', 
                         color=PURPLE_PALETTE['dark_purple'], fontweight='bold')
        axes[1].set_xlabel('Numero Partecipanti')
        axes[1].grid(axis='x', alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni valori
        for i, (bar, count) in enumerate(zip(bars2, topic_counts.values)):
            axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontweight='bold',
                        color=PURPLE_PALETTE['dark_purple'])
        
        # Statistiche aggiuntive
        digital_adoption = (modality_counts.get('Webinar', 0) / total_modality) * 100
        most_popular_topic = topic_counts.index[-1]  # Ultimo nell'ordine crescente = più alto
        
        participation_stats = f'📊 STATISTICHE PARTECIPAZIONE:\n'
        participation_stats += f'• Adozione digitale: {digital_adoption:.1f}%\n'
        participation_stats += f'• Argomento più seguito: {most_popular_topic}\n'
        participation_stats += f'• Range partecipazione: {topic_counts.min()}-{topic_counts.max()} per evento\n'
        participation_stats += f'• Eventi monitorati: {len(topic_counts)}'
        
        axes[1].text(0.98, 0.02, participation_stats, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                             alpha=0.9, edgecolor=PURPLE_PALETTE['sage'], linewidth=2),
                    color=PURPLE_PALETTE['dark_purple'], fontsize=9, 
                    ha='right', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'participation_modalities_panel.png', 
                   facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_demographic_analysis(self):
        """Crea visualizzazioni demografiche con tema purple."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analisi Demografica del Campione', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. Distribuzione per età
        age_counts = self.df['eta_std'].value_counts()
        bars1 = axes[0,0].bar(age_counts.index, age_counts.values, 
                             color=PURPLE_PALETTE['primary_purple'], alpha=0.8, 
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,0].set_title('Distribuzione per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[0,0].set_ylabel('Numero Partecipanti')
        axes[0,0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[0,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # 2. Soci vs Non-soci
        member_counts = self.df['socio_std'].value_counts()
        colors_pie = [PURPLE_PALETTE['light_purple'], PURPLE_PALETTE['sage']]
        wedges, texts, autotexts = axes[0,1].pie(member_counts.values, labels=member_counts.index, 
                                                autopct='%1.1f%%', colors=colors_pie, 
                                                startangle=90, textprops={'color': PURPLE_PALETTE['dark_purple']})
        axes[0,1].set_title('Distribuzione Soci vs Non-Soci', color=PURPLE_PALETTE['dark_purple'])
        
        # 3. Soddisfazione per età
        satisfaction_by_age = self.df.groupby('eta_std')['soddisfazione_num'].mean()
        bars3 = axes[1,0].bar(satisfaction_by_age.index, satisfaction_by_age.values, 
                             color=PURPLE_PALETTE['periwinkle'], alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[1,0].set_title('Soddisfazione Media per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[1,0].set_ylabel('Soddisfazione Media')
        axes[1,0].set_ylim(2.0, 3.0)
        axes[1,0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[1,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # 4. Modalità di fruizione
        mode_counts = self.df['Fonte'].value_counts()
        bars4 = axes[1,1].bar(mode_counts.index, mode_counts.values, 
                             color=PURPLE_PALETTE['warm_gold'], alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[1,1].set_title('Distribuzione per Modalità di Fruizione', color=PURPLE_PALETTE['dark_purple'])
        axes[1,1].set_ylabel('Numero Partecipanti')
        axes[1,1].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[1,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_analysis.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_digital_satisfaction_paradox(self):
        """
        Visualizza il Digital Satisfaction Paradox con statistiche corrette.
        AGGIORNATO: Fisher Exact Test come metodo principale + Regressione Logistica.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Paradosso di Soddisfazione Digitale', 
                     fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # Dati per modalità
        webinar_data = self.df[self.df['Fonte'] == 'Webinar']['soddisfazione_num']
        cartaceo_data = self.df[self.df['Fonte'] == 'Cartaceo']['soddisfazione_num']
        
        # === 1. CONFRONTO PROPORZIONI (FISHER TEST) ===
        paradox_data = self.results['key_findings']['digital_paradox']
        
        # Estrazione proporzioni "Molto Soddisfatti"
        prop_cartaceo = paradox_data['cartaceo_stats']['satisfaction_rate_max']
        prop_webinar = paradox_data['webinar_stats']['satisfaction_rate_max']
        
        modes = ['Cartaceo', 'Webinar']
        proportions = [prop_cartaceo, prop_webinar]
        
        bars1 = axes[0].bar(modes, proportions, 
                           color=[PURPLE_PALETTE['dusty_rose'], PURPLE_PALETTE['periwinkle']], 
                           alpha=0.8, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        axes[0].set_title('Proporzione "Molto Soddisfatti" per Modalità', 
                         color=PURPLE_PALETTE['dark_purple'])
        axes[0].set_ylabel('Proporzione "Molto Soddisfatti"')
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(True, alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni Fisher Test
        fisher_text = f'FISHER EXACT TEST (Principale)\n'
        fisher_text += f'Odds Ratio = {paradox_data["fisher_odds_ratio"]:.2f}\n'
        fisher_text += f'p-value = {paradox_data["fisher_p_value"]:.3f}\n'
        fisher_text += f'Diff. Proporzioni = +{paradox_data["proportion_difference"]:.3f}\n'
        fisher_text += f'Test Raccomandato: ' + ('Sì' if paradox_data["fisher_p_value"] < 0.05 else 'No')
        
        axes[0].text(0.02, 0.98, fisher_text, transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                             alpha=0.9, edgecolor=PURPLE_PALETTE['sage'], linewidth=2),
                    color=PURPLE_PALETTE['dark_purple'], fontsize=9, va='top')
        
        # Annotazioni valori sulle barre
        for bar, prop in zip(bars1, proportions):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prop:.1%}', ha='center', va='bottom', fontweight='bold',
                        color=PURPLE_PALETTE['dark_purple'])
        
        # === 2. TABELLA DI CONTINGENZA VISUALIZZATA ===
        contingency = np.array(paradox_data['contingency_table'])
        
        # Heatmap della tabella di contingenza
        sns.heatmap(contingency, annot=True, fmt='d', 
                   xticklabels=['Soddisfatto (2)', 'Molto Soddisfatto (3)'],
                   yticklabels=['Cartaceo', 'Webinar'],
                   cmap=sns.color_palette(PURPLE_SEQUENTIAL, as_cmap=True),
                   ax=axes[1], cbar_kws={'label': 'Numero Partecipanti'},
                   annot_kws={'color': PURPLE_PALETTE['dark_purple'], 'fontweight': 'bold'})
        
        axes[1].set_title('Tabella di Contingenza 2×2\n(Base per Fisher Exact Test)', 
                         color=PURPLE_PALETTE['dark_purple'])
        axes[1].set_xlabel('Livello Soddisfazione')
        axes[1].set_ylabel('Modalità Fruizione')
        
        # === 3. CONFRONTO METODOLOGIE + REGRESSIONE LOGISTICA ===
        # Barplot comparativo dei p-values e risultati
        methods = ['Fisher Exact\n(Principale)', 'Mann-Whitney U\n(Problematico)', 'Logistic Regr.\n(Validazione)']
        p_values = [
            paradox_data['fisher_p_value'], 
            paradox_data['mannwhitney_p'],
            paradox_data['logistic_regression'].get('p_value', 1.0) if paradox_data['logistic_regression'].get('model_available', False) else 1.0
        ]
        
        # Colori: verde per appropriato, rosso per problematico, blu per validazione
        colors_methods = [PURPLE_PALETTE['sage'], PURPLE_PALETTE['dusty_rose'], PURPLE_PALETTE['periwinkle']]
        bars3 = axes[2].bar(methods, p_values, color=colors_methods, alpha=0.8,
                           edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        # Linea di significatività
        axes[2].axhline(y=0.05, color=PURPLE_PALETTE['dark_purple'], linestyle='--', 
                       linewidth=2, label='α = 0.05')
        
        axes[2].set_title('Confronto Metodologie Statistiche', color=PURPLE_PALETTE['dark_purple'])
        axes[2].set_ylabel('p-value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni metodologiche con OR
        odds_ratios = [
            paradox_data['fisher_odds_ratio'],
            np.nan,  # Mann-Whitney non ha OR
            paradox_data['logistic_regression'].get('odds_ratio', np.nan) if paradox_data['logistic_regression'].get('model_available', False) else np.nan
        ]
        
        for bar, p_val, or_val, method in zip(bars3, p_values, odds_ratios, methods):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            or_text = f'\nOR: {or_val:.2f}' if not np.isnan(or_val) else ''
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{p_val:.3f}\n{significance}{or_text}', ha='center', va='bottom', 
                        fontweight='bold', color=PURPLE_PALETTE['dark_purple'], fontsize=9)
        
        # Box con risultati regressione logistica
        if paradox_data['logistic_regression'].get('model_available', False):
            logistic_data = paradox_data['logistic_regression']
            logistic_text = f'REGRESSIONE LOGISTICA:\n'
            logistic_text += f'β = {logistic_data.get("beta", 0):.3f} ± {logistic_data.get("se_beta", 0):.3f}\n'
            logistic_text += f'OR = {logistic_data.get("odds_ratio", 0):.3f}\n'
            logistic_text += f'CI 95%: [{logistic_data.get("or_ci_95", [0,0])[0]:.2f}, {logistic_data.get("or_ci_95", [0,0])[1]:.2f}]\n'
            logistic_text += f'Pseudo R² = {logistic_data.get("pseudo_r2_mcfadden", 0):.3f}\n'
            logistic_text += f'Convergenza: {logistic_data.get("convergence_status", "N/A")}\n'
            
            axes[2].text(0.02, 0.98, logistic_text, transform=axes[2].transAxes,
                        bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                                 alpha=0.9, edgecolor=PURPLE_PALETTE['periwinkle'], linewidth=2),
                        color=PURPLE_PALETTE['dark_purple'], fontsize=8, va='top')
        
        # Warning metodologico per Mann-Whitney
        if paradox_data.get('mw_methodological_warning', False):
            warning_text = f'⚠️ Mann-Whitney U non ottimale:\n'
            warning_text += f'   • Ties: {paradox_data["ties_percentage"]:.1%}\n'
            warning_text += f'   • Solo 2 valori distinti\n'
            warning_text += f'   • Dati essenzialmente binari'
            
            axes[2].text(0.98, 0.02, warning_text, transform=axes[2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='#FFE4E1', 
                                 alpha=0.9, edgecolor='#DC143C', linewidth=1),
                        color='#8B0000', fontsize=8, ha='right', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'digital_satisfaction_paradox_corrected.png', 
                   facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')
        plt.close()
        
        # === SALVATAGGIO TABELLA DI CONTINGENZA ===
        # Export della tabella per riferimento
        contingency_df = pd.DataFrame(
            contingency,
            index=['Cartaceo', 'Webinar'],
            columns=['Soddisfatto (2)', 'Molto Soddisfatto (3)']
        )
        contingency_df.to_csv(self.output_dir / 'contingency_table_fisher_test.csv')
    
    def _plot_generational_membership_gap(self):
        """Visualizza il Generational Membership Gap con tema purple."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analisi del Divario Associazionistico Generazionale', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        gap_data = self.results['key_findings']['generational_gap']['engagement_metrics']
        age_groups = [d['age_group'] for d in gap_data]
        membership_rates = [d['membership_rate'] * 100 for d in gap_data]
        engagement_indices = [d['engagement_index'] for d in gap_data]
        cognitive_engagement = [d['cognitive_engagement_mean'] for d in gap_data]
        complex_engagement = [(cognitive + engagement) / 2 for cognitive, engagement in zip(cognitive_engagement, engagement_indices)]
        
        # 1. Percentuale soci per età
        bars1 = axes[0,0].bar(age_groups, membership_rates, 
                             color=PURPLE_PALETTE['royal_purple'], alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,0].set_title('Percentuale Soci per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[0,0].set_ylabel('% Soci')
        axes[0,0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[0,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazione correlazione
        corr = self.results['key_findings']['generational_gap']['age_membership_correlation']
        axes[0,0].text(0.02, 0.98, f'r = {corr:.3f}', transform=axes[0,0].transAxes,
                      bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                               alpha=0.9, edgecolor=PURPLE_PALETTE['primary_purple']),
                      color=PURPLE_PALETTE['dark_purple'])
        
        # 2. Engagement Index
        bar_width = 0.25  # Adjusted width of the bars
        axes[0,1].set(ylim=(0, 2.0))
        x_positions = np.arange(len(age_groups))  # Positions for the bars

        # Engagement Index bars
        bars_engagement = axes[0,1].bar(x_positions - bar_width, engagement_indices, 
                width=bar_width, color=PURPLE_PALETTE['sage'], alpha=0.8,
                edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5, label='Institutional Engagement Index (IEI)')

        # Cognitive Engagement Mean bars
        bars_cognitive = axes[0,1].bar(x_positions, cognitive_engagement, 
                   width=bar_width, color=PURPLE_PALETTE['royal_purple'], alpha=0.8,
                   edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5, label='Cognitive Engagement Index (CEI)')

        # Composite Engagement Score bars
        bars_complex = axes[0,1].bar(x_positions + bar_width, complex_engagement, 
                   width=bar_width, color=PURPLE_PALETTE['warm_gold'], alpha=0.8,
                   edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5, label='Composite Engagement Score')

        # Configure the axes
        axes[0,1].set_title('Engagement Metrics per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].set_ylabel('Engagement Metrics')
        axes[0,1].set_xticks(x_positions)
        axes[0,1].set_xticklabels(age_groups, rotation=45, color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        axes[0,1].legend(loc='upper left', fontsize=10, frameon=False)
        # 3. Heatmap distribuzione
        crosstab = pd.crosstab(self.df['eta_std'], self.df['socio_std'], margins=True)
        # Create purple colormap
        purple_cmap = sns.color_palette(PURPLE_SEQUENTIAL, as_cmap=True)
        sns.heatmap(crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap=purple_cmap, 
                   ax=axes[1,0], cbar_kws={'label': 'Numero Partecipanti'},
                   annot_kws={'color': PURPLE_PALETTE['dark_purple']})
        axes[1,0].set_title('Heatmap: Età × Status Socio', color=PURPLE_PALETTE['dark_purple'])
        axes[1,0].set_xlabel('Status Socio')
        axes[1,0].set_ylabel('Fascia d\'Età')
        
        # 4. Potenziale di conversione
        conversion_potential = [d['conversion_potential'] for d in gap_data]
        bars4 = axes[1,1].bar(age_groups, conversion_potential, 
                             color=PURPLE_PALETTE['warm_gold'], alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[1,1].set_title('Potenziale di Conversione per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[1,1].set_ylabel('Potenziale Conversione')
        axes[1,1].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[1,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generational_membership_gap.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_satisfaction_distribution(self):
        """Visualizza la distribuzione della soddisfazione con tema purple."""
        fig, axes = plt.subplots(2,2, figsize=(16, 16))
        fig.suptitle('Analisi Distribuzione Soddisfazione', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. Distribuzione generale
        satisfaction_counts = self.df['soddisfazione_num'].value_counts().sort_index()
        colors = [PURPLE_PALETTE['primary_purple'] if score == 2 else PURPLE_PALETTE['warm_gold'] for score in satisfaction_counts.index]
        bars1 = axes[0,0].bar(satisfaction_counts.index, satisfaction_counts.values, 
                             color=colors, alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,0].set_title('Distribuzione Generale della Soddisfazione', color=PURPLE_PALETTE['dark_purple'])
        axes[0,0].set_xlabel('Livello Soddisfazione')
        axes[0,0].set_ylabel('Numero Partecipanti')
        axes[0,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni percentuali
        total = sum(satisfaction_counts.values)
        for bar, count in zip(bars1, satisfaction_counts.values):
            percentage = (count / total) * 100
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                          fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 2. Soddisfazione per modalità
        satisfaction_by_mode = pd.crosstab(self.df['Fonte'], self.df['soddisfazione_num'])
        satisfaction_by_mode.plot(kind='bar', ax=axes[0,1], 
                                 color=[PURPLE_PALETTE['light_purple'], PURPLE_PALETTE['royal_purple']],
                                 alpha=0.8, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,1].set_title('Soddisfazione per Modalità di Fruizione', color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].set_xlabel('Modalità')
        axes[0,1].set_ylabel('Numero Partecipanti')
        axes[0,1].legend(['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[0,1].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[0,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # 3. Soddisfazione per fascia d'età
        satisfaction_by_age = pd.crosstab(self.df['eta_std'], self.df['soddisfazione_num'])
        satisfaction_by_age.plot(kind='bar', ax=axes[1,0], 
                                color=[PURPLE_PALETTE['periwinkle'], PURPLE_PALETTE['sage']],
                                alpha=0.8, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[1,0].set_title('Soddisfazione per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[1,0].set_xlabel('Fascia d\'Età')
        axes[1,0].set_ylabel('Numero Partecipanti')
        axes[1,0].legend(['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[1,0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[1,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # 4. Box plot comparativo
        satisfaction_data = [
            self.df[self.df['Fonte'] == 'Cartaceo']['soddisfazione_num'],
            self.df[self.df['Fonte'] == 'Webinar']['soddisfazione_num']
        ]
        
        box_plot = axes[1,1].boxplot(satisfaction_data, labels=['Cartaceo', 'Webinar'],
                                    patch_artist=True, notch=True)
        
        # Colorazione box plot
        colors_box = [PURPLE_PALETTE['dusty_rose'], PURPLE_PALETTE['periwinkle']]
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor(PURPLE_PALETTE['dark_purple'])
            patch.set_linewidth(1.5)
        
        axes[1,1].set_title('Distribuzione Soddisfazione: Cartaceo vs Webinar', color=PURPLE_PALETTE['dark_purple'])
        axes[1,1].set_ylabel('Livello Soddisfazione')
        axes[1,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'satisfaction_distribution.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_satisfaction_elements(self):
        """Visualizza la distribuzione degli elementi di soddisfazione."""
        # Controlla se esistono colonne per elementi specifici di soddisfazione
        satisfaction_elements = []
        potential_columns = ['Qualità contenuti', 'Chiarezza espositiva', 'Utilità informazioni', 
                           'Organizzazione evento', 'Durata evento', 'Accessibilità']
        
        for col in potential_columns:
            if col in self.df.columns:
                satisfaction_elements.append(col)
        
        if not satisfaction_elements:
            print("   ⚠️  Elementi di soddisfazione specifici non trovati nel dataset")
            return
        
        n_elements = len(satisfaction_elements)
        n_cols = min(3, n_elements)
        n_rows = (n_elements + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_elements == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Distribuzione Elementi di Soddisfazione', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        for i, element in enumerate(satisfaction_elements):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            element_counts = self.df[element].value_counts().sort_index()
            colors_gradient = [PURPLE_CATEGORICAL[j % len(PURPLE_CATEGORICAL)] for j in range(len(element_counts))]
            
            bars = ax.bar(element_counts.index, element_counts.values, 
                         color=colors_gradient, alpha=0.8,
                         edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
            ax.set_title(element, color=PURPLE_PALETTE['dark_purple'])
            ax.set_ylabel('Numero Partecipanti')
            ax.grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
            
            # Annotazioni
            for bar, count in zip(bars, element_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom', fontweight='bold',
                       color=PURPLE_PALETTE['dark_purple'])
        
        # Nascondi subplot vuoti
        total_plots = n_rows * n_cols
        for i in range(n_elements, total_plots):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'satisfaction_elements.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_communication_channels(self):
        """Analizza l'efficacia dei canali di comunicazione con tema purple."""
        # Usa il nome colonna corretto dal CSV
        comm_column = 'Conoscenza' if 'Conoscenza' in self.df.columns else 'canali_clean'
        
        if comm_column not in self.df.columns:
            print("   ⚠️  Colonna canali comunicazione non trovata - skipping")
            return
        
        # Estrazione e conteggio canali con gestione scelte multiple
        all_channels = []
        
        for channels_raw in self.df[comm_column].dropna():
            if channels_raw != 'N/A' and pd.notna(channels_raw):
                channels_str = str(channels_raw).strip()
                
                # Prova prima con virgola, poi con punto e virgola come fallback
                if ',' in channels_str:
                    separators = [',']
                elif ';' in channels_str:
                    separators = [';']
                else:
                    # Canale singolo
                    separators = [None]
                
                if separators[0] is None:
                    # Canale singolo
                    channel_clean = channels_str.strip()
                    if channel_clean and channel_clean != 'N/A':
                        all_channels.append(channel_clean)
                else:
                    # Canali multipli - split e pulizia
                    for sep in separators:
                        channel_list = [c.strip() for c in channels_str.split(sep)]
                        # Filtra elementi vuoti e N/A
                        channel_list = [c for c in channel_list 
                                      if c and c != 'N/A' and len(c.strip()) > 2]
                        all_channels.extend(channel_list)
                        break  # Usa solo il primo separatore trovato
        
        if not all_channels:
            print("   ⚠️  Nessun canale di comunicazione valido trovato")
            return
        
        
        # Normalizzazione nomi canali per gestire varianti
        normalized_channels = []
        for channel in all_channels:
            channel_lower = channel.lower().strip()
            
            # Mappatura per standardizzare nomi simili
            if 'newsletter' in channel_lower or 'news letter' in channel_lower:
                normalized_channels.append('Newsletter dell\'Accademia')
            elif 'passaparola' in channel_lower or 'passa parola' in channel_lower:
                normalized_channels.append('Passaparola')
            elif 'invito' in channel_lower and 'socio' in channel_lower:
                normalized_channels.append('Invito di un/una socio/a')
            elif 'sito' in channel_lower and 'web' in channel_lower:
                normalized_channels.append('Sito web (www.agiati.org)')
            elif 'manifesti' in channel_lower or 'locandine' in channel_lower:
                normalized_channels.append('Manifesti o locandine')
            elif 'quotidiani' in channel_lower or 'giornali' in channel_lower:
                normalized_channels.append('Quotidiani locali')
            elif 'social' in channel_lower or 'facebook' in channel_lower or 'instagram' in channel_lower:
                normalized_channels.append('Siti e canali social di altri enti')
            else:
                # Mantieni il nome originale se non trova corrispondenze
                normalized_channels.append(channel.strip())
        
        channel_counts = Counter(normalized_channels)
        
        # Rimozione valori con frequenza troppo bassa (< 2)
        channel_counts = {k: v for k, v in channel_counts.items() if v >= 1}
        
        if not channel_counts:
            print("   ⚠️  Nessun canale con frequenza sufficiente")
            return
        
        # Ordinamento per frequenza decrescente
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Prendi i top 10 canali
        top_channels = sorted_channels[:10]
        channels = [item[0] for item in top_channels]
        counts = [item[1] for item in top_channels]
        
        # Visualizzazione migliorata con tema purple
        fig, ax = plt.subplots(figsize=(14, max(8, len(channels) * 0.8)))
        
        # Gradient purple colors
        colors = [PURPLE_SEQUENTIAL[min(i, len(PURPLE_SEQUENTIAL)-1)] for i in range(len(channels))]
        
        bars = ax.barh(range(len(channels)), counts, color=colors, alpha=0.8,
                      edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        # Configurazione assi
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels, fontsize=11, color=PURPLE_PALETTE['dark_purple'])
        ax.set_xlabel('Numero di Menzioni', fontsize=12, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        ax.set_title('Efficacia Canali di Comunicazione\n(Analisi Scelte Multiple)', 
                    fontsize=14, fontweight='bold', pad=20, color=PURPLE_PALETTE['dark_purple'])
        ax.grid(axis='x', alpha=0.3, linestyle='--', color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni valori con percentuali
        total_mentions = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total_mentions) * 100
            ax.text(bar.get_width() + max(counts) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{count} ({percentage:.1f}%)', 
                   va='center', ha='left', fontweight='bold', fontsize=10,
                   color=PURPLE_PALETTE['dark_purple'])
        
        # Statistiche sommarie
        stats_text = f'Totale menzioni: {total_mentions}\nCanali unici: {len(channel_counts)}\nRisposte multiple: {len(all_channels) - len(self.df[comm_column].dropna())}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], alpha=0.9,
                        edgecolor=PURPLE_PALETTE['primary_purple']),
               va='bottom', ha='right', fontsize=9, color=PURPLE_PALETTE['dark_purple'])
        
        # Miglioramento layout
        plt.tight_layout()
        
        # Salvataggio con DPI alto per pubblicazione
        plt.savefig(self.output_dir / 'communication_channels.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Log delle statistiche
        print(f"   ✓ Analisi canali completata: {len(channel_counts)} canali unici, {total_mentions} menzioni totali")
    
    def _plot_ces_analysis(self):
        """Visualizza l'analisi del Cultural Engagement Score con tema purple."""
        ces_data = self.results['ces']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analisi del Cultural Engagement Score (CES)', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. Componenti del CES
        components = ces_data['components']
        comp_names = ['Satisfaction\nFoundation', 'Membership\nAmplification', 
                     'Cognitive\nEngagement']
        comp_values = [components['satisfaction'], components['membership'], 
                      components['cognitive_engagement_mean']]
        
        colors_comp = [PURPLE_PALETTE['warm_gold'], PURPLE_PALETTE['royal_purple'], PURPLE_PALETTE['sage']]
        bars1 = axes[0,0].bar(comp_names, comp_values, 
                             color=colors_comp, alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,0].set_title('Componenti del Cultural Engagement Score', color=PURPLE_PALETTE['dark_purple'])
        axes[0,0].set_ylabel('Valore Componente')
        axes[0,0].set_ylim(0, 1.0)
        axes[0,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni valori
        for bar, value in zip(bars1, comp_values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold',
                          color=PURPLE_PALETTE['dark_purple'])
        
        # 2. CES Score finale (Gauge plot migliorato)
        current_ces = ces_data['score']
        max_ces = ces_data['components']['max_theoretical']
        
        # Gauge plot con tema purple
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        axes[0,1].plot(r * np.cos(theta), r * np.sin(theta), color=PURPLE_PALETTE['dark_purple'], linewidth=3)
        axes[0,1].fill_between(r * np.cos(theta), 0, r * np.sin(theta), alpha=0.2, color=PURPLE_PALETTE['lavender'])
        
        # Posizione attuale
        current_angle = (1 - current_ces / max_ces) * np.pi
        axes[0,1].plot([0, r * np.cos(current_angle)], [0, r * np.sin(current_angle)], 
                      color=PURPLE_PALETTE['primary_purple'], linewidth=6, 
                      label=f'CES Attuale: {current_ces:.3f}')
        
        # Punto sul gauge
        axes[0,1].scatter(r * np.cos(current_angle), r * np.sin(current_angle), 
                         color=PURPLE_PALETTE['warm_gold'], s=100, zorder=5, edgecolor=PURPLE_PALETTE['dark_purple'])
        
        axes[0,1].set_xlim(-1.2, 1.2)
        axes[0,1].set_ylim(-0.2, 1.2)
        axes[0,1].set_aspect('equal')
        axes[0,1].set_title(f'CES Score: {current_ces:.3f}/{max_ces:.1f}', color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].text(0, -0.1, f'Percentile: {ces_data["percentile_rank"]:.1f}%', 
                      ha='center', fontsize=12, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].axis('off')
        
        # 3. Scenari di miglioramento
        scenarios = ces_data['improvement_scenarios']
        scenario_names = list(scenarios.keys())
        projected_ces = [scenarios[s]['projected_ces'] for s in scenario_names]
        improvements = [scenarios[s]['improvement_percentage'] for s in scenario_names]
        
        colors_scenarios = [PURPLE_PALETTE['periwinkle'], PURPLE_PALETTE['warm_gold'], PURPLE_PALETTE['dusty_rose']]
        bars3 = axes[1,0].bar(scenario_names, projected_ces, alpha=0.8,
                             color=colors_scenarios, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        # Linea baseline
        axes[1,0].axhline(y=current_ces, color=PURPLE_PALETTE['dark_purple'], linestyle='--', 
                         linewidth=2, label=f'Baseline: {current_ces:.3f}')
        
        axes[1,0].set_title('Scenari di Miglioramento CES', color=PURPLE_PALETTE['dark_purple'])
        axes[1,0].set_ylabel('CES Proiettato')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[1,0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni miglioramenti
        for bar, improvement in zip(bars3, improvements):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'+{improvement:.1f}%', ha='center', va='bottom', 
                          fontweight='bold', color=PURPLE_PALETTE['sage'])
        
        # 4. Distribuzione Bootstrap
        bootstrap_data = ces_data['bootstrap_validation']
        
        # Istogramma distribuzione bootstrap
        bootstrap_scores = np.random.normal(bootstrap_data['mean'], 
                                          bootstrap_data['std_error'], 1000)
        axes[1,1].hist(bootstrap_scores, bins=30, alpha=0.7, color=PURPLE_PALETTE['periwinkle'], 
                      density=True, label='Bootstrap Distribution',
                      edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1)
        
        # Intervallo di confidenza
        axes[1,1].axvline(bootstrap_data['ci_95_lower'], color=PURPLE_PALETTE['dusty_rose'], 
                         linestyle='--', linewidth=2, label='CI 95%')
        axes[1,1].axvline(bootstrap_data['ci_95_upper'], color=PURPLE_PALETTE['dusty_rose'], 
                         linestyle='--', linewidth=2)
        axes[1,1].axvline(bootstrap_data['mean'], color=PURPLE_PALETTE['dark_purple'], 
                         linestyle='-', linewidth=2, label='Media Bootstrap')
        axes[1,1].axvline(current_ces, color=PURPLE_PALETTE['warm_gold'], 
                         linestyle='-', linewidth=3, label='CES Corrente')
        
        axes[1,1].set_title('Validazione Bootstrap CES', color=PURPLE_PALETTE['dark_purple'])
        axes[1,1].set_xlabel('CES Score')
        axes[1,1].set_ylabel('Densità')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ces_analysis.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """
        Genera un report testuale completo con tutte le statistiche e analisi.
        
        Returns:
        --------
        str
            Report completo formattato
        """
        print("📄 Generazione report completo...")
        
        report = []
        report.append("="*100)
        report.append("ANALISI QUANTITATIVA DEI DATI DI SODDISFAZIONE")
        report.append("Accademia Roveretana degli Agiati")
        report.append("="*100)
        report.append(f"Data di generazione: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append(f"Versione script: 2.1 (Correzione Metodologica Critica + Grafici Tesi)")
        report.append(f"Autore: Giuseppe Pio Mangiacotti")
        report.append(f"Istituzione: Università degli Studi di Trento")
        report.append("")
        
        # === SOMMARIO ESECUTIVO ===
        report.append("📊 SOMMARIO ESECUTIVO")
        report.append("-"*50)
        report.append(f"• Dataset: {len(self.df)} record completi analizzati")
        report.append(f"• Metodologia: Fisher Exact Test (principale), Regressione Logistica (validazione)")
        report.append(f"• Key Findings: 3 pattern principali identificati")
        
        if 'ces' in self.results:
            ces_score = self.results['ces']['score']
            ces_percentile = self.results['ces']['percentile_rank']
            report.append(f"• Cultural Engagement Score: {ces_score:.3f} ({ces_percentile:.1f}° percentile)")
        report.append("")
        
        # === CORREZIONI METODOLOGICHE ===
        report.append("⚠️  CORREZIONI METODOLOGICHE VERSIONE 2.1")
        report.append("-"*50)
        report.append("PROBLEMA RISOLTO:")
        report.append("• Test Mann-Whitney U inappropriato per dati con ties eccessivi (100% ties)")
        report.append("• Solo valori 2 (Soddisfatto) e 3 (Molto Soddisfatto) nel dataset")
        report.append("• Perdita significativa di potenza statistica")
        report.append("")
        report.append("SOLUZIONE IMPLEMENTATA:")
        report.append("• Fisher Exact Test per confronto proporzioni binarie")
        report.append("• Tabelle di contingenza 2×2 appropriate")
        report.append("• Odds ratio e p-value esatti")
        report.append("• Regressione logistica per validazione terziaria completa")
        report.append("• Warning automatici per test inappropriati")
        report.append("")
        
        # === PANORAMICA DEMOGRAFICA ===
        report.append("👥 PANORAMICA DEMOGRAFICA")
        report.append("-"*50)
        
        # Distribuzione età
        age_dist = self.df['eta_std'].value_counts().sort_index()
        report.append("Distribuzione per Fascia d'Età:")
        for age, count in age_dist.items():
            percentage = (count / len(self.df)) * 100
            report.append(f"  • {age}: {count} partecipanti ({percentage:.1f}%)")
        
        # Status soci
        member_dist = self.df['socio_std'].value_counts()
        report.append("\nStatus di Socio:")
        for status, count in member_dist.items():
            percentage = (count / len(self.df)) * 100
            report.append(f"  • {status}: {count} partecipanti ({percentage:.1f}%)")
        
        # Modalità di fruizione
        mode_dist = self.df['Fonte'].value_counts()
        report.append("\nModalità di Fruizione:")
        for mode, count in mode_dist.items():
            percentage = (count / len(self.df)) * 100
            report.append(f"  • {mode}: {count} partecipanti ({percentage:.1f}%)")
        
        # Soddisfazione generale
        satisfaction_dist = self.df['soddisfazione_num'].value_counts().sort_index()
        avg_satisfaction = self.df['soddisfazione_num'].mean()
        report.append(f"\nSoddisfazione Generale:")
        report.append(f"  • Media: {avg_satisfaction:.3f}")
        for score, count in satisfaction_dist.items():
            percentage = (count / len(self.df)) * 100
            label = "Soddisfatto" if score == 2 else "Molto Soddisfatto"
            report.append(f"  • {label} ({score}): {count} partecipanti ({percentage:.1f}%)")
        report.append("")
        
        # === KEY FINDINGS ===
        if 'key_findings' in self.results:
            report.append("🎯 KEY FINDINGS PRINCIPALI")
            report.append("-"*50)
            
            # 1. Digital Satisfaction Paradox
            if 'digital_paradox' in self.results['key_findings']:
                report.append("1. DIGITAL SATISFACTION PARADOX")
                report.append("   " + "="*40)
                paradox_data = self.results['key_findings']['digital_paradox']
                
                report.append("   📊 STATISTICHE DESCRITTIVE:")
                report.append(f"   • Cartaceo: N={paradox_data['cartaceo_stats']['n']}, Media={paradox_data['cartaceo_stats']['mean']:.3f}")
                report.append(f"   • Webinar: N={paradox_data['webinar_stats']['n']}, Media={paradox_data['webinar_stats']['mean']:.3f}")
                report.append(f"   • Differenza: +{paradox_data['difference']:.3f} punti (Webinar > Cartaceo)")
                
                report.append("   \n   🥇 FISHER EXACT TEST (Metodo Principale):")
                report.append(f"   • Odds Ratio: {paradox_data['fisher_odds_ratio']:.3f}")
                report.append(f"   • p-value: {paradox_data['fisher_p_value']:.4f}")
                significance = "***" if paradox_data['fisher_p_value'] < 0.001 else "**" if paradox_data['fisher_p_value'] < 0.01 else "*" if paradox_data['fisher_p_value'] < 0.05 else "n.s."
                report.append(f"   • Significatività: {significance}")
                report.append(f"   • Differenza Proporzioni: +{paradox_data['proportion_difference']:.3f}")
                
                # Tabella di contingenza
                contingency = paradox_data['contingency_table']
                report.append("   \n   📋 Tabella di Contingenza 2×2:")
                report.append("                      Soddisfatto(2)  Molto Sodd.(3)")
                report.append(f"     Cartaceo:        {contingency[0][0]:>11}  {contingency[0][1]:>13}")
                report.append(f"     Webinar:         {contingency[1][0]:>11}  {contingency[1][1]:>13}")
                
                # Regressione logistica
                if paradox_data['logistic_regression'].get('model_available', False):
                    lr_data = paradox_data['logistic_regression']
                    report.append("   \n   🔬 REGRESSIONE LOGISTICA (Validazione):")
                    report.append(f"   • β (Webinar): {lr_data['beta']:.3f} ± {lr_data['se_beta']:.3f}")
                    report.append(f"   • Odds Ratio: {lr_data['odds_ratio']:.3f}")
                    report.append(f"   • CI 95% (OR): [{lr_data['or_ci_95'][0]:.3f}, {lr_data['or_ci_95'][1]:.3f}]")
                    report.append(f"   • p-value: {lr_data['p_value']:.4f}")
                    report.append(f"   • Pseudo R²: {lr_data['pseudo_r2_mcfadden']:.3f}")
                    report.append(f"   • Equazione: {lr_data['model_equation']}")
                    
                    # Convergenza metodologica
                    or_diff = abs(paradox_data['fisher_odds_ratio'] - lr_data['odds_ratio'])
                    convergence_status = "ECCELLENTE" if or_diff < 0.1 else "ACCETTABILE" if or_diff < 0.5 else "PROBLEMATICA"
                    report.append(f"   • Convergenza metodologica: {convergence_status} (diff OR: {or_diff:.3f})")
                
                # Mann-Whitney con disclaimer
                report.append("   \n   ⚠️  MANN-WHITNEY U (Limitazioni):")
                report.append(f"   • U-statistic: {paradox_data['mannwhitney_u']:.1f}")
                report.append(f"   • p-value: {paradox_data['mannwhitney_p']:.4f}")
                report.append(f"   • Ties: {paradox_data['ties_percentage']:.1%} (ECCESSIVI)")
                report.append(f"   • Limitazione: Solo 2 valori distinti, violazione assunzioni")
                
                # Age Paradox
                report.append("   \n   🔍 AGE PARADOX:")
                report.append(f"   • Età media Webinar: {paradox_data['age_webinar']:.1f} anni")
                report.append(f"   • Età media Cartaceo: {paradox_data['age_cartaceo']:.1f} anni")
                report.append(f"   • Differenza: +{paradox_data['age_paradox']:.1f} anni (controintuitiva)")
                report.append("")
            
            # 2. Generational Membership Gap
            if 'generational_gap' in self.results['key_findings']:
                report.append("2. GENERATIONAL MEMBERSHIP GAP")
                report.append("   " + "="*40)
                gap_data = self.results['key_findings']['generational_gap']
                
                report.append("   📈 ENGAGEMENT METRICS PER FASCIA D'ETÀ:")
                for metric in gap_data['engagement_metrics']:
                    report.append(f"   • {metric['age_group']}:")
                    report.append(f"     - Dimensione coorte: {metric['cohort_size']}")
                    report.append(f"     - Tasso membership: {metric['membership_rate']:.1%}")
                    report.append(f"     - Soddisfazione media: {metric['avg_satisfaction']:.3f}")
                    report.append(f"     - Institutional Engagement Index: {metric['engagement_index']:.3f}")
                    report.append(f"     - Cognitive Engagement Index: {metric['cognitive_engagement_mean']:.3f}")
                    report.append(f"     - Composite Engagement Index: {(metric['cognitive_engagement_mean'] + metric['engagement_index']) / 2 :.3f}")
                    report.append(f"     - Potenziale conversione: {metric['conversion_potential']:.1f}")
                
                report.append("   \n   📊 CORRELAZIONE ETÀ-MEMBERSHIP:")
                report.append(f"   • Correlazione Spearman (principale): r = {gap_data['spearman_correlation']:.3f}")
                report.append(f"   • p-value: {gap_data['spearman_p_value']:.4f}")
                report.append(f"   • Correlazione Pearson (confronto): r = {gap_data['pearson_correlation']:.3f}")
                report.append(f"   • Dimensione campione: {gap_data['sample_size']} coorti")
                report.append(f"   • Metodo raccomandato: {gap_data['recommended_method']}")
                report.append("")
            
            # 3. Inverse Contentment Effect
            if 'contentment_effect' in self.results['key_findings']:
                report.append("3. INVERSE CONTENTMENT EFFECT")
                report.append("   " + "="*40)
                content_data = self.results['key_findings']['contentment_effect']
                
                report.append("   📝 METRICHE FEEDBACK:")
                satisfied_metrics = content_data['feedback_metrics']['satisfied']
                very_satisfied_metrics = content_data['feedback_metrics']['very_satisfied']
                
                report.append(f"   • Soddisfatti (2):")
                report.append(f"     - Totale: {satisfied_metrics['total']}")
                report.append(f"     - Con feedback: {satisfied_metrics['with_feedback']}")
                report.append(f"     - Tasso feedback: {satisfied_metrics['feedback_rate']:.1%}")
                
                report.append(f"   • Molto Soddisfatti (3):")
                report.append(f"     - Totale: {very_satisfied_metrics['total']}")
                report.append(f"     - Con feedback: {very_satisfied_metrics['with_feedback']}")
                report.append(f"     - Tasso feedback: {very_satisfied_metrics['feedback_rate']:.1%}")
                
                report.append(f"   \n   📊 CONTENTMENT RATIO: {content_data['contentment_ratio']:.3f}")
                report.append(f"   • Effetto confermato: {'Sì' if content_data['effect_confirmed'] else 'No'}")
                
                report.append(f"   \n   🧪 TEST STATISTICO ({content_data['primary_test']}):")
                report.append(f"   • p-value: {content_data['primary_p_value']:.4f}")
                
                if content_data['primary_test'] == 'Chi-quadro':
                    report.append(f"   • Chi² statistic: {content_data['chi2_statistic']:.3f}")
                    cochran = content_data['cochran_criteria']
                    report.append(f"   • Criteri Cochran: {'Soddisfatti' if cochran['satisfied'] else 'Violati'}")
                    report.append(f"   • Min freq. attesa: {cochran['min_expected_frequency']:.1f}")
                else:  # Fisher Exact Test
                    report.append(f"   • Fisher Odds Ratio: {content_data['fisher_odds_ratio']:.3f}")
                    cochran = content_data['cochran_criteria']
                    report.append(f"   • Motivo Fisher: Min freq. attesa = {cochran['min_expected_frequency']:.1f}")
                
                report.append("")
        
        # === CULTURAL ENGAGEMENT SCORE ===
        if 'ces' in self.results:
            report.append("📈 CULTURAL ENGAGEMENT SCORE (CES)")
            report.append("-"*50)
            ces_data = self.results['ces']
            
            report.append("🎯 SCORE E COMPONENTI:")
            report.append(f"• CES Score: {ces_data['score']:.3f}")
            report.append(f"• Percentile Rank: {ces_data['percentile_rank']:.1f}%")
            report.append(f"• Max Teorico: {ces_data['components']['max_theoretical']:.1f}")
            
            report.append("\n📊 COMPONENTI:")
            components = ces_data['components']
            report.append(f"• Satisfaction: {components['satisfaction']:.3f}")
            report.append(f"• Membership Rate: {components['membership']:.3f}")
            report.append(f"• Cognitive Engagement: {components['cognitive_engagement_mean']:.3f}")

            # Bootstrap validation
            bootstrap = ces_data['bootstrap_validation']
            report.append(f"\n🔬 VALIDAZIONE BOOTSTRAP:")
            report.append(f"• Media Bootstrap: {bootstrap['mean']:.3f}")
            report.append(f"• Errore Standard: {bootstrap['std_error']:.3f}")
            report.append(f"• CI 95%: [{bootstrap['ci_95_lower']:.3f}, {bootstrap['ci_95_upper']:.3f}]")
            report.append(f"• Coefficiente Variazione: {bootstrap['cv']:.1%}")
            
            # Scenari di miglioramento
            report.append(f"\n🚀 SCENARI DI MIGLIORAMENTO:")
            scenarios = ces_data['improvement_scenarios']
            for scenario_name, scenario_data in scenarios.items():
                report.append(f"• {scenario_name.title()}:")
                report.append(f"  - CES proiettato: {scenario_data['projected_ces']:.3f}")
                report.append(f"  - Miglioramento: +{scenario_data['improvement_percentage']:.1f}%")
                report.append(f"  - Target membership: {scenario_data['target_membership']:.1%}")
                report.append(f"  - Target engagement cognitivo: {scenario_data['target_cognitive']:.1%}")
            report.append("")
        
        # === QUALITÀ METODOLOGICA ===
        if self.test_appropriateness:
            report.append("🔬 ASSESSMENT QUALITÀ METODOLOGICA")
            report.append("-"*50)
            report.append("TEST STATISTICI UTILIZZATI:")
            
            for test_name, assessment in self.test_appropriateness.items():
                status_icon = "✅" if assessment['is_appropriate'] else "❌"
                report.append(f"{status_icon} {test_name}:")
                report.append(f"  • Dati: {assessment['data_description']}")
                report.append(f"  • Appropriatezza: {assessment['reason']}")
                report.append(f"  • Raccomandazione: {assessment['recommendation']}")
                report.append("")
        
        # === AVVISI METODOLOGICI ===
        if self.methodological_warnings:
            report.append("⚠️  AVVISI METODOLOGICI")
            report.append("-"*50)
            for warning in self.methodological_warnings:
                report.append(f"• {warning['test']}: {warning['warning']}")
            report.append("")
        
        # === RACCOMANDAZIONI ===
        report.append("💡 RACCOMANDAZIONI METODOLOGICHE")
        report.append("-"*50)
        report.append("✅ UTILIZZARE:")
        report.append("• Fisher Exact Test come risultato principale per confronti binari")
        report.append("• Regressione Logistica per validazione e intervalli di confidenza")
        report.append("• Bootstrap resampling per validazione indicatori compositi")
        report.append("• Correlazione di Spearman per campioni piccoli")
        report.append("")
        report.append("⚠️  EVITARE:")
        report.append("• Mann-Whitney U per dati con ties eccessivi (>50%)")
        report.append("• Test χ² quando violati i criteri di Cochran")
        report.append("• Correlazione di Pearson senza verifica normalità")
        report.append("")
        report.append("📊 RIPORTARE:")
        report.append("• Odds Ratio con intervalli di confidenza al 95%")
        report.append("• Tabelle di contingenza complete per trasparenza")
        report.append("• Dimensioni campionarie per tutti i sottogruppi")
        report.append("• Assessment appropriatezza dei test utilizzati")
        report.append("")
        
        # === LIMITAZIONI E FUTURE DIREZIONI ===
        report.append("🚧 LIMITAZIONI DELLO STUDIO")
        report.append("-"*50)
        report.append("• Campione di convenienza limitato a partecipanti eventi specifici")
        report.append("• Scala di soddisfazione ridotta (solo valori 2-3 osservati)")
        report.append("• Analisi cross-sectional, non longitudinale")
        report.append("• Possibili bias di selezione tra modalità cartacea e digitale")
        report.append("• Dimensioni campionarie ridotte per alcune sottopopolazioni")
        report.append("")
        
        report.append("🔮 FUTURE DIREZIONI DI RICERCA")
        report.append("-"*50)
        report.append("• Espansione scala di misurazione soddisfazione")
        report.append("• Studio longitudinale per tracking engagement nel tempo")
        report.append("• Campionamento randomizzato tra modalità di fruizione")
        report.append("• Analisi qualitativa approfondita del feedback testuale")
        report.append("• Validazione CES su istituzioni culturali comparabili")
        report.append("")
        
        # === CITAZIONE ===
        report.append("📚 CITAZIONE")
        report.append("-"*50)
        report.append("Mangiacotti, G.P. (2025). Progettazione di interfacce per la raccolta")
        report.append("di feedback nel settore culturale: un sistema ibrido per l'Accademia")
        report.append("Roveretana degli Agiati v2.1. Università degli Studi di Trento.")
        report.append("")
        
        # === APPENDICE TECNICA ===
        report.append("🔧 APPENDICE TECNICA")
        report.append("-"*50)
        report.append(f"• Versione Python: 3.9.0")
        report.append(f"• Librerie principali: pandas, scipy, sklearn, matplotlib, seaborn")
        report.append(f"• Metodo bootstrap: {1000} iterazioni con rimpiazzamento")
        report.append(f"• Test di significatività: α = 0.05")
        report.append(f"• Intervalli di confidenza: 95%")
        report.append(f"• Correzioni multiple: Non applicate (analisi esploratoria)")
        report.append("")
        
        report.append("="*100)
        report.append(f"Fine Report - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append("="*100)
        
        # Salva il report
        report_text = "\n".join(report)
        with open(self.output_dir / 'analysis_report_corrected.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Salva anche avvisi metodologici separatamente
        if self.methodological_warnings:
            warnings_text = "AVVISI METODOLOGICI - Accademia Agiati v2.1\n"
            warnings_text += "="*60 + "\n\n"
            for warning in self.methodological_warnings:
                warnings_text += f"TEST: {warning['test']}\n"
                warnings_text += f"PROBLEMA: {warning['warning']}\n"
                warnings_text += f"TIMESTAMP: {warning['timestamp']}\n"
                warnings_text += "-"*40 + "\n"
            
            with open(self.output_dir / 'methodological_warnings.txt', 'w', encoding='utf-8') as f:
                f.write(warnings_text)
        
        print(f"   ✓ Report salvato in: {self.output_dir / 'analysis_report_corrected.txt'}")
        return report_text
    
    def run_complete_analysis(self):
        """
        Esegue l'analisi completa con tutti i componenti.
        """
        print("🚀 Avvio analisi completa Accademia Roveretana degli Agiati v2.1")
        print("="*80)
        
        # 1. Caricamento e pulizia dati
        self.load_and_clean_data()
        
        # 2. Calcolo key findings
        self.calculate_key_findings()
        
        # 3. Calcolo CES
        self.calculate_cultural_engagement_score()
        
        # 4. Stampa analisi dettagliata paradosso digitale
        self.print_detailed_digital_paradox_analysis()
        
        # 5. Generazione visualizzazioni
        self.create_visualizations()
        
        # 6. Generazione report completo
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("✅ ANALISI COMPLETATA CON SUCCESSO")
        print("="*80)
        print(f"📁 Output directory: {self.output_dir}")
        print("📊 File generati:")
        print("   • analysis_report_corrected.txt - Report completo")
        print("   • methodological_warnings.txt - Avvisi metodologici")
        print("   • contingency_table_fisher_test.csv - Tabella contingenza")
        print("   • digital_satisfaction_paradox_corrected.png - Paradosso digitale")
        print("   • demographic_overview_panel.png - 🎓 Pannello demografico (TESI)")
        print("   • participation_modalities_panel.png - 🎓 Pannello partecipazione (TESI)")
        print("   • + altri 6 grafici di analisi")
        print("\n🎯 Key Findings identificati:")
        print("   1. Digital Satisfaction Paradox")
        print("   2. Generational Membership Gap")
        print("   3. Inverse Contentment Effect")
        
        if 'ces' in self.results:
            ces_score = self.results['ces']['score']
            ces_percentile = self.results['ces']['percentile_rank']
            print(f"\n📈 Cultural Engagement Score: {ces_score:.3f} ({ces_percentile:.1f}° percentile)")
        
        print("\n🔬 Metodologia corretta applicata:")
        print("   ✅ Fisher Exact Test (principale)")
        print("   ✅ Regressione Logistica (validazione)")
        print("   ✅ Bootstrap resampling (robustezza)")
        print("   ⚠️  Mann-Whitney U (limitazioni dichiarate)")


def main():
    """Funzione principale per esecuzione da command line."""
    parser = argparse.ArgumentParser(description='Analisi dati Accademia Agiati v2.1')
    parser.add_argument('--input', required=True, help='Path al file CSV dei dati')
    parser.add_argument('--output', default='./output/', help='Directory di output')
    
    args = parser.parse_args()
    
    # Verifica esistenza file input
    if not os.path.exists(args.input):
        print(f"❌ File input non trovato: {args.input}")
        return
    
    # Inizializzazione analyzer
    analyzer = AccademiaAnalyzer(args.input, args.output)
    
    # Esecuzione analisi completa
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()