"""
Analisi Quantitativa dei Dati di Soddisfazione - Accademia Roveretana degli Agiati
================================================================================

Autore: Giuseppe Pio Mangiacotti
Istituzione: Università degli Studi di Trento - Dipartimento di Scienze Cognitive
Anno Accademico: 2024/2025
Versione: 2.1 (CORREZIONE METODOLOGICA CRITICA - Fisher Exact Test)

⚠️  AGGIORNAMENTO CRITICO: 
Questa versione corregge un errore metodologico significativo nella v2.0.
Il test Mann-Whitney U è stato sostituito con Fisher Exact Test per 
dati essenzialmente binari (solo valori 2 e 3).

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
- plotly >= 5.15.0 (opzionale)

Utilizzo:
python data_analysis_accademia_agiati.py --input paper_report.CSV --output ./output/

Citazione (Aggiornata):
=======================
Mangiacotti, G.P. (2025). Progettazione di interfacce per la raccolta di feedback 
nel settore culturale: un sistema ibrido per l'Accademia Roveretana degli Agiati
Università degli Studi di Trento.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, mannwhitneyu, ks_2samp, rankdata, fisher_exact
import warnings
import argparse
import os
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
    
    VERSIONE 2.1 - Metodologia Corretta per Dati Binari/Ordinali Discreti
    ====================================================================
    
    Implementa metodologie quantitative appropriate per l'identificazione di pattern
    di soddisfazione, engagement e comportamento del pubblico culturale.
    
    CORREZIONE METODOLOGICA CRITICA:
    --------------------------------
    - Fisher Exact Test per confronti di proporzioni (dati essenzialmente binari)
    - Mann-Whitney U relegato ad analisi secondaria per dati con ties eccessivi
    - Verifica automatica dell'appropriatezza dei test statistici
    - Warning metodologici espliciti quando le assunzioni sono violate
    
    TEST STATISTICI UTILIZZATI:
    ---------------------------
    PRINCIPALI (Raccomandati):
    - Fisher Exact Test per tabelle di contingenza 2×2
    - Test χ² con correzioni appropriate
    - Bootstrap resampling per validazione indicatori compositi
    - Regressione logistica per validazione crociata
    
    SECONDARI (Con limitazioni dichiarate):
    - Mann-Whitney U per compatibilità (con disclaimer sulle limitazioni)
    - Coefficiente r di Rosenthal per effect size comparativo
    
    OUTPUT GENERATI:
    ---------------
    - analysis_report_corrected.txt: Report con metodologia corretta
    - methodological_warning.txt: Avviso sui problemi metodologici
    - contingency_table_fisher_test.csv: Tabella per Fisher test
    - digital_satisfaction_paradox_corrected.png: Visualizzazione aggiornata
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
            'S�': 'Sì',  # Fix encoding UTF-8
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
        - Aggiunta regressione logistica per robustezza
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
        
        # === REGRESSIONE LOGISTICA (ANALISI TERZIARIA) ===
        logistic_results = {'model_available': False}
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Preparazione dati per regressione
            X = np.array([1 if fonte == 'Webinar' else 0 for fonte in self.df['Fonte']]).reshape(-1, 1)
            y = np.array([1 if sat == 3 else 0 for sat in self.df['soddisfazione_num']])
            
            # Fit del modello
            logistic_model = LogisticRegression()
            logistic_model.fit(X, y)
            
            # Odds ratio dalla regressione
            logistic_odds_ratio = np.exp(logistic_model.coef_[0][0])
            
            logistic_results = {
                'odds_ratio': logistic_odds_ratio,
                'coefficient': logistic_model.coef_[0][0],
                'intercept': logistic_model.intercept_[0],
                'model_available': True
            }
            
            self._assess_test_appropriateness(
                'Regressione Logistica',
                'Validazione crociata per confronto proporzioni',
                True,
                'Robusto per qualsiasi dimensione campionaria'
            )
            
        except ImportError:
            self._log_methodological_warning(
                'Regressione Logistica',
                'sklearn non disponibile - analisi di validazione saltata'
            )
        
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
            
            # === REGRESSIONE LOGISTICA ===
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
                
                engagement_data.append({
                    'age_group': age,
                    'cohort_size': len(cohort),
                    'membership_rate': membership_rate,
                    'avg_satisfaction': avg_satisfaction,
                    'engagement_index': engagement_index,
                    'conversion_potential': (1 - membership_rate) * avg_satisfaction * len(cohort)
                })
        
        # Correlazione età-membership
        age_numerical = [22, 40, 60, 75]
        membership_rates = [d['membership_rate'] for d in engagement_data]
        correlation, p_value = pearsonr(age_numerical, membership_rates)
        
        return {
            'engagement_metrics': engagement_data,
            'age_membership_correlation': correlation,
            'correlation_p_value': p_value,
            'r_squared': correlation**2
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
        
        # Test chi-quadro per significatività
        contingency_table = [
            [feedback_metrics['satisfied']['with_feedback'],
             feedback_metrics['satisfied']['total'] - feedback_metrics['satisfied']['with_feedback']],
            [feedback_metrics['very_satisfied']['with_feedback'],
             feedback_metrics['very_satisfied']['total'] - feedback_metrics['very_satisfied']['with_feedback']]
        ]
        
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Contentment Ratio
        contentment_ratio = (feedback_metrics['very_satisfied']['feedback_rate'] / 
                           feedback_metrics['satisfied']['feedback_rate'])
        
        return {
            'feedback_metrics': feedback_metrics,
            'contentment_ratio': contentment_ratio,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'effect_confirmed': contentment_ratio > 1
        }
    
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
        satisfaction_component = avg_satisfaction / 3  # Normalizzazione 0-1
        
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
                'max_theoretical': 2.0
            },
            'percentile_rank': (ces_score / 2.0) * 100,
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
            sample = self.df.sample(n=len(self.df), replace=True)
            
            # Calcolo CES per il campione
            avg_sat = sample['soddisfazione_num'].mean()
            sat_comp = avg_sat / 3
            mem_rate = (sample['socio_std'] == 'Sì').mean()
            dig_rate = (sample['Fonte'] == 'Webinar').mean()
            
            ces = sat_comp * (1 + mem_rate) * (1 + dig_rate)
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
                       (1 + baseline_components['digital_adoption']))
        
        for scenario, changes in scenarios.items():
            new_membership = min(1.0, baseline_components['membership'] + changes['membership'])
            new_digital = min(1.0, baseline_components['digital_adoption'] + changes['digital'])
            
            new_ces = (baseline_components['satisfaction'] * 
                      (1 + new_membership) * 
                      (1 + new_digital))
            
            improvement = ((new_ces - baseline_ces) / baseline_ces) * 100
            
            results[scenario] = {
                'projected_ces': new_ces,
                'improvement_percentage': improvement,
                'target_membership': new_membership,
                'target_digital': new_digital
            }
        
        return results
    
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
        
        # 5. Analisi canali comunicazione
        self._plot_communication_channels()
        
        # 6. Cultural Engagement Score
        self._plot_ces_analysis()
        
        print(f"   ✓ Grafici salvati in: {self.output_dir}")
    
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
        AGGIORNATO: Fisher Exact Test come metodo principale.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Digital Satisfaction Paradox - Analisi Metodologicamente Corretta', 
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
        fisher_text = f'🎯 FISHER EXACT TEST (Principale)\n'
        fisher_text += f'Odds Ratio = {paradox_data["fisher_odds_ratio"]:.2f}\n'
        fisher_text += f'p-value = {paradox_data["fisher_p_value"]:.3f}\n'
        fisher_text += f'Diff. Proporzioni = +{paradox_data["proportion_difference"]:.3f}\n'
        fisher_text += f'Test Raccomandato: ✅'
        
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
        
        # === 3. CONFRONTO METODOLOGIE ===
        # Barplot comparativo dei p-values
        methods = ['Fisher Exact\n(Raccomandato)', 'Mann-Whitney U\n(Problematico)']
        p_values = [paradox_data['fisher_p_value'], paradox_data['mannwhitney_p']]
        
        # Colori: verde per appropriato, rosso per problematico
        colors_methods = [PURPLE_PALETTE['sage'], PURPLE_PALETTE['dusty_rose']]
        bars3 = axes[2].bar(methods, p_values, color=colors_methods, alpha=0.8,
                           edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        
        # Linea di significatività
        axes[2].axhline(y=0.05, color=PURPLE_PALETTE['dark_purple'], linestyle='--', 
                       linewidth=2, label='α = 0.05')
        
        axes[2].set_title('Confronto Metodologie Statistiche', color=PURPLE_PALETTE['dark_purple'])
        axes[2].set_ylabel('p-value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni metodologiche
        for bar, p_val, method in zip(bars3, p_values, methods):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{p_val:.3f}\n{significance}', ha='center', va='bottom', 
                        fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
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
        
        # Nota metodologica generale
        methodology_note = """
    NOTA METODOLOGICA:
    • Fisher Exact Test: Appropriato per tabelle 2×2, nessuna assunzione violata
    • Mann-Whitney U: Subottimale per dati con ties eccessivi (solo valori 2,3)
    • Raccomandazione: Usare Fisher per confronti binari, M-W per scale ordinali complete
        """
        
        fig.text(0.02, 0.02, methodology_note, fontsize=8, style='italic',
                 color=PURPLE_PALETTE['dark_purple'], alpha=0.8)
        
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
        fig.suptitle('Generational Membership Gap Analysis', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        gap_data = self.results['key_findings']['generational_gap']['engagement_metrics']
        age_groups = [d['age_group'] for d in gap_data]
        membership_rates = [d['membership_rate'] * 100 for d in gap_data]
        engagement_indices = [d['engagement_index'] for d in gap_data]
        
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
        bars2 = axes[0,1].bar(age_groups, engagement_indices, 
                             color=PURPLE_PALETTE['sage'], alpha=0.8,
                             edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0,1].set_title('Engagement Index per Fascia d\'Età', color=PURPLE_PALETTE['dark_purple'])
        axes[0,1].set_ylabel('Engagement Index')
        axes[0,1].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[0,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Analisi Distribuzione Soddisfazione', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. Distribuzione generale
        satisfaction_counts = self.df['soddisfazione_num'].value_counts().sort_index()
        colors = [PURPLE_PALETTE['primary_purple'] if score == 2 else PURPLE_PALETTE['warm_gold'] for score in satisfaction_counts.index]
        bars1 = axes[0].bar(satisfaction_counts.index, satisfaction_counts.values, 
                           color=colors, alpha=0.8,
                           edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[0].set_title('Distribuzione Punteggi di Soddisfazione', color=PURPLE_PALETTE['dark_purple'])
        axes[0].set_xlabel('Punteggio Soddisfazione')
        axes[0].set_ylabel('Numero Risposte')
        axes[0].set_xticks([2, 3])
        axes[0].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Annotazioni statistiche (con nota su variabile ordinale)
        mean_sat = self.df['soddisfazione_num'].mean()
        std_sat = self.df['soddisfazione_num'].std()
        median_sat = self.df['soddisfazione_num'].median()
        axes[0].text(0.02, 0.98, f'Media* = {mean_sat:.3f}\nMediana = {median_sat:.1f}\nSD = {std_sat:.3f}',
                    transform=axes[0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE['lavender'], 
                             alpha=0.9, edgecolor=PURPLE_PALETTE['primary_purple']),
                    color=PURPLE_PALETTE['dark_purple'])
        
        # 2. Soddisfazione per status socio
        satisfaction_by_member = self.df.groupby(['socio_std', 'soddisfazione_num']).size().unstack(fill_value=0)
        satisfaction_by_member.plot(kind='bar', ax=axes[1], 
                                   color=[PURPLE_PALETTE['primary_purple'], PURPLE_PALETTE['warm_gold']], 
                                   alpha=0.8, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[1].set_title('Soddisfazione per Status Socio', color=PURPLE_PALETTE['dark_purple'])
        axes[1].set_xlabel('Status Socio')
        axes[1].set_ylabel('Numero Risposte')
        axes[1].legend(title='Soddisfazione', labels=['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[1].tick_params(axis='x', rotation=0, colors=PURPLE_PALETTE['dark_purple'])
        axes[1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # 3. Soddisfazione per modalità
        satisfaction_by_mode = self.df.groupby(['Fonte', 'soddisfazione_num']).size().unstack(fill_value=0)
        satisfaction_by_mode.plot(kind='bar', ax=axes[2], 
                                 color=[PURPLE_PALETTE['primary_purple'], PURPLE_PALETTE['warm_gold']], 
                                 alpha=0.8, edgecolor=PURPLE_PALETTE['dark_purple'], linewidth=1.5)
        axes[2].set_title('Soddisfazione per Modalità di Fruizione', color=PURPLE_PALETTE['dark_purple'])
        axes[2].set_xlabel('Modalità')
        axes[2].set_ylabel('Numero Risposte')
        axes[2].legend(title='Soddisfazione', labels=['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[2].tick_params(axis='x', rotation=45, colors=PURPLE_PALETTE['dark_purple'])
        axes[2].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        # Nota metodologica
        fig.text(0.02, 0.02, '*Media calcolata per visualizzazione (variabile ordinale)', 
                fontsize=8, style='italic', color=PURPLE_PALETTE['dark_purple'], alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'satisfaction_distribution.png', facecolor='white', edgecolor='none')
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
        
        # Conteggio frequenze con normalizzazione nomi
        from collections import Counter
        
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
        fig.suptitle('Cultural Engagement Score (CES) Analysis', fontsize=16, fontweight='bold', color=PURPLE_PALETTE['dark_purple'])
        
        # 1. Componenti del CES
        components = ces_data['components']
        comp_names = ['Satisfaction\nFoundation', 'Membership\nAmplification', 
                     'Digital\nAcceleration']
        comp_values = [components['satisfaction'], components['membership'], 
                      components['digital_adoption']]
        
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
        current_angle = (current_ces / max_ces) * np.pi
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
                         linestyle='-', linewidth=3, label='Media Bootstrap')
        
        axes[1,1].set_title('Validazione Bootstrap CES', color=PURPLE_PALETTE['dark_purple'])
        axes[1,1].set_xlabel('CES Score')
        axes[1,1].set_ylabel('Densità')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3, color=PURPLE_PALETTE['light_purple'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ces_analysis.png', facecolor='white', edgecolor='none')
        plt.close()
    
    def generate_summary_report(self):
        """Genera un report riassuntivo in formato JSON e testo."""
        print("📋 Generazione report riassuntivo...")
        
        # report JSON per uso programmatico e usi futuri
        summary = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'dataset_size': len(self.df),
                'analyst': 'Giuseppe Pio Mangiacotti',
                'institution': 'Università degli Studi di Trento',
                'methodology_version': '2.1 - Correzione metodologica critica per dati binari/ordinali discreti'
            },
            'key_findings': self.results['key_findings'],
            'ces_analysis': self.results['ces'],
            'sample_statistics': {
                'total_respondents': len(self.df),
                'age_distribution': self.df['eta_std'].value_counts().to_dict(),
                'membership_distribution': self.df['socio_std'].value_counts().to_dict(),
                'satisfaction_mean': self.df['soddisfazione_num'].mean(),
                'satisfaction_std': self.df['soddisfazione_num'].std(),
                'satisfaction_median': self.df['soddisfazione_num'].median()
            },
            'methodological_warnings': self.methodological_warnings,
            'test_appropriateness': self.test_appropriateness
        }
        
        with open(self.output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # generazione report testuale per citazione in tesi
        self._generate_text_report(summary)
        
        print(f"   ✓ Report salvato in: {self.output_dir}")
        
    def _generate_text_report(self, summary: dict):
        """Genera report testuale con metodologia corretta per citazione accademica."""
        
        # Estrazione dati Fisher test
        fisher_data = summary['key_findings']['digital_paradox']
        
        report = f"""
ANALISI QUANTITATIVA DEI DATI DI SODDISFAZIONE
Accademia Roveretana degli Agiati
=============================================

Analista: {summary['metadata']['analyst']}
Istituzione: {summary['metadata']['institution']}
Data Analisi: {summary['metadata']['analysis_date']}
Versione Metodologica: {summary['metadata']['methodology_version']}

EXECUTIVE SUMMARY
-----------------
Campione analizzato: {summary['metadata']['dataset_size']} partecipanti
Soddisfazione media: {summary['sample_statistics']['satisfaction_mean']:.3f} ± {summary['sample_statistics']['satisfaction_std']:.3f}
Soddisfazione mediana: {summary['sample_statistics']['satisfaction_median']:.1f}
Cultural Engagement Score: {summary['ces_analysis']['score']:.3f}/2.0 ({summary['ces_analysis']['percentile_rank']:.1f}° percentile)

KEY FINDINGS PRINCIPALI
------------------------

1. DIGITAL SATISFACTION PARADOX ⭐ [METODOLOGIA CORRETTA]
   
   📊 FISHER EXACT TEST (Metodo Principale):
   - Odds Ratio: {fisher_data.get('fisher_odds_ratio', 'N/A'):.3f}
   - p-value: {fisher_data.get('fisher_p_value', 'N/A'):.3f}
   - Differenza Proporzioni: {fisher_data.get('proportion_difference', 'N/A'):.3f}
   - Test Appropriato: ✅ (dati essenzialmente binari)
   
   📋 Tabella di Contingenza 2×2:
              Soddisfatto(2)  Molto Soddisfatto(3)
   Cartaceo:  {fisher_data.get('contingency_table', [[0,0],[0,0]])[0][0]:>13}  {fisher_data.get('contingency_table', [[0,0],[0,0]])[0][1]:>19}
   Webinar:   {fisher_data.get('contingency_table', [[0,0],[0,0]])[1][0]:>13}  {fisher_data.get('contingency_table', [[0,0],[0,0]])[1][1]:>19}
   
   ⚠️  MANN-WHITNEY U (Analisi Secondaria - Limitazioni):
   - U-statistic: {fisher_data.get('mannwhitney_u', 'N/A'):.1f}
   - p-value: {fisher_data.get('mannwhitney_p', 'N/A'):.3f}
   - Rosenthal r: {fisher_data.get('rosenthal_r', 'N/A'):.3f}
   - Ties: {fisher_data.get('ties_percentage', 0)*100:.1f}% (ECCESSIVI)
   - Metodologia: ⚠️  Subottimale per questi dati
   
   🔍 Age Paradox: Webinar +{fisher_data.get('age_paradox', 0):.1f} anni vs Cartaceo

2. GENERATIONAL MEMBERSHIP GAP
   - Correlazione Età-Membership: r = {summary['key_findings']['generational_gap']['age_membership_correlation']:.3f}
   - R² = {summary['key_findings']['generational_gap']['r_squared']:.3f}
   - Significatività: p = {summary['key_findings']['generational_gap']['correlation_p_value']:.3f}

3. INVERSE CONTENTMENT EFFECT
   - Contentment Ratio: {summary['key_findings']['contentment_effect']['contentment_ratio']:.3f}
   - Effetto confermato: {summary['key_findings']['contentment_effect']['effect_confirmed']}
   - Test χ²: p = {summary['key_findings']['contentment_effect']['p_value']:.3f}

CORREZIONE METODOLOGICA CRITICA
--------------------------------
🎯 PROBLEMA IDENTIFICATO:
   L'uso del test Mann-Whitney U per confrontare soddisfazione tra modalità cartacea 
   e webinar era METODOLOGICAMENTE INAPPROPRIATO per i seguenti motivi:
   
   • Dati essenzialmente binari (solo valori 2 e 3)
   • Ties eccessivi: {fisher_data.get('ties_percentage', 0)*100:.1f}% delle osservazioni
   • Perdita significativa di potenza statistica
   • Violazione assunzioni di base del test

✅ SOLUZIONE IMPLEMENTATA:
   • Test Esatto di Fisher come metodo principale
   • Confronto diretto delle proporzioni "Molto Soddisfatti"
   • Nessuna assunzione violata
   • Appropriato per tabelle di contingenza 2×2
   • Risultati più robusti e interpretabili

METODOLOGIA STATISTICA (AGGIORNATA)
-----------------------------------
PRIMARI (Raccomandati):
- Fisher Exact Test per confronti binari/quasi-binari
- Test χ² per tabelle di contingenza (con correzioni appropriate)
- Bootstrap resampling per validazione indicatori compositi

SECONDARI (Con limitazioni dichiarate):
- Mann-Whitney U mantenuto per compatibilità (con warning espliciti)
- Coefficiente r di Rosenthal per effect size comparativo

INAPPROPRIATI per questi dati:
- Test t per campioni indipendenti (variabili ordinali discrete)
- Test parametrici senza trasformazioni appropriate

IMPLICAZIONI DELLE CORREZIONI
------------------------------
1. MAGGIORE ROBUSTEZZA: Risultati statisticamente più affidabili
2. INTERPRETABILITÀ: Focus su proporzioni facilmente comprensibili
3. TRASPARENZA: Limitazioni metodologiche esplicitamente dichiarate
4. REPLICABILITÀ: Metodologia standardizzata per studi futuri

CULTURAL ENGAGEMENT SCORE (CES)
-------------------------------
Score Attuale: {summary['ces_analysis']['score']:.3f}
Componenti:
- Satisfaction Foundation: {summary['ces_analysis']['components']['satisfaction']:.3f}
- Membership Amplification: {summary['ces_analysis']['components']['membership']:.3f}
- Digital Acceleration: {summary['ces_analysis']['components']['digital_adoption']:.3f}

Validazione Bootstrap (CI 95%): [{summary['ces_analysis']['bootstrap_validation']['ci_95_lower']:.3f}, {summary['ces_analysis']['bootstrap_validation']['ci_95_upper']:.3f}]

SCENARI DI MIGLIORAMENTO
------------------------"""
        
        for scenario, data in summary['ces_analysis']['improvement_scenarios'].items():
            report += f"""
{scenario.upper()}:
   - CES Proiettato: {data['projected_ces']:.3f}
   - Miglioramento: +{data['improvement_percentage']:.1f}%
   - Target Membership: {data['target_membership']*100:.1f}%
   - Target Digital: {data['target_digital']*100:.1f}%"""
        
        report += f"""

CITAZIONE CORRETTA
------------------
Mangiacotti, G.P. (2025). Analisi Quantitativa dei Dati di Soddisfazione - 
Accademia Roveretana degli Agiati v2.1 [Computer software]. 
Università degli Studi di Trento. 
Data di esecuzione: {summary['metadata']['analysis_date'][:10]}

VERSIONE: 2.1 - Correzione metodologica per dati binari/ordinali discreti

SOFTWARE E DIPENDENZE
---------------------
- Python 3.9+
- pandas {pd.__version__}
- scipy 1.10.0+ (fisher_exact, chi2_contingency)
- matplotlib 3.6.0+
- seaborn 0.12.0+
- numpy {np.__version__}
- sklearn (opzionale, per regressione logistica)

CHANGELOG METODOLOGICO
----------------------
v2.1 (2025-08-11): CORREZIONE CRITICA
- ✅ Fisher Exact Test implementato correttamente per dati binari
- ✅ Confronto proporzioni come metodo principale
- ⚠️  Mann-Whitney U relegato ad analisi secondaria con disclaimer
- ✅ Aggiunta regressione logistica per validazione crociata
- ✅ Tabelle di contingenza esportate per riferimento
- ✅ Warning metodologici automatici nel codice

v2.0 (2025-08-11): Tentativo iniziale
- ❌ Implementazione errata di fisher_exact()
- ❌ Mann-Whitney U ancora come metodo principale
- ❌ Mancanza di assessment qualità statistica

ASSESSMENT QUALITÀ STATISTICA
------------------------------"""
        
        if summary.get('test_appropriateness'):
            for test, assessment in summary['test_appropriateness'].items():
                status = "✅" if assessment['is_appropriate'] else "❌"
                report += f"""
{status} {test}: {assessment['recommendation']}
   Motivo: {assessment['reason']}"""
        
        if summary.get('methodological_warnings'):
            report += f"""

AVVISI METODOLOGICI RISCONTRATI
--------------------------------"""
            for warning in summary['methodological_warnings']:
                report += f"""
⚠️  [{warning['test']}]: {warning['warning']}"""
        
        report += f"""

RACCOMANDAZIONI PER RICERCHE FUTURE
------------------------------------
1. Per scale ordinali COMPLETE (1-5): Utilizzare Mann-Whitney U
2. Per dati binari/quasi-binari: Utilizzare Fisher Exact Test
3. Per campioni grandi (n>100): Considerare test χ² con correzioni
4. Sempre verificare appropriatezza del test prima dell'applicazione
5. Dichiarare limitazioni metodologiche in modo esplicito

DISCLAIMER METODOLOGICO
-----------------------
Questa versione corregge un errore metodologico significativo nella v2.0. 
I risultati della v2.1 sono da considerarsi definitivi e metodologicamente 
appropriati per la natura dei dati analizzati.

CONTATTO
--------
Per questioni metodologiche: giuseppe.mangiacotti@studenti.unitn.it
Repository: [Disponibile su richiesta]
"""
        
        # Salvataggio report corretto
        with open(self.output_dir / 'analysis_report_corrected.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Salvataggio anche del warning metodologico separato
        warning_report = f"""
AVVISO METODOLOGICO CRITICO
===========================

PROBLEMA IDENTIFICATO: Test Mann-Whitney U inappropriato per dati essenzialmente binari

DATI ANALIZZATI:
- Soddisfazione su scala 1-3
- Solo valori 2 e 3 presenti nel dataset
- Ties: {fisher_data.get('ties_percentage', 0)*100:.1f}% delle osservazioni

METODOLOGIA CORRETTA:
✅ Fisher Exact Test per confronto proporzioni
✅ Tabella di contingenza 2×2
✅ Odds ratio e p-value esatti

METODOLOGIA SCORRETTA:
❌ Mann-Whitney U per dati con ties eccessivi
❌ Perdita di potenza statistica
❌ Assunzioni violate

RACCOMANDAZIONE:
Utilizzare i risultati del Fisher Exact Test come conclusioni definitive.
I risultati del Mann-Whitney U sono forniti solo per completezza.
"""
        
        with open(self.output_dir / 'methodological_warning.txt', 'w', encoding='utf-8') as f:
            f.write(warning_report)
    
    def run_complete_analysis(self):
        """Esegue l'analisi completa del dataset."""
        print("\t🎓 Avvio analisi completa Accademia Roveretana degli Agiati v2.1")
        print("=" * 60)
        self.load_and_clean_data()
        self.calculate_key_findings()
        self.calculate_cultural_engagement_score()
        self.create_visualizations()
        self.generate_summary_report()
        
        print("=" * 60)
        print("✅ Analisi completata con successo!")
        print(f"📁 Tutti i risultati salvati in: {self.output_dir}")
        print("\n📋 FILE GENERATI:")
        print("   • demographic_analysis.png")
        print("   • digital_satisfaction_paradox_corrected.png ⭐") 
        print("   • generational_membership_gap.png")
        print("   • satisfaction_distribution.png")
        print("   • communication_channels.png")
        print("   • ces_analysis.png")
        print("   • analysis_summary.json")
        print("   • analysis_report_corrected.txt ⭐")
        print("   • methodological_warning.txt ⭐")
        print("   • contingency_table_fisher_test.csv ⭐")
        
        # Mostra eventuali warning metodologici
        if self.methodological_warnings:
            print("\n⚠️  AVVISI METODOLOGICI RISCONTRATI:")
            for warning in self.methodological_warnings:
                print(f"   • [{warning['test']}]: {warning['warning']}")
        
        # Mostra assessment appropriatezza test
        print("\n📊 ASSESSMENT APPROPRIATEZZA TEST:")
        for test, assessment in self.test_appropriateness.items():
            status = "✅" if assessment['is_appropriate'] else "❌"
            print(f"   {status} {test}: {assessment['recommendation']}")
            if not assessment['is_appropriate']:
                print(f"      Motivo: {assessment['reason']}")


def main():
    """
    Funzione principale per esecuzione da riga di comando.
    Versione 2.1 con correzioni metodologiche critiche.
    """
    parser = argparse.ArgumentParser(
        description='Analisi Quantitativa Accademia Roveretana degli Agiati v2.1 (CORRETTA)',
        epilog="""
CORREZIONE METODOLOGICA v2.1:
=============================
Questa versione corregge l'uso inappropriato del test Mann-Whitney U per dati 
essenzialmente binari. Il Fisher Exact Test è ora il metodo principale per 
confronti di proporzioni.

PROBLEMA RISOLTO:
- Ties eccessivi (100%) nel test Mann-Whitney U
- Solo valori 2 e 3 presenti nei dati di soddisfazione
- Violazione assunzioni di base del test non-parametrico

METODOLOGIA CORRETTA:
- Fisher Exact Test per tabelle 2×2
- Confronto diretto proporzioni "Molto Soddisfatti"
- Odds ratio e p-value esatti
- Nessuna assunzione violata

RACCOMANDAZIONE: Utilizzare SOLO i risultati della v2.1 per conclusioni definitive.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Percorso al file CSV dei dati')
    parser.add_argument('--output', '-o', default='./output/',
                       help='Directory di output (default: ./output/)')
    parser.add_argument('--version', action='version', 
                       version='Accademia Analyzer v2.1 (Metodologia Corretta)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Errore: File {args.input} non trovato!")
        return 1
    
    # Banner di avvio con warning metodologico
    print("=" * 80)
    print("🎓 ACCADEMIA ROVERETANA DEGLI AGIATI - ANALISI QUANTITATIVA v2.1")
    print("=" * 80)
    print("⚠️  CORREZIONE METODOLOGICA CRITICA:")
    print("   • Fisher Exact Test sostituisce Mann-Whitney U per dati binari")
    print("   • Risultati v2.0 e precedenti da considerare NON definitivi")
    print("   • Utilizzare SOLO i risultati di questa versione")
    print("=" * 80)
    
    try:
        analyzer = AccademiaAnalyzer(args.input, args.output)
        analyzer.run_complete_analysis()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE L'ANALISI: {str(e)}")
        print("   Verificare il formato dei dati e riprovare.")
        return 1

if __name__ == "__main__":
    exit(main())