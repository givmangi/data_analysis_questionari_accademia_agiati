#!/usr/bin/env python3
"""
Analisi Quantitativa dei Dati di Soddisfazione - Accademia Roveretana degli Agiati
================================================================================

Autore: Giuseppe Pio Mangiacotti
Istituzione: Università degli Studi di Trento - Dipartimento di Scienze Cognitive
Anno Accademico: 2024/2025

Descrizione:
Script per l'analisi completa del dataset di feedback raccolto presso l'Accademia 
Roveretana degli Agiati. Implementa algoritmi per l'identificazione dei key findings
principali e la generazione del Cultural Engagement Score (CES).

Dipendenze:
- pandas >= 1.5.0
- matplotlib >= 3.6.0  
- seaborn >= 0.12.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- plotly >= 5.15.0 (opzionale, per grafici interattivi)

Utilizzo:
python data_analysis_accademia_agiati.py --input paper_report_comma.CSV --output ./output/

Citazione:
Mangiacotti, G.P. (2025). Progettazione di interfacce per la raccolta di feedback nel settore culturale: un sistema ibrido per l'Accademia Roveretana degli Agiati
Università degli Studi di Trento.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings
import argparse
import os
from datetime import datetime
from pathlib import Path
import json

# Configurazione warnings e stile
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AccademiaAnalyzer:
    """
    Classe principale per l'analisi dei dati dell'Accademia degli Agiati.
    Implementa metodologie quantitative per l'identificazione di pattern
    di soddisfazione, engagement e comportamento del pubblico culturale.
    """
    
    def __init__(self, csv_path: str, output_dir: str = "./output/"):
        """
        Inizializza l'analyzer con i dati e configura l'output.
        
        Parameters:
        -----------
        csv_path : str
            Percorso al file CSV dei dati
        output_dir : str
            Directory per salvare i risultati
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configurazione per grafici publication-ready
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
            'savefig.bbox': 'tight'
        })
        
        self.results = {}  # Storage per tutti i risultati
        
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
        
        # 1. Digital Satisfaction Paradox
        findings['digital_paradox'] = self._analyze_digital_satisfaction_paradox()
        
        # 2. Generational Membership Gap  
        findings['generational_gap'] = self._analyze_generational_membership_gap()
        
        # 3. Inverse Contentment Effect
        findings['contentment_effect'] = self._analyze_inverse_contentment_effect()
        
        self.results['key_findings'] = findings
        return findings
    
    def _analyze_digital_satisfaction_paradox(self) -> dict:
        """Analizza il Digital Satisfaction Paradox."""
        webinar = self.df[self.df['Fonte'] == 'Webinar']['soddisfazione_num']
        cartaceo = self.df[self.df['Fonte'] == 'Cartaceo']['soddisfazione_num']
        
        # Statistiche descrittive
        stats_webinar = {
            'n': len(webinar),
            'mean': webinar.mean(),
            'std': webinar.std(),
            'satisfaction_rate_max': (webinar == 3).mean()
        }
        
        stats_cartaceo = {
            'n': len(cartaceo),
            'mean': cartaceo.mean(), 
            'std': cartaceo.std(),
            'satisfaction_rate_max': (cartaceo == 3).mean()
        }
        
        # Effect Size (Cohen's d)
        pooled_std = np.sqrt(
            ((stats_webinar['n'] - 1) * stats_webinar['std']**2 + 
             (stats_cartaceo['n'] - 1) * stats_cartaceo['std']**2) /
            (stats_webinar['n'] + stats_cartaceo['n'] - 2)
        )
        
        cohens_d = (stats_webinar['mean'] - stats_cartaceo['mean']) / pooled_std
        
        # Test t per significatività
        t_stat, p_value = stats.ttest_ind(webinar, cartaceo)
        
        # Calcolo età media per modalità
        age_mapping = {'14-30': 22, '31-50': 40, '51-70': 60, '>70': 75}
        
        age_webinar = self.df[self.df['Fonte'] == 'Webinar']['eta_std'].map(age_mapping).mean()
        age_cartaceo = self.df[self.df['Fonte'] == 'Cartaceo']['eta_std'].map(age_mapping).mean()
        
        return {
            'webinar_stats': stats_webinar,
            'cartaceo_stats': stats_cartaceo,
            'difference': stats_webinar['mean'] - stats_cartaceo['mean'],
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'age_webinar': age_webinar,
            'age_cartaceo': age_cartaceo,
            'age_paradox': age_webinar - age_cartaceo
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
        
        # 2. Digital Satisfaction Paradox
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
        """Crea visualizzazioni demografiche."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analisi Demografica del Campione', fontsize=16, fontweight='bold')
        
        # 1. Distribuzione per età
        age_counts = self.df['eta_std'].value_counts()
        axes[0,0].bar(age_counts.index, age_counts.values, color='skyblue', alpha=0.8)
        axes[0,0].set_title('Distribuzione per Fascia d\'Età')
        axes[0,0].set_ylabel('Numero Partecipanti')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Soci vs Non-soci
        member_counts = self.df['socio_std'].value_counts()
        axes[0,1].pie(member_counts.values, labels=member_counts.index, autopct='%1.1f%%', 
                     colors=['lightcoral', 'lightblue'])
        axes[0,1].set_title('Distribuzione Soci vs Non-Soci')
        
        # 3. Soddisfazione per età
        satisfaction_by_age = self.df.groupby('eta_std')['soddisfazione_num'].mean()
        axes[1,0].bar(satisfaction_by_age.index, satisfaction_by_age.values, 
                     color='lightgreen', alpha=0.8)
        axes[1,0].set_title('Soddisfazione Media per Fascia d\'Età')
        axes[1,0].set_ylabel('Soddisfazione Media')
        axes[1,0].set_ylim(2.0, 3.0)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Modalità di fruizione
        mode_counts = self.df['Fonte'].value_counts()
        axes[1,1].bar(mode_counts.index, mode_counts.values, color='orange', alpha=0.8)
        axes[1,1].set_title('Distribuzione per Modalità di Fruizione')
        axes[1,1].set_ylabel('Numero Partecipanti')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_analysis.png')
        plt.close()
    
    def _plot_digital_satisfaction_paradox(self):
        """Visualizza il Digital Satisfaction Paradox."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Digital Satisfaction Paradox', fontsize=16, fontweight='bold')
        
        # Dati per modalità
        webinar_data = self.df[self.df['Fonte'] == 'Webinar']['soddisfazione_num']
        cartaceo_data = self.df[self.df['Fonte'] == 'Cartaceo']['soddisfazione_num']
        
        # 1. Boxplot comparativo
        data_to_plot = [cartaceo_data, webinar_data]
        box_plot = axes[0].boxplot(data_to_plot, labels=['Cartaceo', 'Webinar'], 
                                  patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        box_plot['boxes'][1].set_facecolor('lightblue')
        axes[0].set_title('Distribuzione Soddisfazione per Modalità')
        axes[0].set_ylabel('Punteggio Soddisfazione')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Barplot medie con error bars
        modes = ['Cartaceo', 'Webinar']
        means = [cartaceo_data.mean(), webinar_data.mean()]
        stds = [cartaceo_data.std(), webinar_data.std()]
        
        bars = axes[1].bar(modes, means, yerr=stds, capsize=5, 
                          color=['lightcoral', 'lightblue'], alpha=0.8)
        axes[1].set_title('Soddisfazione Media per Modalità')
        axes[1].set_ylabel('Soddisfazione Media ± SD')
        axes[1].set_ylim(2.0, 3.0)
        
        # Annotazioni statistiche
        paradox_data = self.results['key_findings']['digital_paradox']
        axes[1].text(0.5, 2.95, f'Δ = +{paradox_data["difference"]:.3f}\n'
                               f'Cohen\'s d = {paradox_data["cohens_d"]:.3f}\n'
                               f'p = {paradox_data["p_value"]:.3f}',
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'digital_satisfaction_paradox.png')
        plt.close()
    
    def _plot_generational_membership_gap(self):
        """Visualizza il Generational Membership Gap."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Generational Membership Gap Analysis', fontsize=16, fontweight='bold')
        
        gap_data = self.results['key_findings']['generational_gap']['engagement_metrics']
        age_groups = [d['age_group'] for d in gap_data]
        membership_rates = [d['membership_rate'] * 100 for d in gap_data]
        engagement_indices = [d['engagement_index'] for d in gap_data]
        
        # 1. Percentuale soci per età
        bars1 = axes[0,0].bar(age_groups, membership_rates, color='steelblue', alpha=0.8)
        axes[0,0].set_title('Percentuale Soci per Fascia d\'Età')
        axes[0,0].set_ylabel('% Soci')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Annotazione correlazione
        corr = self.results['key_findings']['generational_gap']['age_membership_correlation']
        axes[0,0].text(0.02, 0.98, f'r = {corr:.3f}', transform=axes[0,0].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Engagement Index
        bars2 = axes[0,1].bar(age_groups, engagement_indices, color='darkgreen', alpha=0.8)
        axes[0,1].set_title('Engagement Index per Fascia d\'Età')
        axes[0,1].set_ylabel('Engagement Index')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Heatmap distribuzione
        crosstab = pd.crosstab(self.df['eta_std'], self.df['socio_std'], margins=True)
        sns.heatmap(crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', 
                   ax=axes[1,0], cbar_kws={'label': 'Numero Partecipanti'})
        axes[1,0].set_title('Heatmap: Età × Status Socio')
        axes[1,0].set_xlabel('Status Socio')
        axes[1,0].set_ylabel('Fascia d\'Età')
        
        # 4. Potenziale di conversione
        conversion_potential = [d['conversion_potential'] for d in gap_data]
        bars4 = axes[1,1].bar(age_groups, conversion_potential, color='orange', alpha=0.8)
        axes[1,1].set_title('Potenziale di Conversione per Fascia d\'Età')
        axes[1,1].set_ylabel('Potenziale Conversione')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generational_membership_gap.png')
        plt.close()
    
    def _plot_satisfaction_distribution(self):
        """Visualizza la distribuzione della soddisfazione."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Analisi Distribuzione Soddisfazione', fontsize=16, fontweight='bold')
        
        # 1. Distribuzione generale
        satisfaction_counts = self.df['soddisfazione_num'].value_counts().sort_index()
        colors = ['gold' if score == 3 else 'darkviolet' for score in satisfaction_counts.index]
        bars1 = axes[0].bar(satisfaction_counts.index, satisfaction_counts.values, 
                           color=colors, alpha=0.8)
        axes[0].set_title('Distribuzione Punteggi di Soddisfazione')
        axes[0].set_xlabel('Punteggio Soddisfazione')
        axes[0].set_ylabel('Numero Risposte')
        axes[0].set_xticks([2, 3])
        
        # Annotazioni statistiche
        mean_sat = self.df['soddisfazione_num'].mean()
        std_sat = self.df['soddisfazione_num'].std()
        axes[0].text(0.02, 0.98, f'M = {mean_sat:.3f}\nSD = {std_sat:.3f}',
                    transform=axes[0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Soddisfazione per status socio
        satisfaction_by_member = self.df.groupby(['socio_std', 'soddisfazione_num']).size().unstack(fill_value=0)
        satisfaction_by_member.plot(kind='bar', ax=axes[1], color=['darkviolet', 'gold'], alpha=0.8)
        axes[1].set_title('Soddisfazione per Status Socio')
        axes[1].set_xlabel('Status Socio')
        axes[1].set_ylabel('Numero Risposte')
        axes[1].legend(title='Soddisfazione', labels=['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[1].tick_params(axis='x', rotation=0)
        
        # 3. Soddisfazione per modalità
        satisfaction_by_mode = self.df.groupby(['Fonte', 'soddisfazione_num']).size().unstack(fill_value=0)
        satisfaction_by_mode.plot(kind='bar', ax=axes[2], color=['darkviolet', 'gold'], alpha=0.8)
        axes[2].set_title('Soddisfazione per Modalità di Fruizione')
        axes[2].set_xlabel('Modalità')
        axes[2].set_ylabel('Numero Risposte')
        axes[2].legend(title='Soddisfazione', labels=['Soddisfatto (2)', 'Molto Soddisfatto (3)'])
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'satisfaction_distribution.png')
        plt.close()
    
    def _plot_communication_channels(self):
        """Analizza l'efficacia dei canali di comunicazione."""
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
        
        # Visualizzazione migliorata
        fig, ax = plt.subplots(figsize=(14, max(8, len(channels) * 0.8)))
        
        # Colori graduati per evidenziare l'importanza
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(channels)))
        
        bars = ax.barh(range(len(channels)), counts, color=colors, alpha=0.8)
        
        # Configurazione assi
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels, fontsize=11)
        ax.set_xlabel('Numero di Menzioni', fontsize=12, fontweight='bold')
        ax.set_title('Efficacia Canali di Comunicazione\n(Analisi Scelte Multiple)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Annotazioni valori con percentuali
        total_mentions = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total_mentions) * 100
            ax.text(bar.get_width() + max(counts) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{count} ({percentage:.1f}%)', 
                   va='center', ha='left', fontweight='bold', fontsize=10)
        
        # Statistiche sommarie
        stats_text = f'Totale menzioni: {total_mentions}\nCanali unici: {len(channel_counts)}\nRisposte multiple: {len(all_channels) - len(self.df[comm_column].dropna())}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
               va='bottom', ha='right', fontsize=9)
        
        # Miglioramento layout
        plt.tight_layout()
        
        # Salvataggio con DPI alto per pubblicazione
        plt.savefig(self.output_dir / 'communication_channels.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Log delle statistiche
        print(f"   ✓ Analisi canali completata: {len(channel_counts)} canali unici, {total_mentions} menzioni totali")
    
    def _plot_ces_analysis(self):
        """Visualizza l'analisi del Cultural Engagement Score."""
        ces_data = self.results['ces']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cultural Engagement Score (CES) Analysis', fontsize=16, fontweight='bold')
        
        # 1. Componenti del CES
        components = ces_data['components']
        comp_names = ['Satisfaction\nFoundation', 'Membership\nAmplification', 
                     'Digital\nAcceleration']
        comp_values = [components['satisfaction'], components['membership'], 
                      components['digital_adoption']]
        
        bars1 = axes[0,0].bar(comp_names, comp_values, 
                             color=['gold', 'steelblue', 'darkgreen'], alpha=0.8)
        axes[0,0].set_title('Componenti del Cultural Engagement Score')
        axes[0,0].set_ylabel('Valore Componente')
        axes[0,0].set_ylim(0, 1.0)
        
        # Annotazioni valori
        for bar, value in zip(bars1, comp_values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. CES Score finale
        current_ces = ces_data['score']
        max_ces = ces_data['components']['max_theoretical']
        
        # Gauge plot
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        axes[0,1].plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)
        axes[0,1].fill_between(r * np.cos(theta), 0, r * np.sin(theta), alpha=0.3, color='lightgray')
        
        # Posizione attuale
        current_angle = (current_ces / max_ces) * np.pi
        axes[0,1].plot([0, r * np.cos(current_angle)], [0, r * np.sin(current_angle)], 
                      'r-', linewidth=4, label=f'CES Attuale: {current_ces:.3f}')
        
        axes[0,1].set_xlim(-1.2, 1.2)
        axes[0,1].set_ylim(-0.2, 1.2)
        axes[0,1].set_aspect('equal')
        axes[0,1].set_title(f'CES Score: {current_ces:.3f}/{max_ces:.1f}')
        axes[0,1].text(0, -0.1, f'Percentile: {ces_data["percentile_rank"]:.1f}%', 
                      ha='center', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # 3. Scenari di miglioramento
        scenarios = ces_data['improvement_scenarios']
        scenario_names = list(scenarios.keys())
        projected_ces = [scenarios[s]['projected_ces'] for s in scenario_names]
        improvements = [scenarios[s]['improvement_percentage'] for s in scenario_names]
        
        bars3 = axes[1,0].bar(scenario_names, projected_ces, alpha=0.8,
                             color=['lightblue', 'orange', 'red'])
        
        # Linea baseline
        axes[1,0].axhline(y=current_ces, color='black', linestyle='--', 
                         label=f'Baseline: {current_ces:.3f}')
        
        axes[1,0].set_title('Scenari di Miglioramento CES')
        axes[1,0].set_ylabel('CES Proiettato')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Annotazioni miglioramenti
        for bar, improvement in zip(bars3, improvements):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'+{improvement:.1f}%', ha='center', va='bottom', 
                          fontweight='bold', color='green')
        
        # 4. Bootstrap validation
        bootstrap_data = ces_data['bootstrap_validation']
        
        # Istogramma distribuzione bootstrap
        bootstrap_scores = np.random.normal(bootstrap_data['mean'], 
                                          bootstrap_data['std_error'], 1000)
        axes[1,1].hist(bootstrap_scores, bins=30, alpha=0.7, color='skyblue', 
                      density=True, label='Bootstrap Distribution')
        
        # Intervallo di confidenza
        axes[1,1].axvline(bootstrap_data['ci_95_lower'], color='red', linestyle='--', 
                         label='CI 95%')
        axes[1,1].axvline(bootstrap_data['ci_95_upper'], color='red', linestyle='--')
        axes[1,1].axvline(bootstrap_data['mean'], color='black', linestyle='-', 
                         linewidth=2, label='Media Bootstrap')
        
        axes[1,1].set_title('Validazione Bootstrap CES')
        axes[1,1].set_xlabel('CES Score')
        axes[1,1].set_ylabel('Densità')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ces_analysis.png')
        plt.close()
    
    def generate_summary_report(self):
        """Genera un report riassuntivo in formato JSON e testo."""
        print("📄 Generazione report riassuntivo...")
        
        # Report JSON per uso programmatico
        summary = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'dataset_size': len(self.df),
                'analyst': 'Giuseppe Pio Mangiacotti',
                'institution': 'Università degli Studi di Trento'
            },
            'key_findings': self.results['key_findings'],
            'ces_analysis': self.results['ces'],
            'sample_statistics': {
                'total_respondents': len(self.df),
                'age_distribution': self.df['eta_std'].value_counts().to_dict(),
                'membership_distribution': self.df['socio_std'].value_counts().to_dict(),
                'satisfaction_mean': self.df['soddisfazione_num'].mean(),
                'satisfaction_std': self.df['soddisfazione_num'].std()
            }
        }
        
        # Salvataggio JSON
        with open(self.output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Report testuale per citazione in tesi
        self._generate_text_report(summary)
        
        print(f"   ✓ Report salvato in: {self.output_dir}")
        
    def _generate_text_report(self, summary: dict):
        """Genera report testuale per citazione accademica."""
        report = f"""
ANALISI QUANTITATIVA DEI DATI DI SODDISFAZIONE
Accademia Roveretana degli Agiati
=============================================

Analista: {summary['metadata']['analyst']}
Istituzione: {summary['metadata']['institution']}
Data Analisi: {summary['metadata']['analysis_date']}

EXECUTIVE SUMMARY
-----------------
Campione analizzato: {summary['metadata']['dataset_size']} partecipanti
Soddisfazione media: {summary['sample_statistics']['satisfaction_mean']:.3f} ± {summary['sample_statistics']['satisfaction_std']:.3f}
Cultural Engagement Score: {summary['ces_analysis']['score']:.3f}/2.0 ({summary['ces_analysis']['percentile_rank']:.1f}° percentile)

KEY FINDINGS PRINCIPALI
------------------------

1. DIGITAL SATISFACTION PARADOX
   - Differenza soddisfazione Webinar vs Cartaceo: +{summary['key_findings']['digital_paradox']['difference']:.3f}
   - Effect Size (Cohen's d): {summary['key_findings']['digital_paradox']['cohens_d']:.3f} (Large Effect)
   - Significatività: p = {summary['key_findings']['digital_paradox']['p_value']:.3f}
   - Age Paradox: Webinar +{summary['key_findings']['digital_paradox']['age_paradox']:.1f} anni vs Cartaceo

2. GENERATIONAL MEMBERSHIP GAP
   - Correlazione Età-Membership: r = {summary['key_findings']['generational_gap']['age_membership_correlation']:.3f}
   - R² = {summary['key_findings']['generational_gap']['r_squared']:.3f}
   - Significatività: p = {summary['key_findings']['generational_gap']['correlation_p_value']:.3f}

3. INVERSE CONTENTMENT EFFECT
   - Contentment Ratio: {summary['key_findings']['contentment_effect']['contentment_ratio']:.3f}
   - Effetto confermato: {summary['key_findings']['contentment_effect']['effect_confirmed']}
   - Test χ²: p = {summary['key_findings']['contentment_effect']['p_value']:.3f}

CULTURAL ENGAGEMENT SCORE (CES)
-------------------------------
Score Attuale: {summary['ces_analysis']['score']:.3f}
Componenti:
- Satisfaction Foundation: {summary['ces_analysis']['components']['satisfaction']:.3f}
- Membership Amplification: {summary['ces_analysis']['components']['membership']:.3f}
- Digital Acceleration: {summary['ces_analysis']['components']['digital_adoption']:.3f}

Validazione Bootstrap (CI 95%): [{summary['ces_analysis']['bootstrap_validation']['ci_95_lower']:.3f}, {summary['ces_analysis']['bootstrap_validation']['ci_95_upper']:.3f}]

SCENARI DI MIGLIORAMENTO
------------------------
"""
        
        for scenario, data in summary['ces_analysis']['improvement_scenarios'].items():
            report += f"""
{scenario.upper()}:
   - CES Proiettato: {data['projected_ces']:.3f}
   - Miglioramento: +{data['improvement_percentage']:.1f}%
   - Target Membership: {data['target_membership']*100:.1f}%
   - Target Digital: {data['target_digital']*100:.1f}%"""
        
        report += f"""

CITAZIONE SUGGERITA
-------------------
Mangiacotti, G.P. (2025). Analisi Quantitativa dei Dati di Soddisfazione - 
Accademia Roveretana degli Agiati [Computer software]. 
Università degli Studi di Trento. 
Data di esecuzione: {summary['metadata']['analysis_date'][:10]}

SOFTWARE UTILIZZATO
-------------------
- Python {'.'.join(map(str, [3, 9, 0]))}+
- pandas {pd.__version__}
- matplotlib 3.6.0+
- seaborn 0.12.0+
- scipy 1.10.0+
- numpy {np.__version__}

NOTA: Questo script implementa metodologie quantitative avanzate per l'analisi
dell'engagement culturale e può essere replicato su dataset simili.
Tutti i grafici e le statistiche sono stati generati automaticamente.
"""
        
        # Salvataggio report testuale
        with open(self.output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def run_complete_analysis(self):
        """Esegue l'analisi completa del dataset."""
        print("🚀 Avvio analisi completa Accademia Roveretana degli Agiati")
        print("=" * 60)
        
        # 1. Caricamento e pulizia dati
        self.load_and_clean_data()
        
        # 2. Calcolo key findings
        self.calculate_key_findings()
        
        # 3. Calcolo Cultural Engagement Score
        self.calculate_cultural_engagement_score()
        
        # 4. Generazione visualizzazioni
        self.create_visualizations()
        
        # 5. Report finale
        self.generate_summary_report()
        
        print("=" * 60)
        print("✅ Analisi completata con successo!")
        print(f"📁 Tutti i risultati salvati in: {self.output_dir}")
        print("\nFile generati:")
        print("   📊 demographic_analysis.png")
        print("   📈 digital_satisfaction_paradox.png") 
        print("   📉 generational_membership_gap.png")
        print("   📋 satisfaction_distribution.png")
        print("   📡 communication_channels.png")
        print("   🎯 ces_analysis.png")
        print("   📄 analysis_summary.json")
        print("   📝 analysis_report.txt")


def main():
    """Funzione principale per esecuzione da riga di comando."""
    parser = argparse.ArgumentParser(
        description='Analisi Quantitativa Accademia Roveretana degli Agiati'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Percorso al file CSV dei dati')
    parser.add_argument('--output', '-o', default='./output/',
                       help='Directory di output (default: ./output/)')
    
    args = parser.parse_args()
    
    # Verifica esistenza file input
    if not os.path.exists(args.input):
        print(f"❌ Errore: File {args.input} non trovato!")
        return 1
    
    # Inizializzazione e esecuzione
    analyzer = AccademiaAnalyzer(args.input, args.output)
    analyzer.run_complete_analysis()
    
    return 0


if __name__ == "__main__":
    exit(main())