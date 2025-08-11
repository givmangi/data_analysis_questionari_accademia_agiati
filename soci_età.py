import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assumendo che df sia già caricato e pulito come nel codice precedente
df = pd.read_csv('paper_report_comma.csv', delimiter=";", encoding='utf-8-sig')
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')

plt.style.use('seaborn-v0_8-colorblind')

# 1. GRAFICO A BARRE RAGGRUPPATE (Raccomandato per confronti diretti)
plt.figure(figsize=(12, 8))
# Crea una tabella di contingenza
contingency_table = pd.crosstab(df['Età'], df['Socio'])
print("Tabella di contingenza:")
print(contingency_table)

# Grafico a barre raggruppate
contingency_table.plot(kind='bar', figsize=(12, 8), color=['#ff9999', '#66b3ff'])
plt.title('Distribuzione Soci vs Non Soci per Fascia d\'Età', fontsize=18, fontweight='bold')
plt.xlabel('Fascia d\'Età', fontsize=12)
plt.ylabel('Numero di Partecipanti', fontsize=12)
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafici/soci_per_età_barre.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. GRAFICO A BARRE IMPILATE (Mostra proporzioni totali)
plt.figure(figsize=(12, 8))
contingency_table.plot(kind='bar', stacked=True, figsize=(12, 8), 
                      color=['#ff9999', '#66b3ff'])
plt.title('Distribuzione Soci vs Non Soci per Fascia d\'Età (Impilato)', fontsize=18, fontweight='bold')
plt.xlabel('Fascia d\'Età', fontsize=12)
plt.ylabel('Numero di Partecipanti', fontsize=12)
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafici/soci_per_età_impilato.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. GRAFICO A BARRE PERCENTUALI (Mostra proporzioni relative)
plt.figure(figsize=(12, 8))
# Calcola le percentuali per ogni fascia d'età
percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
percentage_table.plot(kind='bar', stacked=True, figsize=(12, 8),
                     color=['#ff9999', '#66b3ff'])
plt.title('Percentuale Soci vs Non Soci per Fascia d\'Età', fontsize=18, fontweight='bold')
plt.xlabel('Fascia d\'Età', fontsize=12)
plt.ylabel('Percentuale (%)', fontsize=12)
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafici/soci_per_età_percentuale.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. HEATMAP (Ottima per visualizzare pattern)
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table.T, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Numero di Partecipanti'})
plt.title('Heatmap: Distribuzione Soci per Fascia d\'Età', fontsize=18, fontweight='bold')
plt.xlabel('Fascia d\'Età', fontsize=12)
plt.ylabel('Status', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/soci_per_età_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. GRAFICO A BARRE ORIZZONTALI (Alternativa per nomi lunghi)
plt.figure(figsize=(12, 8))
contingency_table.plot(kind='barh', figsize=(12, 8), color=['#ff9999', '#66b3ff'])
plt.title('Distribuzione Soci vs Non Soci per Fascia d\'Età (Orizzontale)', fontsize=18, fontweight='bold')
plt.ylabel('Fascia d\'Età', fontsize=12)
plt.xlabel('Numero di Partecipanti', fontsize=12)
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('grafici/soci_per_età_orizzontale.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. ANALISI STATISTICA AGGIUNTIVA
print("\n=== ANALISI STATISTICA ===")

# Calcola statistiche descrittive
total_by_age = contingency_table.sum(axis=1)
soci_percentage = (contingency_table.iloc[:, 1] / total_by_age * 100).round(2)

print("\nPercentuale di soci per fascia d'età:")
for age, pct in soci_percentage.items():
    print(f"{age}: {pct}% soci")

# Test Chi-quadrato per verificare se c'è associazione significativa
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nTest Chi-quadrato:")
print(f"Chi2 statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Gradi di libertà: {dof}")

if p_value < 0.05:
    print("C'è un'associazione significativa tra età e status di socio (p < 0.05)")
else:
    print("Non c'è un'associazione significativa tra età e status di socio (p >= 0.05)")

# 7. GRAFICO COMBINATO CON SUBPLOT
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Barre raggruppate
contingency_table.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff'])
ax1.set_title('Valori Assoluti', fontsize=16, fontweight='bold')
ax1.set_xlabel('Fascia d\'Età')
ax1.set_ylabel('Numero Partecipanti')
ax1.tick_params(axis='x', rotation=45)

# Subplot 2: Barre percentuali
percentage_table.plot(kind='bar', stacked=True, ax=ax2, color=['#ff9999', '#66b3ff'])
ax2.set_title('Percentuali per Fascia', fontsize=16, fontweight='bold')
ax2.set_xlabel('Fascia d\'Età')
ax2.set_ylabel('Percentuale (%)')
ax2.tick_params(axis='x', rotation=45)

# Subplot 3: Heatmap
sns.heatmap(contingency_table.T, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title('Heatmap Distribuzione Soci per Fascia d\'Età', fontsize=16, fontweight='bold')
ax3.set_xlabel('Fascia d\'Età')
ax3.set_ylabel('Status')

# Subplot 4: Distribuzione totale per età
total_by_age.plot(kind='bar', ax=ax4, color='lightgreen')
ax4.set_title('Totale Partecipanti per Età', fontsize=16, fontweight='bold')
ax4.set_xlabel('Fascia d\'Età')
ax4.set_ylabel('Numero Partecipanti')
ax4.tick_params(axis='x', rotation=45)

plt.suptitle('Analisi Completa: Distribuzione Soci per Fascia d\'Età', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('grafici/analisi_completa_soci_età.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nTutti i grafici sono stati salvati nella cartella 'grafici/'")
print(f"Tabelle generate:")
print("- Contingency table (valori assoluti)")
print("- Percentage table (percentuali per fascia d'età)")