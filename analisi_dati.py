import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('paper_report_comma.csv', delimiter=";", encoding='utf-8-sig')
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')

df = df.dropna(subset=['Soddisfazione (1-3)'])
df['Soddisfazione (1-3)'] = df['Soddisfazione (1-3)'].astype(int)


sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'serif'

# Grafico 1: Distribuzione per Fascia d'Età (Grafico a Barre Orizzontali)
plt.figure(figsize=(10, 6))
age_order = sorted(df['Età'].unique()) 
sns.countplot(y='Età', data=df, order=age_order)
plt.title('Distribuzione dei Partecipanti per Fascia d\'Età', fontsize=16)
plt.xlabel('Numero di Partecipanti', fontsize=12)
plt.ylabel('Fascia d\'Età', fontsize=12)
plt.tight_layout() 
plt.savefig('grafici/età.png') 
plt.close() 

# Grafico 2: Distribuzione per Titolo di Studio (Grafico a Barre Orizzontali)
plt.figure(figsize=(10, 6))
education_order = df['Titolo di studio'].value_counts().index # Ordina per numero di partecipanti
sns.countplot(y='Titolo di studio', data=df, order=education_order)
plt.title('Distribuzione dei Partecipanti per Titolo di Studio', fontsize=16)
plt.xlabel('Numero di Partecipanti', fontsize=12)
plt.ylabel('Titolo di Studio', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/titoli_di_studio.png')
plt.close()

# Grafico 3: Ripartizione Soci vs. Non Soci (Grafico a Torta)
plt.figure(figsize=(8, 8))
df['Socio'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Ripartizione tra Soci e Non Soci', fontsize=16)
plt.ylabel('') # Rimuove l'etichetta dell'asse y
plt.tight_layout()
plt.savefig('grafici/soci_distribuzione.png')
plt.close()

# Grafico 4: Distribuzione dei Punteggi di Soddisfazione (Grafico a Barre Verticali)
plt.figure(figsize=(10, 6))
sns.countplot(x='Soddisfazione (1-3)', data=df, palette='viridis', hue="Soddisfazione (1-3)", legend=False)
plt.title('Distribuzione dei Punteggi di Soddisfazione', fontsize=16)
plt.xlabel('Punteggio di Soddisfazione (1-3)', fontsize=12)
plt.ylabel('Numero di Risposte', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/distribuzione_soddisfazione.png')
plt.close()

# Grafico 5: Canali di Comunicazione (Grafico a Barre Orizzontali)
plt.figure(figsize=(12, 8))
df['Conoscenza_clean'] = df['Conoscenza'].str.split(',').str[0].str.strip()
conoscenza_order = df['Conoscenza_clean'].value_counts().index
sns.countplot(y='Conoscenza_clean', data=df, order=conoscenza_order)
plt.title('Canali di Comunicazione più Efficaci', fontsize=16)
plt.xlabel('Numero di Partecipanti', fontsize=12)
plt.ylabel('Canale di Comunicazione', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/mezzi_di_diffusione.png')
plt.close()

# Grafico 6: Partecipazione per Argomento (Grafico a Barre Orizzontali)
plt.figure(figsize=(12, 8))
argomento_order = df['Argomento'].value_counts().index
sns.countplot(y='Argomento', data=df, order=argomento_order)
plt.title('Partecipazione per Argomento Trattato', fontsize=16)
plt.xlabel('Numero di Partecipanti', fontsize=12)
plt.ylabel('Argomento', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/rgomenti.png')
plt.close()

# Grafico 7: Elementi Positivi (Grafico a Barre Orizzontali)
plt.figure(figsize=(12, 8))
positive_elements_clean = df['Elementi positivi'].str.split(',').explode().str.strip()
positive_elements_clean = positive_elements_clean.dropna()
positive_elements_clean = positive_elements_clean[positive_elements_clean != '']
positive_elements_clean = positive_elements_clean.reset_index(drop=True)
elementi_order = positive_elements_clean.value_counts().index
sns.countplot(y=positive_elements_clean, order=elementi_order)
plt.title('Elementi Positivi più Apprezzati (Seaborn)', fontsize=16)
plt.xlabel('Numero di Menzioni', fontsize=12)
plt.ylabel('Elemento Positivo', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/elementi_positivi_seaborn.png', dpi=300, bbox_inches='tight')
plt.close()

# Grafico 8: soddisfazione per Fascia d'Età e Soci vs Non Soci (Boxplot)
plt.figure(figsize=(12, 8))
sns.boxplot(x='Età', y='Soddisfazione (1-3)', data=df, order=age_order, hue='Socio', palette='Set2')
plt.title('Soddisfazione per Fascia d\'Età', fontsize=16)
plt.xlabel('Fascia d\'Età', fontsize=12)
plt.ylabel('Punteggio di Soddisfazione', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/soddisfazione_età_soci.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Socio', y='Soddisfazione (1-3)', data=df)
plt.title('Soddisfazione: Soci vs. Non Soci', fontsize=16)
plt.xlabel('Status', fontsize=12)
plt.ylabel('Punteggio di Soddisfazione', fontsize=12)
plt.tight_layout()
plt.savefig('grafici/soddisfazione_soci.png')
plt.close()

