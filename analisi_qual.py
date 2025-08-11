from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('paper_report_comma.csv', delimiter=";", encoding='utf-8-sig')
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')

def save_open_responses(df):
    # Salva tutte le risposte aperte in un file CSV
    open_responses = df[['Approfondimenti', 'Proposte']].dropna(how='all')
    open_responses.to_csv('open_responses.csv', index=False, encoding='utf-8-sig')
    

def analyze_open_responses(df):
    #creazione metrica engagement: 1 punto per ogni risposta aperta; Min 0, Max 2
    df['has_approfondimenti'] = df['Approfondimenti'].notna()
    df['has_proposte'] = df['Proposte'].notna()
    df['engagement_level'] = df['has_approfondimenti'].astype(int) + df['has_proposte'].astype(int)
    engagement_by_age = df.groupby('Età')['engagement_level'].mean()
    
    # definizione di nuovo profilo: "high engagers" (engagement_level == 2)
    high_engagers = df[df['engagement_level'] == 2]
    
    return engagement_by_age, high_engagers

def extract_themes(responses_series):
    themes = {
        'storia': ['storia', 'storico', 'storica', 'passato'],
        'filosofia': ['filosofia', 'filosofico', 'filosofica', 'pensiero'],
        'scienza': ['scienza', 'scientifico', 'ricerca', 'studio'],
        'attualità': ['attuale', 'moderno', 'contemporaneo', 'oggi']
    }
    
    theme_counts = {}
    for theme, keywords in themes.items():
        count = 0
        for response in responses_series.dropna():
            if any(keyword in response.lower() for keyword in keywords):
                count += 1
        theme_counts[theme] = count
    
    return theme_counts

def categorizza(responses_series):
    categories = {
        'specific_request': [],      
        'general_praise': [],
        'constructive_feedback': [], 
        'no_response': []
    }
    
    for idx, response in responses_series.items():
        if pd.isna(response):
            categories['no_response'].append(idx)
        elif len(response.split()) > 10:  
            categories['constructive_feedback'].append(idx)
        # ... altri criteri
    
    return categories


all_words = []
for response in df['Approfondimenti'].dropna():
    words = re.findall(r'\b\w+\b', response.lower())
    all_words.extend(words)

word_freq = Counter(all_words)
print("Parole più comuni:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}") 

df['engagement_score'] = (
    df['Approfondimenti'].notna().astype(int) + 
    df['Proposte'].notna().astype(int)
)

response_rate_approfondimenti = df['Approfondimenti'].notna().sum() / len(df) * 100
response_rate_proposte = df['Proposte'].notna().sum() / len(df) * 100

avg_length_approfondimenti = df['Approfondimenti'].str.split().str.len().mean()
avg_length_proposte = df['Proposte'].str.split().str.len().mean()

print(f"Percentuale di risposte ad Approfondimenti: {response_rate_approfondimenti:.2f}%")
print(f"Percentuale di risposte a Proposte: {response_rate_proposte:.2f}%")
print(f"Lunghezza media delle risposte ad Approfondimenti: {avg_length_approfondimenti:.2f} parole")
print(f"Lunghezza media delle risposte a Proposte: {avg_length_proposte:.2f} parole")
plt.style.use('seaborn-v0_8-colorblind')

dettagliatori = df[df['Approfondimenti'].str.len() > 50]
print("profili dei rispondenti dettagliati:")
print(dettagliatori[['Età', 'Socio', 'Titolo di studio']].value_counts())

save_open_responses(df)
engagement_by_age, high_engagers = analyze_open_responses(df)
print("Engagement per fascia d'età:")
print(engagement_by_age)
print("High engagers:")
print(high_engagers[['Età', 'Socio', 'Titolo di studio']])
plt.figure(figsize=(12, 8))
sns.barplot(x=engagement_by_age.index, y=engagement_by_age.values, hue=engagement_by_age,palette='viridis', legend=False)
plt.title("Engagement medio per fascia d'età")
plt.xlabel("Età")
plt.ylabel("Livello di Engagement")
plt.savefig('grafici/engagement_per_età.png', dpi=300, bbox_inches='tight')

themes = extract_themes(df['Approfondimenti'].dropna()) 
print("Temi estratti dalle risposte:")
for theme, count in themes.items():
    print(f"{theme}: {count}")