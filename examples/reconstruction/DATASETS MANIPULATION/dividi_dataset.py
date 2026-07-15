import pandas as pd

# Definisci il valore di X
X = 385 # Sostituisci 400 con il valore che desideri

# 1. Leggi il file CSV originale
# Usiamo sep=';' perché il tuo file usa il punto e virgola come separatore
# Usiamo decimal=',' perché i numeri nel tuo file usano la virgola per i decimali
df = pd.read_csv('QUERY_CSV_j2_3_EXCEL.csv', sep=';', decimal=',')

# 2. Filtra il DataFrame mantenendo SOLO le righe in cui trajectory_id è MINORE di X
# Questo di fatto "elimina" tutte le righe con trajectory_id >= X
df_filtrato = df[df['trajectory_id'] > X]

# 3. Salva il risultato in un nuovo file CSV
df_filtrato.to_csv('QUERY_CSV_j2_3test_EXCEL.csv', sep=';', decimal=',', index=False)

print(f"Filtraggio completato! Il nuovo file contiene {len(df_filtrato)} righe.")