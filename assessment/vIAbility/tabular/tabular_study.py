import pandas as pd

diseases = {
    # --- Tus términos originales (con typos incluidos) ---
    'uforklarlig': 'unexplained',
    'uforkalrlig': 'unexplained',
    'uforkarlig': 'unexplained',
    'ufroklarlig': 'unexplained',
    'ufforklarlig': 'unexplained',
    'uforig': 'unexplained',
    'ufforlig': 'unexplained',
    'ufrolig': 'unexplained',
    'uforlig': 'unexplained',
    'uforklarig': 'unexplained',
    'endometriose': 'endometriosis',
    'mann': 'male',
    'endomertiose': 'endometriosis',
    'tuba': 'tubal',
    'idiopatisk': 'unexplained',
    'ovariell svikt': 'ovarian failure',
    'pco': 'pcos',
    'pcos': 'pcos',
    'asthenozoospermi': 'asthenozoospermia',
    'kvin': 'female',
    'uspesifisert': 'unexplained', 
    'amenorre': 'amenorrhea',
    'antefletert': 'anteverted',
    'uterin': 'uterus',
    'anovulasjon': 'anovulation',
    'psos': 'pcos',
    'sekundær': '',
    'reduserte': '',
    'ovarialreserver': 'ovarian reserve',
    'amonore': 'amenorrhea',
    'alder': 'age',
    'anteflekt': 'anteverted',
    'mi': '',
    'nan': '',
    'faktor': '',
    'factor': '',
    'faktort': '',
    'refert': '',
    'seksuell': '',
    'problem': '',
    'infertilitet': '',
    'serilisert': '',
    'sterilisert': '',
    'forhold': '',
    'inf.': '',
    'infetilitet': '',
    'ukjent årsak': 'unexplained',
    'mulig': '',
    'moderat peritoneal': '', # Rescatado para no perder el grado de la endometriosis
    'klar': '',
    'cholecystectomert': '',   # Colecistectomía (operada de vesícula)
    'annet': '',
    'bilateralt': 'bilateral',                    # Rescatado (es un dato médico crucial para las trompas)
    'sekretorisk endometrium': '',
    'tresjiktet endometrium': '',
    'endometrium lutealfase.': '',
    'uterus m': 'uterus',
    'cyt. 17 m': '',
    'tette eggledere': 'blocked_fallopian_tubes', # Conserva el significado real
    'tubotomi x 2': 'tubal',     # Conserva el antecedente quirúrgico
    'hø ovarium 4-5 folliker 12-14mm': '',
    ' ve ovarium 2 follikler ca.12 mm': '',
    'usikkre celleforandringer': 'ASC-US',
    '3 store folkler (5-6 mindre antral-folikler)': '',
    '4': ''
}


encoding_map = {
    # Ovulatory / Endocrine Factor
    'pcos': 'ovulatory_factor',
    'anovulation': 'ovulatory_factor',
    'amenorrhea': 'ovulatory_factor',
    'amenore': 'ovulatory_factor',
    'hormonstatus': 'ovulatory_factor',
    
    # Tubal Factor
    'tubal': 'tubal_factor',
    'blocked_fallopian_tubes': 'tubal_factor',
    'bilateral': 'tubal_factor',
    
    # Endometriosis Factor
    'endometriosis': 'endometriosis_factor',
    # Uterine / Mechanical Factor
    'myom': 'endometriosis_factor',
    'endometrium': 'endometriosis_factor',
    
    # Male Factor
    'male': 'male_factor',
    'asthenozoospermia': 'male_factor',
    
    # Age / Egg Quality Factor
    'age': 'age_factor',
    'ovarian reserve': 'age_factor',
    
    # Unexplained Factor
    'unexplained': 'unexplained_factor'
}

# Explicitly defining the target columns for the One-Hot Matrix
encoded_columns = [
    'ovulatory_factor', 'tubal_factor', 'endometriosis_factor', 
    'male_factor', 'age_factor', 'uterine_factor', 'unexplained_factor'
]

def clean_diagnose(diagnose):
    final = []
    for d in diagnose:
        d = d.lower().strip()
        d = d.replace('+', '/')
        d = d.replace('og', '/')
        d = d.replace(',', '/')
        list_diseases = d.strip().split('/')
        final.append([])
        if d != '':
            for disease in list_diseases:
                if disease in diseases:
                    if diseases[disease] != '':
                        final[-1].append(diseases[disease])
                else:
                    crop_words = disease.strip().split()
                    found = False
                    for word in crop_words:
                        for key in diseases.keys():
                            if key in word:
                                if diseases[key]!= '':
                                    final[-1].append(diseases[key])
                                found = True
                                break
                    if not found and word != '':
                        final[-1].append(word)
    return final

def one_hot_encode(df, list_diseases):
    encoded_rows = []
    for tags in list_diseases:
        patient_record = {factor: 0 for factor in encoded_columns}
        for tag in tags:
            clean_tag = tag.strip().lower()
            if clean_tag in encoding_map:
                target_col = encoding_map[clean_tag]
                patient_record[target_col] = 1
        encoded_rows.append(patient_record)
    df_diagnoses = pd.DataFrame(encoded_rows)

    df = df.reset_index(drop=True)
    df_diagnoses = df_diagnoses.reset_index(drop=True)
    return pd.concat([df, df_diagnoses], axis=1)