import xml.etree.ElementTree as ET
import pandas as pd


file_path = "en_US-uk.tmx"
source_language = "en_US"
target_language = "uk"
data_file_path = "EN-UA_data.csv"
mis_file_path = "EN-UA_mis.txt"


def parse_tmx(file_path, src_lang, tgt_lang):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'xml': 'http://www.w3.org/XML/1998/namespace'}

    data = []
    for tu in root.findall(".//tu"):
        src_text = tu.find(f".//tuv[@xml:lang='{src_lang}']/seg", namespaces=ns).text
        tgt_text = tu.find(f".//tuv[@xml:lang='{tgt_lang}']/seg", namespaces=ns).text
        data.append((src_text, tgt_text))

    df = pd.DataFrame(data, columns=[f"{src_lang}_text", f"{tgt_lang}_text"])
    return df


def count_words(text):
    words = text.split()
    return len(words)


def table_counts(df):
    df[f"{source_language}_len"] = df[f"{source_language}_text"].apply(count_words)
    df[f"{target_language}_len"] = df[f"{target_language}_text"].apply(count_words)
    df["diff_len"] = abs(df[f"{source_language}_len"] - df[f"{target_language}_len"])
    df = df.sort_values('diff_len', ascending=False)

    return df


df = parse_tmx(file_path, source_language, target_language)

print(df)
print(df.info())

df = table_counts(df)
df.to_csv(data_file_path, sep="\t", index=False)

print(df)
print(df.info())

mismatch = df[df["diff_len"] > 1]

print(mismatch)
print(mismatch.info())

mismatch.to_csv(mis_file_path, sep="\t", index=False)





