import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
import sys

grambank_folder = "./grambank-v1.0.3/grambank-grambank-7ae000c/cldf/"

table_codes = "codes.csv"
table_langs = "languages.csv"
table_params = "parameters.csv"
table_vals = "values.csv"
table_fams = "families.csv"

table_data = "data.csv"
table_param_space = "param_space.csv"
table_corr = "corr_matrix.csv"

top_N_corr = 16
table_top_corr = f"top_{top_N_corr}_corr.csv"
image_top_corr = f"top_{top_N_corr}_corr.png"

out_file = "output.txt"
out_record = False

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)

    chi2, _, _, _ = chi2_contingency(confusion_matrix)

    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def merge_data(table_output_file):
    parameters = pd.read_csv(grambank_folder + table_params)
    parameters = parameters.rename(columns={'ID': 'Parameter_ID',
                                            "Name": "Parameter_Name"})
    parameters.drop(columns=['Description', 'ColumnSpec', 'Patrons', 'Grambank_ID_desc'], inplace=True)

    languages = pd.read_csv(grambank_folder + table_langs)
    languages = languages.rename(columns={'ID': 'Language_ID',
                                          "Name": "Language_Name"})
    languages.drop(columns=["ISO639P3code", "Glottocode", "Family_level_ID", "Language_level_ID",
                            "provenance", "lineage"], inplace=True)

    codes = pd.read_csv(grambank_folder + table_codes)
    codes = codes.rename(columns={'ID': 'Code_ID',
                                  "Description": "Code_Description"})
    codes.drop(columns=["Parameter_ID", "Name"], inplace=True)

    values = pd.read_csv(grambank_folder + table_vals)
    values.drop(columns=["Comment", "Source", "Source_comment", "Coders"], inplace=True)

    data = pd.merge(values, languages, on='Language_ID')
    data = pd.merge(data, parameters, on='Parameter_ID')
    data = pd.merge(data, codes, on='Code_ID')

    #data.drop(columns=["Language_ID", "Parameter_ID", "Code_ID"], inplace=True)

    # print(data.info())
    # print(data.head())
    # print(data.isnull().sum())

    data.to_csv(table_output_file, index=False)

    return data


def param_space(table_output_file):
    values = pd.read_csv(grambank_folder + table_vals)

    param_space = values.pivot(index='Language_ID', columns='Parameter_ID', values='Value')

    param_space = param_space.reset_index()

    languages = pd.read_csv(grambank_folder + table_langs)
    languages = languages.rename(columns={"Name": "Language_Name",
                                          "ID": "Language_ID"})
    languages['Macroarea'] = pd.Categorical(languages['Macroarea'])
    languages['Macroarea_Code'] = languages['Macroarea'].cat.codes
    languages['Family_name'] = pd.Categorical(languages['Family_name'])
    languages['Family_name_Code'] = languages['Family_name'].cat.codes
    languages['level'] = pd.Categorical(languages['level'])
    languages['level_code'] = languages['level'].cat.codes
    languages.drop(columns=["ISO639P3code", "Glottocode", "Family_level_ID", "Language_level_ID",
                            "provenance", "lineage", "Family_name", "Macroarea", "level"], inplace=True)

    data = pd.merge(param_space, languages, on='Language_ID')

    data = data.replace('?', np.nan)
    nan_counts = data.isna().sum()
    data = data[nan_counts.sort_values().index]

    # print(data.info())
    # print(data.isnull().sum())

    data.to_csv(table_output_file, index=False)
    return data


def top_corr(corr_matrix, N=5):
    np.fill_diagonal(corr_matrix.values, 0)
    # print(corr_matrix)

    flat_corr = corr_matrix.values.flatten()

    unique_values, unique_indices = np.unique(flat_corr, return_index=True)
    sorted_indices = unique_indices[np.argsort(unique_values)][::-1]

    selected_indexes = set()
    top_N_entries = []
    for index in sorted_indices:
        row, col = np.unravel_index(index, corr_matrix.shape)
        if row not in selected_indexes and col not in selected_indexes:
            selected_indexes.add(row)
            selected_indexes.add(col)
            top_N_entries.append((row, col, corr_matrix.iloc[row, col]))

            if len(top_N_entries) == N:
                break

    selected_data = [(corr_matrix.index[i], corr_matrix.columns[j], value)
                     for i, j, value in top_N_entries]

    unique_indexes = set()
    for entry in selected_data:
        unique_indexes.add(entry[0])
        unique_indexes.add(entry[1])

    unique_indexes = list(unique_indexes)

    top_corr_natrix = corr_matrix.loc[unique_indexes, unique_indexes]

    return selected_data, top_corr_natrix


def plot_corr(corr_matrix, image_output_file, named=True):
    corr_matrix = corr_matrix.loc[corr_matrix.sum(axis=0).sort_values(ascending=False).index, corr_matrix.sum(axis=1).sort_values(ascending=False).index]

    spaces = [0.3, 0.2, 0.7, 0.8]
    if named:
        params = pd.read_csv(grambank_folder + table_params)
        params.set_index('ID', inplace=True)

        corr_matrix.rename(index=params['Name'], inplace=True)

        spaces = [0.5, 0.2, 0.5, 0.8]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes(spaces)

    im = ax.imshow(corr_matrix, cmap='inferno', interpolation='none')
    cbar = plt.colorbar(im)

    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.index)

    plt.savefig(image_output_file)
    plt.show()


def param_corr(param_space, output_table_file):
    data = param_space.select_dtypes(include=['number'])
    data = data.drop(columns=["Latitude", "Longitude"])

    float_columns = data.select_dtypes(include=['float64'])
    data[float_columns.columns] = float_columns.astype('Int64')

    # print(data)
    # print(data.info())

    corr_matrix = data.corr(method=cramers_v)
    # print(corr_matrix)
    corr_matrix.to_csv(output_table_file)

    return corr_matrix


def corr_by_param(param_id, corr_matrix, top_N=5, named=True):
    correlation_series = corr_matrix.loc[param_id]
    sorted_correlations = correlation_series.sort_values(ascending=False)
    sorted_correlations = sorted_correlations[sorted_correlations >= 0.5]

    if named:
        params = pd.read_csv(grambank_folder + table_params)
        params.set_index('ID', inplace=True)

        sorted_correlations.rename(index=params['Name'], inplace=True)

    return sorted_correlations.head(top_N)


if __name__ == '__main__':
    if out_record:
        output_file = open(out_file, "w")
        sys.stdout = output_file

    data_mix = merge_data(table_data)

    data = pd.read_csv(table_data)
    unique = data.nunique()

    print(f"\t{table_data}\tsize: {len(data.index)}\tcolumns: {data.columns}")
    print(data.info())
    print(f"\t{table_data}\tempty values per column:")
    print(data.isnull().sum())
    print(f"\t{table_data}\tunique values per column:")
    print(unique)

    columns_with_few_unique_values = unique[unique < 50].index
    unique_value_dict = {}
    for column in columns_with_few_unique_values:
        unique_values = data[column].unique()
        counts = data[column].value_counts(dropna=False)
        unique_value_dict[column] = counts.to_dict()
        # print(f"Column '{column}' has {unique[column]} unique values:")
        # print(unique_values)

    for col, values in unique_value_dict.items():
        print(f"Column: {col}\t unique values count")
        for value, count in values.items():
            print(f"\t{value}:\t{count}")

    parameter_space = param_space(table_param_space)



    data = pd.read_csv(table_param_space)

    print(f"\t{table_param_space}\tsize: {len(data.index)}\tcolumns: {data.columns}")
    print(data.info())
    print(f"\t{table_param_space}\tempty values per column:")
    print(data.isnull().sum())
    print(f"\t{table_param_space}\tunique values per column:")
    print(data.nunique())

    corr = param_corr(data, table_corr)




    corr_matrix = pd.read_csv(table_corr, index_col=0)

    print(f"\t{table_corr}\tsize: {len(corr_matrix.index)}\tcolumns: {corr_matrix.columns}")
    print(corr_matrix.info())
    # plot_corr(corr_matrix, "corr_matrix.png", False)

    top_N_entries, top_corr_matrix = top_corr(corr_matrix, top_N_corr)
    top_corr_matrix.to_csv(table_top_corr)




    top_corr_matrix = pd.read_csv(table_top_corr, index_col=0)

    print(f"\t{table_top_corr}\tsize: {len(top_corr_matrix.index)}\tcolumns: {top_corr_matrix.columns}")
    print(top_corr_matrix.info())

    plot_corr(top_corr_matrix, image_top_corr, True)

    params = pd.read_csv(grambank_folder + table_params)
    params.set_index('ID', inplace=True)

    # resolved_entries = []
    # for entry in top_N_entries:
    #     id1, id2, correlation = entry
    #     name1 = params.loc[id1, 'Name'] if id1 in params.index else id1
    #     name2 = params.loc[id2, 'Name'] if id2 in params.index else id2
    #     resolved_entries.append((name1, name2, correlation))
    #
    # #resolved_list = [(params.loc[a]['Name'], params.loc[b]['Name'], corr) for a, b, corr in top_N_entries]
    #
    # for entry in resolved_entries:
    #     print(f' - {entry[0]}\n\t{entry[2]} - correlation\n - {entry[1]}\n')

    for param in top_corr_matrix.index:
        param_name = params.loc[param, 'Name'] if param in params.index else param
        print(f"\nCorrelations for {param}\n\t{param_name}")
        corr_params = corr_by_param(param, corr_matrix, 10, True)
        for corr_param in corr_params.index:
            print(f"{round(corr_params.loc[corr_param], 2)} - {corr_param}")

    if out_record:
        sys.stdout = sys.__stdout__
        output_file.close()
