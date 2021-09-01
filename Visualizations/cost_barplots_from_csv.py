import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import create_df

# Set the name of the CSV to use:
input_file = 'barplot-1'


def name_columns(df):
    n_cols = df.shape[1]
    names = []
    for i in range(n_cols):
        names.append("Cycle "+str(i))
    df.columns = names


def name_rows(df):
    df.rename(lambda i: f"$A_{str(i)}$", axis='index', inplace=True)


plt.close("all")
df = create_df.create_df_from_csv(input_file)
name_columns(df)
name_rows(df)
print(df)
df.plot.bar(stacked=True, width=0.99)
plt.title("Accumulated costs in the different cycles.")
plt.xlabel("Algorithm")
plt.ylabel(r"Accumulated costs in $\gamma\varphi$")
plt.savefig('Plots/'+input+'.png')
plt.show()
