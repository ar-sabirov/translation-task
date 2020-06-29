import click
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@click.command()
@click.option('--path', type=str)
def main(**params):
    df = pd.read_csv(params['path'], sep='\t', index_col=0)
    gs_split = GroupShuffleSplit(n_splits=1, train_size=0.8)
    train_indicies, val_indicies = list(gs_split.split(df, groups=df['eng_name']))[0]
    df.iloc[train_indicies].to_csv('train_subset.tsv', sep='\t')
    df.iloc[val_indicies].to_csv('val_subset.tsv', sep='\t')
    
if __name__ == "__main__":
    main()
