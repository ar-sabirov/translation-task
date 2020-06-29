import pandas as pd


def _get_vocab():

    def get_chars(lower, upper):
        low_i = ord(lower)
        up_i = ord(upper)
        return ''.join([chr(i) for i in range(low_i, up_i + 1)])

    bounds = [('а', 'я'), ('a', 'z'), ('0', '9')]

    letters = [' ё_'] + [get_chars(a, b) for a, b in bounds]

    return {a: i for i, a in enumerate(''.join(letters))}


def save_predictions(pred_arr):
    df_res = pd.DataFrame(pred_arr, columns=['answer'], dtype=bool)
    df_res.to_csv('result.tsv', sep='\t', index=None)
