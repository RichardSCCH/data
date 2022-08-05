import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from parseAstJava import parse_java

comments = pd.read_csv('comments.tsv', sep='\t', header=None)
comments.columns = ["id", "com"]

print(comments.shape)

funs = pd.read_json('functions.json', typ='series')
funs = funs.reset_index()
funs.columns = ["id", "fun"]

print(funs.shape)

df_cd = pd.merge(funs, comments, on='id', how='inner')
df_cd = df_cd.set_index("id")

print("df_cd:", df_cd.shape)

def add_missing_closing_tags(text, opening_tag, closing_tag, repl=None):
    if repl is None:
        repl = closing_tag
    otc = text.count(opening_tag)
    ctc = text.count(closing_tag)
    if otc > ctc:
        text += repl*(otc-ctc)
    return text

def replace_whitespaces_comments(text):
    text = re.sub(r"(?<!(public )|(rivate )|(tected ))enum(?![a-z0-9A-Z])", r"enums", text)

    text = re.sub("(\\t)|(//\\n)", "<whitespace_character>", "<newline_character>"+text)
    text = re.sub("(\\n)", "<newline_character><newline_character>", text)
    text = re.sub(r"}\s*;(<newline_character>)+?$", "}<newline_character>", text)
    text = re.sub(r"((<whitespace_character>)|(<newline_character>)|([;}{,)|])|(case.*?:))(<whitespace_character>)*\s*(//.*?<newline_character>)", r"\1 ", text)
    text = re.sub("\s*(//((?!\").)+?<newline_character>)", " ", text)

    text = add_missing_closing_tags(text, "/*", "*/")
    text = re.sub(r"/\*.*?\*/", " ", text)
    text = add_missing_closing_tags(text, "{<newline_character>", r"}((<newline_character>)|( catch)|( else))", "}<newline_character>")
    text = re.sub("(<whitespace_character>)|(<newline_character>)", " ", text)
    return text


def split_single_special_characters(text):
    return re.sub(r"([()\[\]{};.,@_])", r" \1 ", text)


def split_camel_case(text):
    return re.sub(r"(?<=[a-z])([A-Z])", r" \1", text)


def trim_whitespaces(text):
    return str.strip(re.sub("\s+", ' ', text))


def split_combined_special_characters(text):
    return re.sub(r"([<>|&!/*\-+=]+)", r" \1 ", text)

def parse_ast(text):
    try:
        return parse_java(text)
    except:
        return ""

df_cd["fun"] = df_cd["fun"].apply(replace_whitespaces_comments)
df_cd = df_cd.loc[df_cd["fun"].str.strip().str.len() != 0]

print("---create AST---")
print("df_cd:", df_cd.shape)

df_cd["ast"] = df_cd["fun"].apply(parse_ast).apply(lambda x: x.encode('unicode-escape').decode('ascii'))
print("funs head", df_cd.head())
df_cd = df_cd.loc[df_cd["ast"].str.strip().str.len() != 0]


print("---preprocess functions---")
print("df_cd:", df_cd.shape)
df_cd["fun"] = df_cd["fun"].apply(trim_whitespaces)
df_cd["fun_processed"] = df_cd["fun"].apply(lambda s: trim_whitespaces(
    split_combined_special_characters(split_camel_case(split_single_special_characters(s)))))


print("after preprocess - df_cd:", df_cd.shape)

X = df_cd[["fun", "ast", "fun_processed"]]
y = df_cd["com"]

print("---split data---")
print("X size:", X.shape)
print("y size:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

os.mkdir("train")
os.mkdir("test")
os.mkdir("valid")
os.mkdir("dev")

X_train["fun_processed"].to_csv("train/code.original_subtoken", index=False, header=False)
X_train["fun_processed"].to_csv("train/train.spl.src", index=False, header=False)
X_train["fun"].to_csv("train/train.txt.src", index=False, header=False)
X_train["ast"].to_csv("train/train.ast.src", index=False, header=False)
y_train.to_csv("train/javadoc.original", index=False, header=False)
y_train.to_csv("train/train.txt.tgt", index=False, header=False)


X_test["fun_processed"].to_csv("test/code.original_subtoken", index=False, header=False)
X_test["fun_processed"].to_csv("test/test.spl.src", index=False, header=False)
X_test["fun"].to_csv("test/test.txt.src", index=False, header=False)
X_test["ast"].to_csv("test/test.ast.src", index=False, header=False)
y_test.to_csv("test/javadoc.original", index=False, header=False)
y_test.to_csv("test/test.txt.tgt", index=False, header=False)

X_val["fun_processed"].to_csv("dev/code.original_subtoken", index=False, header=False)
X_val["fun_processed"].to_csv("valid/valid.spl.src", index=False, header=False)
X_val["fun"].to_csv("valid/valid.txt.src", index=False, header=False)
X_val["ast"].to_csv("valid/valid.ast.src", index=False, header=False)
y_val.to_csv("dev/javadoc.original", index=False, header=False)
y_val.to_csv("valid/valid.txt.tgt", index=False, header=False)
