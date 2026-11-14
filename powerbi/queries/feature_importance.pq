let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\feature_importance.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"model_name", type text},
        {"feature", type text},
        {"shap_importance", type number},
        {"modality", type text}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"model_name", "feature"})
in
    RemoveDuplicates
