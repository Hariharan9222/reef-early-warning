let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\model_metrics.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"model_name", type text},
        {"ablation_set", type text},
        {"lead_time", type text},
        {"macro_f1", type number},
        {"pr_auc", type number},
        {"roc_auc", type number},
        {"severe_recall", type number},
        {"brier", type number},
        {"false_alarm_rate", type number}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"model_name", "lead_time"})
in
    RemoveDuplicates
