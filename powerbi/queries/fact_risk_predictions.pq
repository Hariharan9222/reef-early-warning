let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\fact_risk_predictions.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"site_id", type text},
        {"date_id", Int64.Type},
        {"date", type date},
        {"model_name", type text},
        {"lead_time_days", Int64.Type},
        {"pred_class", Int64.Type},
        {"prob_none", type number},
        {"prob_mild", type number},
        {"prob_severe", type number},
        {"alert_level", type text},
        {"actual_class", Int64.Type}
    }),
    AddAlertSort = Table.AddColumn(ChangeTypes, "alert_sort_order", each
        if [alert_level] = "Bleaching Alert" then 1
        else if [alert_level] = "Bleaching Warning" then 2
        else if [alert_level] = "Bleaching Watch" then 3
        else 4,
        Int64.Type
    ),
    RemoveDuplicates = Table.Distinct(AddAlertSort, {"site_id", "date_id", "lead_time_days"})
in
    RemoveDuplicates
