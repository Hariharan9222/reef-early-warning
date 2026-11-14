let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\fact_sst.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"site_id", type text},
        {"date_id", Int64.Type},
        {"date", type date},
        {"sst", type number},
        {"ssta", type number},
        {"dhw", type number},
        {"days_above_threshold", type number}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"site_id", "date_id"})
in
    RemoveDuplicates
