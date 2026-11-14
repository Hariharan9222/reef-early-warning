let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\dim_date.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"date_id", Int64.Type},
        {"date", type date},
        {"year", Int64.Type},
        {"month", Int64.Type},
        {"month_name", type text},
        {"week", Int64.Type},
        {"season", type text}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"date_id"})
in
    RemoveDuplicates
