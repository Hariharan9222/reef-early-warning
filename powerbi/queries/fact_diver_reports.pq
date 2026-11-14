let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\fact_diver_reports.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"report_id", type text},
        {"site_id", type text},
        {"date_id", Int64.Type},
        {"date", type date},
        {"lat", type number},
        {"lon", type number},
        {"depth", type number},
        {"bleaching_percent", type number},
        {"severity_class", type text},
        {"observer_confidence", type number},
        {"source", type text}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"report_id"})
in
    RemoveDuplicates
