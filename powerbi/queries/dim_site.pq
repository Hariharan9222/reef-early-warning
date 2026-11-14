let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\dim_site.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"site_id", type text},
        {"site_name", type text},
        {"region", type text},
        {"country", type text},
        {"latitude", type number},
        {"longitude", type number},
        {"reef_type", type text}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"site_id"})
in
    RemoveDuplicates
