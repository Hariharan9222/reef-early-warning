let
    FolderPath = Text.TrimEnd(Text.From(DataFolderPath), "\"),
    Source = Csv.Document(File.Contents(FolderPath & "\fact_acoustic.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders, {
        {"site_id", type text},
        {"date_id", Int64.Type},
        {"date", type date},
        {"aci", type number},
        {"adi", type number},
        {"shrimp_band_energy", type number},
        {"fish_band_energy", type number},
        {"acoustic_health_index", type number}
    }),
    RemoveDuplicates = Table.Distinct(ChangeTypes, {"site_id", "date_id"})
in
    RemoveDuplicates
