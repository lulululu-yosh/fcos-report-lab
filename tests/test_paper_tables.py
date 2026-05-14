from fcos_report.paper_tables import available_tables, load_table


def test_expected_tables_exist():
    tables = set(available_tables())
    assert {"bpr", "ambiguity", "centerness_ablation", "sota_testdev"}.issubset(tables)


def test_bpr_values_are_in_percent_range():
    df = load_table("bpr")
    assert df["bpr"].between(0, 100).all()
    assert df.loc[df["method"].eq("FCOS") & df["with_fpn"].eq(True), "bpr"].iloc[0] == 98.40
