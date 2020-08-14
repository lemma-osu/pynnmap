import pytest

from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.parser.xml_stand_metadata_parser import XMLStandMetadataParser
from pynnmap.parser.xml_stand_metadata_parser import Flags


@pytest.fixture
def attr_fn():
    return "tests/data/stand_attr.csv"


@pytest.fixture
def attr_xml_fn():
    return "tests/data/stand_attr.xml"


@pytest.fixture
def metadata_parser(attr_xml_fn):
    return XMLStandMetadataParser(attr_xml_fn)


@pytest.fixture
def id_field():
    return "FCID"


@pytest.fixture
def default_obj(attr_fn, metadata_parser, id_field):
    return StandAttributes(attr_fn, metadata_parser, id_field=id_field)


@pytest.fixture
def attr_fn_missing_metadata():
    return "tests/data/stand_attr_missing_metadata.csv"


def test_index(default_obj):
    assert list(default_obj.df.index) == [1, 8, 15, 22]


def test_columns(default_obj):
    expected = [
        "BA_GE_3",
        "TPH_GE_3",
        "VEGCLASS",
        "FORTYPBA",
        "PSME_BA",
        "TPH_PISI_GE_150",
    ]
    assert list(default_obj.df.columns) == expected


def test_num_rows(default_obj):
    assert len(default_obj.df) == 4


def test_incorrect_id_field(attr_fn, metadata_parser):
    with pytest.raises(ValueError):
        _ = StandAttributes(attr_fn, metadata_parser, id_field="FOO")


def test_missing_metadata(attr_fn_missing_metadata, metadata_parser, id_field):
    with pytest.raises(ValueError):
        _ = StandAttributes(
            attr_fn_missing_metadata, metadata_parser, id_field=id_field
        )


def test_flags_continuous_accuracy_attrs(default_obj):
    df = default_obj.get_attr_df(flags=Flags.CONTINUOUS | Flags.ACCURACY)
    expected = ["BA_GE_3", "TPH_GE_3", "PSME_BA", "TPH_PISI_GE_150"]
    assert list(df.columns) == expected


def test_flags_continuous_species_attrs(default_obj):
    df = default_obj.get_attr_df(flags=Flags.CONTINUOUS | Flags.SPECIES)
    expected = ["PSME_BA"]
    assert list(df.columns) == expected


def test_flags_project_accuracy_attrs(default_obj):
    df = default_obj.get_attr_df(flags=Flags.PROJECT | Flags.ACCURACY)
    expected = ["BA_GE_3", "TPH_GE_3", "VEGCLASS", "PSME_BA"]
    assert list(df.columns) == expected


def test_id_flags(default_obj):
    df = default_obj.get_attr_df(flags=Flags.ID)
    expected = ["FCID"]
    assert list(df.columns) == expected
