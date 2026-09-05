from dataset.keys import (
    classify_key,
    group_id,
    is_recording_group,
    segment_index,
    split_lang,
)


def test_classify_matches_audit():
    assert classify_key("de__2zsNO2V9K4-00556-00150066-00150154") == "yodas"
    assert classify_key("de_9565_9808_002822") == "mls"
    assert classify_key("en_EN_B00012_S00913_W000420") == "emilia"
    assert classify_key("zh_emilia_zh_0008288409") == "emilia"
    assert classify_key("zh_WenetSpeech4TTS_123") == "wenetspeech4tts"
    assert classify_key("id_train_12-345-6") == "gigaspeech2"
    assert classify_key("pt_M047-0950") == "alcaim"
    assert classify_key("ru_" + "a" * 32) == "golos"
    assert classify_key("fr_abcdefghijk_0007") == "mtedx"
    assert classify_key("pt_M047--0950") == "unknown"


def test_group_ids():
    assert group_id("de__2zsNO2V9K4-00556-00150066-00150154", "yodas") == "_2zsNO2V9K4"
    assert group_id("de_9565_9808_002822", "mls") == "9565"
    assert group_id("en_EN_B00012_S00913_W000420", "emilia") == "EN_B00012_S00913"
    assert group_id("zh_emilia_zh_0008288409", "emilia") is None
    assert group_id("id_train_12-345-6", "gigaspeech2") == "12-345"
    assert group_id("fr_abcdefghijk_0007", "mtedx") == "abcdefghijk"
    assert group_id("pt_M047-0950", "alcaim") == "M047"
    assert group_id("ru_" + "a" * 32, "golos") is None
    assert group_id("zh_WenetSpeech4TTS_123", "wenetspeech4tts") is None


def test_segment_index():
    assert segment_index("de__2zsNO2V9K4-00556-00150066-00150154", "yodas") == 556
    assert segment_index("id_train_12-345-6", "gigaspeech2") == 6
    assert segment_index("fr_abcdefghijk_0007", "mtedx") == 7
    assert segment_index("de_9565_9808_002822", "mls") is None


def test_recording_groups():
    assert is_recording_group("yodas")
    assert is_recording_group("gigaspeech2")
    assert is_recording_group("mtedx")
    assert not is_recording_group("mls")
    assert not is_recording_group("emilia")


def test_split_lang():
    assert split_lang("de_9565_9808_002822") == ("de", "9565_9808_002822")
