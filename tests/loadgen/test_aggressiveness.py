from loadgen.aggressiveness import DEFAULT_SETTINGS, AggressivenessSetting


def test_default_settings_has_five_points_baseline_first():
    assert len(DEFAULT_SETTINGS) == 5
    assert DEFAULT_SETTINGS[0].name == "baseline"
    assert DEFAULT_SETTINGS[0].sink_len is None
    assert DEFAULT_SETTINGS[0].recent_window is None


def test_default_settings_eviction_points_share_sink_len_64():
    eviction_points = DEFAULT_SETTINGS[1:]
    assert all(s.sink_len == 64 for s in eviction_points)


def test_default_settings_windows_descend_in_aggressiveness_order():
    windows = [s.recent_window for s in DEFAULT_SETTINGS[1:]]
    assert windows == [2048, 1024, 512, 256]


def test_default_settings_names_are_unique():
    names = [s.name for s in DEFAULT_SETTINGS]
    assert len(names) == len(set(names))


def test_aggressiveness_setting_is_a_plain_dataclass_with_expected_fields():
    s = AggressivenessSetting(name="x", sink_len=1, recent_window=2)
    assert (s.name, s.sink_len, s.recent_window) == ("x", 1, 2)
