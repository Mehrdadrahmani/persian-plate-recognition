from parkyar import licensing as L

RFC_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")


def test_ed25519_rfc8032_vector():
    assert L.public_key(RFC_SEED).hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    sig = L.sign(RFC_SEED, b"")
    assert sig.hex().startswith("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065")
    assert L.verify(L.public_key(RFC_SEED), b"", sig)
    assert not L.verify(L.public_key(RFC_SEED), b"x", sig)


def test_activation_keys():
    pub = L.public_key(RFC_SEED)
    key = L.make_key(RFC_SEED, "ABCD-0000-1111-2222")
    assert L.key_is_valid(key, "ABCD-0000-1111-2222", pub)
    assert L.key_is_valid(key.lower().replace("-", " "), "ABCD-0000-1111-2222", pub)  # forgiving input
    assert not L.key_is_valid(key, "ABCD-0000-1111-2223", pub)  # another computer
    assert not L.key_is_valid("not a key", "ABCD-0000-1111-2222", pub)


def test_trial_counter(tmp_path):
    lic = L.License(tmp_path / "data", use_registry=False, home=tmp_path / "home")
    assert lic.remaining == L.FREE_RECOGNITIONS and lic.can_recognize()
    for _ in range(L.FREE_RECOGNITIONS):
        lic.consume()
    assert not lic.can_recognize()
    # a new data folder does not reset the trial (the home copy remembers it)
    lic2 = L.License(tmp_path / "other", use_registry=False, home=tmp_path / "home")
    assert lic2.remaining == 0
    # editing the counter file does not help either
    (tmp_path / "home" / ".parkyar_usage").write_text("0:deadbeef")
    (tmp_path / "other" / ".usage").unlink(missing_ok=True)
    assert L.License(tmp_path / "other", use_registry=False, home=tmp_path / "home").remaining == 0
    assert not lic.activate("wrong key")
