import itertools

from amp import app


def test_text_surface_cache_reuses_between_frames():
    cache = app.TextSurfaceCache()
    counter = itertools.count()

    def renderer():
        return f"surface-{next(counter)}"

    cache.start_frame()
    first = cache.fetch("osc1", "freq", (1, 2, 3), "large", renderer)
    cache.finish_frame()

    cache.start_frame()
    second = cache.fetch("osc1", "freq", (1, 2, 3), "large", renderer)
    cache.finish_frame()

    assert first == second
    assert next(counter) == 1


def test_text_surface_cache_discards_stale_text():
    cache = app.TextSurfaceCache()
    counter = itertools.count()

    def renderer():
        return f"surface-{next(counter)}"

    cache.start_frame()
    cache.fetch("osc1", "freq=440", (1, 2, 3), "small", renderer)
    cache.finish_frame()

    cache.start_frame()
    cache.fetch("osc1", "freq=441", (1, 2, 3), "small", renderer)
    cache.finish_frame()

    cache.start_frame()
    cache.fetch("osc1", "freq=440", (1, 2, 3), "small", renderer)
    cache.finish_frame()

    assert next(counter) == 3


def test_text_surface_cache_invalidate_clears_entries():
    cache = app.TextSurfaceCache()
    counter = itertools.count()

    def renderer():
        return f"surface-{next(counter)}"

    cache.start_frame()
    cache.fetch("osc1", "freq=440", (1, 2, 3), "small", renderer)
    cache.finish_frame()

    cache.invalidate("osc1")

    cache.start_frame()
    cache.fetch("osc1", "freq=440", (1, 2, 3), "small", renderer)
    cache.finish_frame()

    assert next(counter) == 2
