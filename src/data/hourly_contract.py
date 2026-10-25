"""Small shared check for positional hourly windows; no Web dependencies."""
import numpy as np


def validate_hourly_timeline(timeline):
    timeline = np.asarray(timeline)
    if (timeline.ndim != 1 or not len(timeline)
            or not np.isfinite(timeline).all()
            or not np.equal(timeline, np.floor(timeline)).all()
            or not np.all(np.diff(timeline) == 1)):
        raise ValueError('Expected unique contiguous integer hours')
