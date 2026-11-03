def build_windows(frame_count, horizon=30, stride=15):
    if horizon < 1 or stride < 1:
        raise ValueError('positive horizon and stride required')
    for start in range(0, frame_count-horizon, stride):
        yield start, start+horizon
