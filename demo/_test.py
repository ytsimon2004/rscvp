from neuralib.io.dataset import google_drive_file

with google_drive_file('1dKpZt6eF4szvl7svWRdBQkOfTVLQi4Xg', rename_file='classifier.csv') as f:
    print(f)
