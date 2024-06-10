import os
import shutil


def avg(left_value: float, right_value: float, left_num: int, right_num: int) -> float:
    return (left_value * left_num + right_value * right_num) / (left_num + right_num)


def replace_in_filenames(path: str, old_string: str, new_string: str) -> None:
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if old_string in file:
                new_file = file.replace(old_string, new_string)
                os.rename(os.path.join(root, file), os.path.join(root, new_file))
        for _dir in dirs:
            if old_string in _dir:
                new_dir = _dir.replace(old_string, new_string)
                os.rename(os.path.join(root, _dir), os.path.join(root, new_dir))


def add_newline_if_missing(file_path: str):
    # some txt (timeframe) has no \n in the last line and trigger error in depth sensing cpp
    with open(file_path, 'r+') as file:
        file_contents = file.read()
        if file_contents.endswith('\n'):
            return True
        else:
            file.write('\n')
            return False


def fix_timestamp_file(timestamp_file_path: str, json_path: str):
    indices = []
    mss = []
    cnt = 0
    with open(timestamp_file_path, 'r') as f:
        for line in f:
            idx, ms = line.strip().split(',')
            indices.append(int(idx))
            mss.append(int(ms))
            cnt += 1
    json_cnt = 0
    for filename in os.listdir(json_path):
        if not filename.endswith('.json'):
            continue
        json_cnt += 1

    if json_cnt < cnt:
        print('Number of timestamps is more than frames')
        shutil.copy(timestamp_file_path, timestamp_file_path + '.old')
        with open(timestamp_file_path, 'w') as f:
            for i in range(json_cnt):
                f.write(f'{indices[i]},{mss[i]}\n')
    elif json_cnt > cnt:
        print('Number of timestamps is less than frames')
        shutil.copy(timestamp_file_path, timestamp_file_path + '.old')
        with open(timestamp_file_path, 'w') as f:
            for i in range(json_cnt):
                f.write(f'{i + 1},{round(1000 / 30 * (i + 1))}\n')
    else:
        print('Timestamp file is correct')
